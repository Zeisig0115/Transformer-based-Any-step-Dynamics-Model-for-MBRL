import torch
import numpy as np
from tqdm import tqdm

from agent import AGENT
from buffer import BUFFER
from components.static_fns import STATICFUNC

from .base_trainer import BASETrainer

class OFFTrainer(BASETrainer):
    """ offline MBRL trainer """

    def __init__(self, args):
        if args.env == "neorl":
            task, data_type, version = tuple(args.env_name.split('-'))
            args.env_name = task + '-' + version
            args.data_type = data_type

        super(OFFTrainer, self).__init__(args)

        # init armpo agent
        task = args.env_name.split('-')[0]
        if args.env == "neorl": task = "neorl-" + task
        if args.env == "maze": task = task + "-" + args.env_name.split('-')[1]
        static_fn = STATICFUNC[task.lower()]
        self.agent = AGENT["admpo"](
            obs_shape=args.obs_shape,
            hidden_dims=args.ac_hidden_dims,
            action_dim=args.action_dim,
            action_space=args.action_space,
            static_fn=static_fn,
            max_arm_step=args.max_arm_step,
            arm_hidden_dim=args.arm_hidden_dim,
            actor_freq=args.actor_freq,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            model_lr=args.model_lr,
            tau=args.tau,
            gamma=args.gamma,
            alpha=args.alpha,
            auto_alpha=args.auto_alpha,
            alpha_lr=args.alpha_lr,
            target_entropy=args.target_entropy,
            penalty_coef=args.penalty_coef,
            deterministic_backup=args.deterministic_backup,
            q_clip=args.q_clip,
            device=args.device
        )
        self.agent.train()

        # lr schedule
        if args.lr_schedule:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.agent.actor_optim, args.n_epochs)
        else:
            self.lr_scheduler = None

        # init replay buffer to store environmental data
        self.memory = BUFFER["seq-sample"](
            buffer_size=args.buffer_size,
            obs_shape=args.obs_shape,
            action_dim=args.action_dim
        )
        rew_bias = 1 if args.env == "maze" else 0
        if args.env == "neorl":
            dataset, _ = self.env.get_dataset(data_type=args.data_type, train_num=1000, need_val=False)
            self.memory.load_neorl_dataset(dataset, rew_bias)
        else:
            self.memory.load_dataset(self.env.get_dataset(), self.env._max_episode_steps, rew_bias)

        # creat memory to store model data
        model_buffer_size = args.rollout_batch_size*args.rollout_length*args.model_retain_epochs
        self.model_memory = BUFFER["vanilla"](
            buffer_size=model_buffer_size,
            obs_shape=args.obs_shape,
            action_dim=args.action_dim
        )

        # other parameters
        self.max_arm_step = args.max_arm_step
        self.rollout_freq = args.rollout_freq
        self.rollout_batch_size = args.rollout_batch_size
        self.rollout_length = args.rollout_length
        self.model_retain_epochs = args.model_retain_epochs
        self.real_ratio = args.real_ratio
        self.n_epochs = args.n_epochs
        self.step_per_epoch = args.step_per_epoch

    def _eval_policy(self):
        """ evaluate policy """
        episode_rewards = []
        for _ in range(self.eval_n_episodes):
            done = False
            episode_rewards.append(0)
            obs = self.eval_env.reset()
            while not done:
                action, _ = self.agent.act(obs, deterministic=True)
                obs, reward, done, _ = self.eval_env.step(action)
                episode_rewards[-1] += reward
        return episode_rewards

    def run(self):
        """ train {args.algo} on {args.env} for {args.n_epochs} epochs"""

        # init
        records = {
            "epoch": [], "loss": {"actor": [], "critic1": [], "critic2": []}, "alpha": [], 
            "reward_mean": [], "reward_std": [], "reward_min": [], "reward_max": [],
            "score_mean": [], "score_std": [], "score_min": [], "score_max": []
        }

        model_loss = self.agent.learn_dynamics_from(self.memory, self.batch_size)
        actor_loss, critic1_loss, critic2_loss, eval_reward, eval_score = [None]*5

        num_steps = 0
        for e in range(self.n_epochs):
            pbar = tqdm(range(self.step_per_epoch), desc="[Epoch {}] Training {} on {}.{} (seed: {})".format(
                e, self.args.algo.upper(), self.args.env.title(), self.args.env_name, self.seed))

            for it in pbar:
                if num_steps % self.rollout_freq == 0:
                    # rollout
                    init_seq_transitions = self.memory.sample_nstep(self.rollout_batch_size, self.max_arm_step-1)
                    fake_transitions = self.agent.rollout(init_seq_transitions, self.rollout_length)
                    self.model_memory.store_batch(**fake_transitions)

                # update policy
                real_sample_size = int(self.batch_size*self.real_ratio)
                fake_sample_size = self.batch_size - real_sample_size
                real_batch = self.memory.sample(batch_size=real_sample_size)
                fake_batch = self.model_memory.sample(batch_size=fake_sample_size)
                transitions = {key: np.concatenate(
                    (real_batch[key], fake_batch[key]), axis=0) for key in real_batch.keys()}
                transitions.pop("timeout")
                learning_info = self.agent.learn(**transitions)
                actor_loss = learning_info["loss"]["actor"]
                critic1_loss = learning_info["loss"]["critic1"]
                critic2_loss = learning_info["loss"]["critic2"]
                alpha = learning_info["alpha"]

                num_steps += 1

                pbar.set_postfix(
                    alpha=alpha,
                    model_loss=model_loss,
                    actor_loss=actor_loss, 
                    critic1_loss=critic1_loss, 
                    critic2_loss=critic2_loss, 
                    eval_reward=eval_reward,
                    eval_score=eval_score
                )

            # update lr
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # evaluate policy
            episode_rewards = self._eval_policy()
            records["epoch"].append(e)
            records["loss"]["actor"].append(actor_loss)
            records["loss"]["critic1"].append(critic1_loss)
            records["loss"]["critic2"].append(critic2_loss)
            records["alpha"].append(alpha)
            records["reward_mean"].append(float(np.mean(episode_rewards)))
            records["reward_std"].append(float(np.std(episode_rewards)))
            records["reward_min"].append(float(np.min(episode_rewards)))
            records["reward_max"].append(float(np.max(episode_rewards)))
            eval_reward = records["reward_mean"][-1]
            
            self.logger.add_scalar("loss/model", model_loss, e)
            self.logger.add_scalar("loss/actor", actor_loss, e)
            self.logger.add_scalar("loss/critic1", critic1_loss, e)
            self.logger.add_scalar("loss/critic2", critic2_loss, e)
            self.logger.add_scalar("alpha", alpha, e)
            self.logger.add_scalar("eval/reward", eval_reward, e)

            if hasattr(self.eval_env, "get_normalized_score"):
                records["score_mean"].append(self.eval_env.get_normalized_score(records["reward_mean"][-1])*100)
                records["score_std"].append(self.eval_env.get_normalized_score(records["reward_std"][-1])*100)
                records["score_min"].append(self.eval_env.get_normalized_score(records["reward_min"][-1])*100)
                records["score_max"].append(self.eval_env.get_normalized_score(records["reward_max"][-1])*100)
                eval_score = self.eval_env.get_normalized_score(eval_reward)*100
                self.logger.add_scalar("eval/score", eval_score, e)

            # save
            self._save(records)

        self.logger.close()

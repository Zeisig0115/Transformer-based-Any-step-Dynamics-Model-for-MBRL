# pip install "gymnasium[mujoco]" torch==2.2.2 tqdm
import gymnasium as gym
import torch, torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import trange
from gym.wrappers import RecordVideo

from agent import SACAgent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHT = Path(r"C:\PyCharm\ADMPO\result\mujoco\Hopper-v5\admpo\25-0504-003707\model\agent_seed-0.pth")  # ← 你的权重路径

# ---------- 1. 环境 ----------
def make_env(record=False, seed=0):
    env = gym.make("Hopper-v5",
                   render_mode="rgb_array" if record else None,
                   max_episode_steps=1000)
    if record:
        env = RecordVideo(
            env, video_folder="videos", episode_trigger=lambda ep: ep == 0,
            name_prefix="SAC_Hopper_demo")
    env.reset(seed=seed)
    return env

env = make_env(record=True, seed=0)

# ---------- 2. 创建 & 加载 Agent ----------
agent = SACAgent(
    obs_shape=env.observation_space.shape[0],
    hidden_dims=(64, 64),                 # ← 填训练时用的 hidden 结构
    action_dim=env.action_space.shape[0],
    action_space=env.action_space,
    actor_freq=1,
    actor_lr=3e-4,
    critic_lr=3e-4,
    device=DEVICE
)

try:
    agent.load_model(WEIGHT)               # 如果权重是完整 save_model 打包
except KeyError:                           # fallback: 只有 actor
    agent.actor.load_state_dict(torch.load(WEIGHT, map_location=DEVICE))
agent.eval()

# ---------- 3. 评测 ----------
def evaluate(policy, n_ep=5):
    returns = []
    for ep in trange(n_ep, desc="Evaluating"):
        obs, _ = env.reset(seed=ep)
        done = False
        ep_ret = 0.0
        while not done:
            action, _ = policy.act(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_ret += reward
        returns.append(ep_ret)
        print(f"Episode {ep}: return = {ep_ret:.1f}")
    print(f"\n平均回报: {np.mean(returns):.1f} ± {np.std(returns):.1f}")
    env.close()

evaluate(agent, n_ep=5)


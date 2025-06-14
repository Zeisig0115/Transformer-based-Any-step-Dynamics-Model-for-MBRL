# import torch
# import torch.nn as nn
# from torch.nn import functional as F
#
# class Swish(nn.Module):
#     def __init__(self):
#         super(Swish, self).__init__()
#
#     def forward(self, x):
#         x = x * torch.sigmoid(x)
#         return x
#
# class ResBlock(nn.Module):
#     def __init__(
#         self,
#         input_dim,
#         output_dim,
#         activation=Swish(),
#         layer_norm=True,
#         with_residual=True,
#         dropout=0.1
#     ):
#         super().__init__()
#
#         self.linear = nn.Linear(input_dim, output_dim)
#         self.activation = activation
#         self.layer_norm = nn.LayerNorm(output_dim) if layer_norm else None
#         self.dropout = nn.Dropout(dropout) if dropout else None
#         self.with_residual = with_residual
#
#     def forward(self, x):
#         y = self.activation(self.linear(x))
#         if self.dropout is not None:
#             y = self.dropout(y)
#         if self.with_residual:
#             y = x + y
#         if self.layer_norm is not None:
#             y = self.layer_norm(y)
#         return y
#
# class ARModel(nn.Module):
#     """ Any-step RNN-based Dynamics Model (ARM) """
#
#     def __init__(
#         self,
#         obs_dim,
#         action_dim,
#         output_dim,
#         hidden_dim=200,
#         rnn_num_layers=3,
#         dropout=0.1,
#         device="cuda:0"
#     ):
#         super().__init__()
#         self.obs_dim = obs_dim
#         self.action_dim = action_dim
#         self.hidden_dim = hidden_dim
#         self.output_dim = output_dim
#         self.device = device
#
#         # rnn with any-step action sequence as input
#         self.rnn_layer = nn.GRU(
#             input_size=obs_dim+action_dim,
#             hidden_size=hidden_dim,
#             num_layers=rnn_num_layers,
#             batch_first=True
#         )
#         # merging layer
#         self.out_layer = nn.Sequential(
#             ResBlock(hidden_dim, hidden_dim, dropout=dropout),
#             ResBlock(hidden_dim, hidden_dim, dropout=dropout),
#             ResBlock(hidden_dim, hidden_dim, dropout=dropout),
#             ResBlock(hidden_dim, hidden_dim, dropout=dropout),
#             nn.Linear(hidden_dim, self.output_dim)
#         )
#
#         self.to(device)
#
#     def forward(self, obs, act_seq, h_state=None):
#         self.rnn_layer.flatten_parameters()
#         obs = obs[:, None].expand(-1, act_seq.shape[1], -1)
#         rnn_in = torch.cat((obs, act_seq), dim=-1)
#         rnn_out, h_state = self.rnn_layer(rnn_in, h_state)
#         rnn_out = rnn_out[:, -1]
#         output = self.out_layer(rnn_out)
#         return output, h_state


import torch
import torch.nn as nn
from torch.nn import functional as F

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


class Swish(nn.Module):
    """Swish activation (x * sigmoid(x))."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class ResBlock(nn.Module):
    """Simple residual MLP block used after the sequence encoder."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: nn.Module | None = None,
        layer_norm: bool = True,
        with_residual: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = activation or Swish()
        self.layer_norm = nn.LayerNorm(output_dim) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.with_residual = with_residual and (input_dim == output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.activation(self.linear(x))
        if self.dropout is not None:
            y = self.dropout(y)
        if self.with_residual:
            y = x + y
        if self.layer_norm is not None:
            y = self.layer_norm(y)
        return y


class ARModel(nn.Module):
    """GRU + attention dynamics model for model‑based RL.

    Workflow (batch size = B, window length = T):
        1) Repeat obs k=T times and concatenate with act_seq  →  (B, T, obs_dim+act_dim)
        2) Encode with multi‑layer GRU → rnn_out  (B, T, hidden_dim)
        3) Use **dot‑product attention pooling** with learnable query to get a global
           context vector c  (B, hidden_dim)
        4) Feed c through several ResBlocks + Linear → predict (next_state, reward)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        rnn_num_layers: int = 3,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        device: str | torch.device = "cuda:0",
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 1. GRU encoder
        self.rnn_layer = nn.GRU(
            input_size=obs_dim + action_dim,
            hidden_size=hidden_dim,
            num_layers=rnn_num_layers,
            batch_first=True,
        )

        # 2. Multi‑Head Attention pooling: query is a learned parameter (1, 1, H)
        self.num_heads = num_attention_heads
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            batch_first=True,
            dropout=dropout,
        )

        # 3. Prediction head
        self.head = nn.Sequential(
            ResBlock(hidden_dim, hidden_dim, dropout=dropout),
            ResBlock(hidden_dim, hidden_dim, dropout=dropout),
            ResBlock(hidden_dim, hidden_dim, dropout=dropout),
            ResBlock(hidden_dim, hidden_dim, dropout=dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        self.to(device)

    # ---------------------------------------------------------------------
    # forward
    # ---------------------------------------------------------------------
    def forward(
        self,
        obs: torch.Tensor,          # (B, obs_dim)
        act_seq: torch.Tensor,      # (B, T, action_dim)
        h_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode a window and predict next state & reward.

        Returns
        -------
        output : Tensor
            (B, output_dim)
        h_state : Tensor | None
            Final hidden state of the GRU; can be passed to next call for efficiency.
        """
        B, T, _ = act_seq.size()

        # Step 1: build RNN input  (B, T, obs_dim + action_dim)
        obs_seq = obs.unsqueeze(1).expand(-1, T, -1)
        rnn_in = torch.cat((obs_seq, act_seq), dim=-1)

        # Step 2: GRU encoding  →  rnn_out: (B, T, H)
        self.rnn_layer.flatten_parameters()
        rnn_out, h_state = self.rnn_layer(rnn_in, h_state)

        # Step 3: Attention pooling
        # Broadcast learned query to batch → (B, 1, H)
        query = self.query.expand(B, -1, -1)
        # attn_out: (B, 1, H)
        attn_out, _ = self.attn(query=query, key=rnn_out, value=rnn_out, need_weights=False)
        context = attn_out.squeeze(1)  # (B, H)

        # Step 4: Prediction
        output = self.head(context)
        return output, h_state



# import math
# import torch
# import torch.nn as nn
#
#
# class Swish(nn.Module):
#     def __init__(self):
#         super(Swish, self).__init__()
#
#     def forward(self, x):
#         return x * torch.sigmoid(x)
#
#
# class ResBlock(nn.Module):
#     def __init__(
#         self,
#         input_dim,
#         output_dim,
#         activation=None,
#         layer_norm=True,
#         with_residual=True,
#         dropout=0.1
#     ):
#         super().__init__()
#         self.linear = nn.Linear(input_dim, output_dim)
#         self.activation = activation or Swish()
#         self.layer_norm = nn.LayerNorm(output_dim) if layer_norm else None
#         self.dropout = nn.Dropout(dropout) if dropout else None
#         self.with_residual = with_residual
#
#     def forward(self, x):
#         y = self.activation(self.linear(x))
#         if self.dropout is not None:
#             y = self.dropout(y)
#         if self.with_residual:
#             y = x + y
#         if self.layer_norm is not None:
#             y = self.layer_norm(y)
#         return y
#
#
# class LearnablePositionalEncoding(nn.Module):
#     """
#     Learnable absolute positional encoding using a parameter tensor.
#     """
#     def __init__(self, d_model, max_len=500):
#         super().__init__()
#         self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
#         nn.init.trunc_normal_(self.pe, std=0.02)
#         self.dropout = nn.Dropout(0.1)
#
#     def forward(self, x):
#         L = x.size(1)
#         return self.dropout(x + self.pe[:, :L, :])
#
#
# class ARModel(nn.Module):
#     """
#     Sequence model using Transformer Encoder, combining obs and act_seq per time step.
#     """
#     def __init__(
#         self,
#         obs_dim,
#         action_dim,
#         output_dim,
#         d_model=64,
#         nhead=4,
#         num_layers=3,
#         dim_feedforward=256,
#         dropout=0.1,
#         max_seq_len=5,
#         device="cuda:0"
#     ):
#         super().__init__()
#         self.device = device
#         self.obs_dim = obs_dim
#         self.action_dim = action_dim
#
#         # Project (obs + act) into model dimension
#         self.input_proj = nn.Linear(obs_dim + action_dim, d_model)
#
#         # Learnable positional encoding
#         self.pos_enc = LearnablePositionalEncoding(d_model, max_len=max_seq_len)
#
#         # Transformer Encoder
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             activation='relu',
#             batch_first=True
#         )
#         self.transformer_encoder = nn.TransformerEncoder(
#             encoder_layer,
#             num_layers=num_layers,
#             norm=nn.LayerNorm(d_model)
#         )
#
#         # Output MLP head
#         self.mlp_head = nn.Sequential(
#             ResBlock(d_model, d_model, dropout=dropout),
#             ResBlock(d_model, d_model, dropout=dropout),
#             ResBlock(d_model, d_model, dropout=dropout),
#             ResBlock(d_model, d_model, dropout=dropout),
#             nn.Linear(d_model, output_dim)
#         )
#
#         self.to(device)
#
#     def forward(self, obs, act_seq):
#         """
#         Args:
#             obs:      Tensor of shape [B, obs_dim]
#             act_seq:  Tensor of shape [B, T, action_dim]
#         Returns:
#             out:      Tensor of shape [B, output_dim]
#         """
#         B, T, _ = act_seq.size()
#
#         # 1. Expand obs to [B, T, obs_dim]
#         obs_seq = obs[:, None, :].expand(-1, T, -1)
#
#         # 2. Concatenate obs_seq with act_seq → [B, T, obs_dim + action_dim]
#         x = torch.cat([obs_seq, act_seq], dim=-1)
#
#         # 3. Input projection → [B, T, d_model]
#         x = self.input_proj(x)
#
#         # 4. Add positional encoding
#         x = self.pos_enc(x)
#
#         # 5. Transformer encoding
#         enc_out = self.transformer_encoder(x)  # [B, T, d_model]
#
#         # 6. Use last time step for prediction
#         last = enc_out[:, -1, :]  # [B, d_model]
#
#         # 7. Output prediction
#         out = self.mlp_head(last)  # [B, output_dim]
#         return out, None


import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.bias = nn.Parameter(torch.zeros(1, 1, num_features))

    def normalize(self, x):
        mean = x.mean(1, keepdim=True)
        std = (x.var(1, keepdim=True, unbiased=False) + self.eps).sqrt()
        x_norm = (x - mean) / std
        if self.affine:
            x_norm = x_norm * self.weight + self.bias
        return x_norm, (mean, std)

    def denormalize(self, y, stats):
        mean, std = stats
        return y * std + mean


class SqrtProjection(nn.Module):
    def __init__(self, in_size, out_size, bias=False):
        super().__init__()
        self.sqrt_in = math.ceil(math.sqrt(in_size))
        self.sqrt_out = math.ceil(math.sqrt(out_size))
        self.pad_in = self.sqrt_in**2 - in_size
        self.out_size = out_size
        self.T1 = nn.Linear(self.sqrt_in, self.sqrt_out, bias=bias)
        self.T2 = nn.Linear(self.sqrt_in, self.sqrt_out, bias=bias)

    def forward(self, x):
        prefix_shape = x.shape[:-1]
        if self.pad_in > 0:
            x = F.pad(x, (0, self.pad_in))
        x = x.reshape(*prefix_shape, self.sqrt_in, self.sqrt_in)
        x = self.T1(x).transpose(-1, -2)
        x = self.T2(x).transpose(-1, -2)
        return x.reshape(*prefix_shape, -1)[..., :self.out_size]


class HAMSTARBlock(nn.Module):
    def __init__(self, d_model, num_periods, n_steps=3, dropout=0.1):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("HAMSTARBlock requires an even d_model.")

        self.d_half = d_model // 2
        self.K = num_periods
        self.n_steps = n_steps

        self.log_eta = nn.Parameter(torch.tensor(-2.3))

        self.L_lower = nn.Parameter(torch.randn(num_periods, num_periods) * 0.01)
        self.L_diag_raw = nn.Parameter(torch.full((num_periods,), -2.3))

        self.u_raw = nn.Parameter(torch.randn(num_periods) * 0.01)
        self.v_adapt = nn.Linear(self.d_half, 1, bias=True)

        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _get_A_base(self):
        mask = torch.tril(
            torch.ones(
                self.K,
                self.K,
                device=self.L_lower.device,
                dtype=self.L_lower.dtype,
            ),
            diagonal=-1,
        )
        lower = self.L_lower * mask
        diag = F.softplus(self.L_diag_raw)
        L = lower + torch.diag(diag)
        return L @ L.t()

    def _build_transition(self, A_eff, eta):
        batch_size = A_eff.shape[0]
        K = self.K
        eye = torch.eye(K, device=A_eff.device, dtype=A_eff.dtype)
        eye = eye.unsqueeze(0).expand(batch_size, -1, -1)
        eta2 = eta**2

        M_qq = eye - (eta2 / 2) * A_eff
        M_qp = eta * eye
        M_pq = -eta * torch.bmm(A_eff, eye - (eta2 / 4) * A_eff)
        M_pp = eye - (eta2 / 2) * A_eff

        top = torch.cat([M_qq, M_qp], dim=-1)
        bottom = torch.cat([M_pq, M_pp], dim=-1)
        return torch.cat([top, bottom], dim=-2)

    def forward(self, x):
        batch_size, K, _ = x.shape
        q = x[..., :self.d_half]
        p = x[..., self.d_half:]

        eta = F.softplus(self.log_eta).clamp(max=0.5)
        A_base = self._get_A_base()

        q_summary = q.mean(dim=1)
        alpha = F.softplus(self.v_adapt(q_summary)).view(batch_size, 1, 1)

        u = self.u_raw / (self.u_raw.norm() + 1e-6)
        u_col = u.view(1, K, 1)
        uu_t = torch.bmm(
            u_col.expand(batch_size, -1, -1),
            u_col.transpose(-1, -2).expand(batch_size, -1, -1),
        )
        A_eff = A_base.unsqueeze(0) + alpha * uu_t

        if K > 32 or batch_size * K * K > 20000000:
            for _ in range(self.n_steps):
                fq = torch.matmul(A_eff, q)
                p = p - (eta / 2) * fq
                q = q + eta * p
                fq = torch.matmul(A_eff, q)
                p = p - (eta / 2) * fq
        else:
            transition = self._build_transition(A_eff, eta)
            transition_n = torch.linalg.matrix_power(transition, self.n_steps)
            state = torch.cat([q, p], dim=1)
            state = torch.bmm(transition_n, state)
            q = state[:, :K, :]
            p = state[:, K:, :]

        y = torch.cat([q, p], dim=-1)
        y = self.proj(y)
        return self.norm(self.dropout(y) + x)


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = getattr(configs, "period_len", 24)

        self.latent = getattr(configs, "emb_num", None)
        if self.latent is None:
            self.latent = getattr(configs, "emb_numb", None)
        if self.latent is None:
            self.latent = getattr(configs, "basis_num", 128)
        self.latent = int(self.latent)
        if self.latent % 2 != 0:
            self.latent += 1

        self.e_layers = getattr(configs, "e_layers", 2)
        self.dropout = getattr(configs, "dropout", 0.1)
        self.n_steps = getattr(configs, "n_steps", 6)

        self.num_periods_in = (self.seq_len + self.period_len - 1) // self.period_len
        self.num_periods_out = (self.pred_len + self.period_len - 1) // self.period_len

        self.use_revin = bool(getattr(configs, "use_revin", True))
        if self.use_revin:
            self.revin = RevIN(self.enc_in)

        self.trend_proj = SqrtProjection(self.seq_len, self.pred_len)
        self.embed = nn.Linear(self.period_len, self.latent)
        self.pos_embed = nn.Parameter(
            torch.randn(1, 1, self.num_periods_in, self.latent) * 0.02
        )

        self.layers = nn.ModuleList(
            [
                HAMSTARBlock(
                    self.latent,
                    self.num_periods_in,
                    n_steps=self.n_steps,
                    dropout=self.dropout,
                )
                for _ in range(self.e_layers)
            ]
        )

        self.block_decoder = nn.Linear(self.num_periods_in, self.num_periods_out)
        self.predictor = nn.Linear(self.latent, self.period_len)
        self.channel_scale = nn.Parameter(torch.ones(self.enc_in))

    def forward(self, x):
        batch_size, seq_len, num_channels = x.shape

        if self.use_revin:
            x, stats = self.revin.normalize(x)
        else:
            stats = None

        x_trend = F.avg_pool1d(
            x.permute(0, 2, 1),
            kernel_size=self.period_len,
            stride=1,
            padding=self.period_len // 2,
        )
        if x_trend.shape[2] != seq_len:
            x_trend = x_trend[..., :seq_len]
        trend_pred = self.trend_proj(x_trend).permute(0, 2, 1)

        x_seasonal = x - x_trend.permute(0, 2, 1)
        x_channel_first = x_seasonal.permute(0, 2, 1)

        pad_len = self.num_periods_in * self.period_len - seq_len
        if pad_len > 0:
            x_channel_first = F.pad(x_channel_first, (0, pad_len), mode="circular")

        x_periods = x_channel_first.view(
            batch_size,
            num_channels,
            self.num_periods_in,
            self.period_len,
        )

        z = self.embed(x_periods) + self.pos_embed
        z = z.view(batch_size * num_channels, self.num_periods_in, self.latent)

        for layer in self.layers:
            z = layer(z)

        z = z.transpose(1, 2)
        z = self.block_decoder(z).transpose(1, 2)

        y_seasonal = self.predictor(z)
        y_seasonal = y_seasonal.view(batch_size, num_channels, -1)
        y_seasonal = y_seasonal[:, :, : self.pred_len].permute(0, 2, 1)

        y_hat = y_seasonal * self.channel_scale.view(1, 1, -1) + trend_pred

        if self.use_revin:
            y_hat = self.revin.denormalize(y_hat, stats)

        return y_hat

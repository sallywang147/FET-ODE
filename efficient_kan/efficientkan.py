import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LogisticBasis(nn.Module):
    """
    Per-input-dimension logistic basis:
      a, b : (in_dim, num_basis)
      forward(x): (B, in_dim, num_basis)
    """
    def __init__(self, in_dim: int, num_basis: int):
        super().__init__()
        self.in_dim = in_dim
        self.num_basis = num_basis
        self.a = nn.Parameter(torch.randn(in_dim, num_basis))
        self.b = nn.Parameter(torch.randn(in_dim, num_basis))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim)
        assert x.dim() == 2 and x.size(1) == self.in_dim
        x = x.unsqueeze(-1)  # (B, in_dim, 1)
        return 2.0 / (1.0 + torch.exp(-self.a * (x - self.b)))  # (B, in_dim, num_basis)


class KANLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],

        # --- NEW: logistic basis branch ---
        enable_logistic_basis=True,
        num_basis=10,
        scale_logistic=1.0,
        enable_standalone_scale_logistic=True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # ---- spline grid ----
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        # ---- original KAN weights ----
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, grid_size + spline_order))
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        # ---- NEW: logistic basis branch ----
        self.enable_logistic_basis = enable_logistic_basis
        self.num_basis = num_basis
        self.scale_logistic = scale_logistic
        self.enable_standalone_scale_logistic = enable_standalone_scale_logistic

        if self.enable_logistic_basis:
            self.logistic_basis = LogisticBasis(in_features, num_basis)
            # Weight from (in_features * num_basis) -> out_features
            self.logistic_weight = nn.Parameter(torch.empty(out_features, in_features * num_basis))
            if self.enable_standalone_scale_logistic:
                self.logistic_scaler = nn.Parameter(torch.empty(out_features))  # simple per-output scale

        self.reset_parameters()

    def reset_parameters(self):
        # base
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)

        # spline
        with torch.no_grad():
            noise = (
                (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5)
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(self.grid.T[self.spline_order : -self.spline_order], noise)
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

        # logistic
        if self.enable_logistic_basis:
            # Initialize logistic_weight small-ish (similar spirit to adding a new branch)
            nn.init.kaiming_uniform_(self.logistic_weight, a=math.sqrt(5) * self.scale_logistic)
            if self.enable_standalone_scale_logistic:
                nn.init.ones_(self.logistic_scaler)  # start at 1.0

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid = self.grid  # (in, grid_size + 2*spline_order + 1)
        x = x.unsqueeze(-1)  # (B, in, 1)

        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:(-k)]) * bases[:, :, 1:]
            )

        assert bases.size() == (x.size(0), self.in_features, self.grid_size + self.spline_order)
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(0, 1)  # (in, B, coeff)
        B = y.transpose(0, 1)                  # (in, B, out)
        solution = torch.linalg.lstsq(A, B).solution  # (in, coeff, out)
        result = solution.permute(2, 0, 1)            # (out, in, coeff)

        assert result.size() == (self.out_features, self.in_features, self.grid_size + self.spline_order)
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (self.spline_scaler.unsqueeze(-1) if self.enable_standalone_scale_spline else 1.0)

    @property
    def scaled_logistic_weight(self):
        if not self.enable_logistic_basis:
            return None
        w = self.logistic_weight
        # global scale + optional per-output scale
        w = w * self.scale_logistic
        if self.enable_standalone_scale_logistic:
            w = w * self.logistic_scaler.unsqueeze(-1)
        return w

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x2d = x.reshape(-1, self.in_features)  # (B*, in)

        # base + spline (original)
        base_output = F.linear(self.base_activation(x2d), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x2d).view(x2d.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        out = base_output + spline_output

        # NEW: logistic basis branch
        if self.enable_logistic_basis:
            phi = self.logistic_basis(x2d)              # (B*, in, num_basis)
            phi_flat = phi.reshape(x2d.size(0), -1)     # (B*, in*num_basis)
            logistic_out = F.linear(phi_flat, self.scaled_logistic_weight)  # (B*, out)
            out = out + logistic_out

        out = out.reshape(*original_shape[:-1], self.out_features)
        return out

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        # unchanged from your version (kept as-is)
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(1, 0, 2)  # (batch, in, out)

        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device)
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(self.grid_size + 1, dtype=torch.float32, device=x.device).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1] - uniform_step * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:] + uniform_step * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0, regularize_logistic_l1=0.0):
        """
        Original spline regularization + optional logistic L1.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        reg_act = l1_fake.sum()
        p = l1_fake / (reg_act + 1e-12)
        reg_ent = -torch.sum(p * (p + 1e-12).log())

        reg = regularize_activation * reg_act + regularize_entropy * reg_ent

        if self.enable_logistic_basis and regularize_logistic_l1 != 0.0:
            reg = reg + regularize_logistic_l1 * self.logistic_weight.abs().mean()

        return reg


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

# ---------------- LogisticBasis ----------------
class LogisticBasis(nn.Module):
    def __init__(self, in_dim: int, num_basis: int):
        super().__init__()
        self.in_dim = in_dim
        self.num_basis = num_basis
        self.a = nn.Parameter(torch.randn(in_dim, num_basis) * 0.2)
        self.b = nn.Parameter(torch.randn(in_dim, num_basis) * 0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2 and x.size(1) == self.in_dim
        x = x.unsqueeze(-1)  # (B, in_dim, 1)
        return 2.0 / (1.0 + torch.exp(-self.a * (x - self.b)))  # (B, in_dim, num_basis)

# ---------------- KANLinear (+ logistic branch) ----------------
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
        grid_range=(-1.0, 1.0),

        use_logistic_basis=True,
        num_basis=10,
        scale_logistic=1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, grid_size + spline_order))

        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.use_logistic_basis = use_logistic_basis
        self.num_basis = num_basis
        self.scale_logistic = scale_logistic

        if self.use_logistic_basis:
            self.logistic_basis = LogisticBasis(in_features, num_basis)
            self.logistic_weight = nn.Parameter(torch.empty(out_features, in_features * num_basis))
            self.logistic_bias = nn.Parameter(torch.zeros(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
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

        if self.use_logistic_basis:
            nn.init.kaiming_uniform_(self.logistic_weight, a=math.sqrt(5) * self.scale_logistic)
            nn.init.zeros_(self.logistic_bias)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            left = (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)])
            right = (grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:(-k)])
            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        A = self.b_splines(x).transpose(0, 1)  # (in, batch, coeff)
        B = y.transpose(0, 1)                  # (in, batch, out)
        solution = torch.linalg.lstsq(A, B).solution
        return solution.permute(2, 0, 1).contiguous()

    @property
    def scaled_spline_weight(self):
        if self.enable_standalone_scale_spline:
            return self.spline_weight * self.spline_scaler.unsqueeze(-1)
        return self.spline_weight

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        orig = x.shape
        x2 = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x2), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x2).view(x2.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        out = base_output + spline_output

        if self.use_logistic_basis:
            phi = self.logistic_basis(x2).reshape(x2.size(0), -1)
            out = out + F.linear(phi, self.logistic_weight, self.logistic_bias)

        return out.reshape(*orig[:-1], self.out_features)

# ---------------- Kuramoto Front-End (local coupling) ----------------
class Kuramoto2D(nn.Module):
    """
    Treat each pixel as an oscillator phase theta.
    Update:
      theta <- theta + dt * (omega + K * sum_neighbors sin(theta_neighbor - theta))
    Using 4-neighborhood via conv kernels.
    """
    def __init__(self, H=28, W=28, steps=10, dt=0.15, learn_K=True, learn_omega=True):
        super().__init__()
        self.H, self.W = H, W
        self.steps = steps
        self.dt = dt

        # Learnable scalar coupling strength
        self.K = nn.Parameter(torch.tensor(0.5)) if learn_K else torch.tensor(0.5)

        # Optional learnable natural frequency field (1,1,H,W)
        if learn_omega:
            self.omega = nn.Parameter(torch.zeros(1, 1, H, W))
        else:
            self.register_buffer("omega", torch.zeros(1, 1, H, W))

        # 4-neighbor kernel (fixed)
        k = torch.zeros(1, 1, 3, 3)
        k[0, 0, 0, 1] = 1.0
        k[0, 0, 2, 1] = 1.0
        k[0, 0, 1, 0] = 1.0
        k[0, 0, 1, 2] = 1.0
        self.register_buffer("neighbor_kernel", k)

    def forward(self, x_img: torch.Tensor) -> torch.Tensor:
        """
        x_img: (B,1,H,W) in [0,1]
        returns features: (B, 2*H*W) from [cos(theta), sin(theta)]
        """
        B, C, H, W = x_img.shape
        assert C == 1 and H == self.H and W == self.W

        # initialize phase from pixel intensity
        theta = math.pi * (2.0 * x_img - 1.0)  # map [0,1] -> [-pi, pi]

        omega = self.omega.expand(B, -1, -1, -1)

        for _ in range(self.steps):
            # neighbor theta sum via conv; but we need sin(theta_n - theta)
            # compute sin(theta_n) and cos(theta_n) neighbor sums
            sin_t = torch.sin(theta)
            cos_t = torch.cos(theta)

            sin_n = F.conv2d(sin_t, self.neighbor_kernel, padding=1)
            cos_n = F.conv2d(cos_t, self.neighbor_kernel, padding=1)

            # Using identity:
            # sum sin(theta_n - theta) = cos(theta)*sum sin(theta_n) - sin(theta)*sum cos(theta_n)
            coupling = cos_t * sin_n - sin_t * cos_n

            theta = theta + self.dt * (omega + self.K * coupling)

        feat = torch.cat([torch.cos(theta), torch.sin(theta)], dim=1)  # (B,2,H,W)
        return feat.view(B, -1)  # (B, 2*H*W)

# ---------------- Full Model: Kuramoto -> KANLinear classifier ----------------
class KuramotoKANClassifier(nn.Module):
    def __init__(self, H=28, W=28, num_classes=10, kuramoto_steps=10, num_basis=8):
        super().__init__()
        self.osc = Kuramoto2D(H=H, W=W, steps=kuramoto_steps, dt=0.15, learn_K=True, learn_omega=True)
        in_dim = 2 * H * W
        self.head = KANLinear(
            in_dim,
            num_classes,
            grid_size=5,
            spline_order=3,
            base_activation=nn.SiLU,
            use_logistic_basis=True,
            num_basis=num_basis,
        )

    def forward(self, x_img):
        feat = self.osc(x_img)
        logits = self.head(feat)
        return logits

# ---------------- Training script ----------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    batch_size = 128
    epochs = 3
    lr = 1e-3

    transform = transforms.Compose([
        transforms.ToTensor(),  # (1,28,28) in [0,1]
    ])

    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = KuramotoKANClassifier(kuramoto_steps=10, num_basis=8).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def eval_acc():
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()
        model.train()
        return correct / max(total, 1)

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for step, (x, y) in enumerate(train_loader, 1):
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()

            running += loss.item()
            if step % 200 == 0:
                print(f"epoch {ep} step {step}: loss={running/200:.4f}")
                running = 0.0

        acc = eval_acc()
        print(f"epoch {ep}: test acc={acc:.4f}")

if __name__ == "__main__":
    main()

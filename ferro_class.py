import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class BatchedFerroelectricBasis(nn.Module):
    """
    Clean Ferroelectric hysteresis basis function.

    Simple formula:
    P = Ps * tanh(k * (E + Ec * branch_sign)) + bias

    Parameters:
    - k: Controls slope/sharpness of switching
    - Ec: Coercive field (horizontal shift, where P crosses zero)
    - Ps: Saturation polarization (vertical scale)
    - bias: Vertical offset

    Branch behavior:
    - Upper branch (sign=+1): P = Ps * tanh(k * (E + Ec)) → crosses 0 at E = -Ec
    - Lower branch (sign=-1): P = Ps * tanh(k * (E - Ec)) → crosses 0 at E = +Ec
    """

    def __init__(self, in_dim, out_dim, num_basis, use_noise=False, gate_slope=10.0, alpha=0.8, noise_std=0.05):
        super(BatchedFerroelectricBasis, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_basis = num_basis
        self.use_noise = use_noise
        self.noise_std = noise_std
        self.gate_slope = gate_slope
        self.alpha = alpha

        # Learnable parameters (4 total)
        self.k = nn.Parameter(torch.rand(in_dim, out_dim, num_basis) * 2 + 0.5)      # slope [0.5, 2.5]
        self.Ec = nn.Parameter(torch.rand(in_dim, out_dim, num_basis) * 2 + 0.5)     # coercive field [0.5, 2.5]
        self.Ps = nn.Parameter(torch.rand(in_dim, out_dim, num_basis) * 1.5 + 0.5)   # saturation [0.5, 2.0]
        self.bias = nn.Parameter(torch.randn(in_dim, out_dim, num_basis) * 0.1)
        self.coef = nn.Parameter(torch.randn(in_dim, out_dim, num_basis))

        # State buffers
        self.register_buffer("prev_x", torch.zeros(1, in_dim, out_dim, num_basis))
        self.register_buffer("branch_sign", torch.ones(1, in_dim, out_dim, num_basis))  # +1 upper, -1 lower

    def forward(self, x, return_activations=False):
        if x.dim() > 2:
            x = x.view(x.shape[0], -1)

        batch_size = x.shape[0]
        x_exp = x.unsqueeze(2).unsqueeze(3).expand(-1, -1, self.out_dim, self.num_basis)

        outputs = []
        all_basis = []

        for b in range(batch_size):
            x_sample = x_exp[b:b + 1]
            dx = x_sample - self.prev_x

            # Direction detection
            is_moving_up = torch.sigmoid(self.gate_slope * dx)

            # Coercive field crossing detection
            crossed_pos_Ec = torch.sigmoid(self.gate_slope * (x_sample - self.Ec))
            crossed_neg_Ec = torch.sigmoid(self.gate_slope * (-x_sample - self.Ec))

            # Branch switching logic
            switch_to_upper = is_moving_up * crossed_pos_Ec
            switch_to_lower = (1 - is_moving_up) * crossed_neg_Ec

            target_sign = switch_to_upper * 1.0 + switch_to_lower * (-1.0) + \
                          (1 - switch_to_upper - switch_to_lower) * self.branch_sign
            

            # Smooth update with momentum
            self.branch_sign = self.alpha * self.branch_sign + (1.0 - self.alpha) * target_sign

            # Simple hysteresis formula
            # P = Ps * tanh(k * (E + Ec * sign))
            shifted_x = x_sample + self.Ec * self.branch_sign

            basis = self.Ps * torch.tanh(self.k * shifted_x) + self.bias

            if self.use_noise:
                noise = torch.randn_like(basis) * self.noise_std
                basis = basis + noise.detach()

            self.prev_x.copy_(x_sample.detach())
            all_basis.append(basis)

        basis_all = torch.cat(all_basis, dim=0)
        weighted = basis_all * self.coef
        output = weighted.sum(dim=(1, 3))

        if return_activations:
            return output, basis_all.detach(), self.coef.detach()
        else:
            return output

    def reset_state(self):
        self.prev_x.zero_()
        self.branch_sign.fill_(1.0)


class FerroelectricBasis(nn.Module):
    """
    Clean Ferroelectric hysteresis basis function.

    Simple formula:
    P = Ps * tanh(k * (E + Ec * branch_sign)) + bias

    Parameters:
    - k: Controls slope/sharpness of switching
    - Ec: Coercive field (horizontal shift, where P crosses zero)
    - Ps: Saturation polarization (vertical scale)
    - bias: Vertical offset

    Branch behavior:
    - Upper branch (sign=+1): P = Ps * tanh(k * (E + Ec)) → crosses 0 at E = -Ec
    - Lower branch (sign=-1): P = Ps * tanh(k * (E - Ec)) → crosses 0 at E = +Ec
    """

    def __init__(self, in_dim, out_dim, num_basis, use_noise=False, gate_slope=10.0, alpha=0.8, noise_std=0.05):
        super(FerroelectricBasis, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_basis = num_basis
        self.use_noise = use_noise
        self.noise_std = noise_std
        self.gate_slope = gate_slope
        self.alpha = alpha

        # Learnable parameters (4 total)
        self.k = nn.Parameter(torch.rand(in_dim, out_dim, num_basis) * 2 + 0.5)      # slope [0.5, 2.5]
        self.Ec = nn.Parameter(torch.rand(in_dim, out_dim, num_basis) * 2 + 0.5)     # coercive field [0.5, 2.5]
        self.Ps = nn.Parameter(torch.rand(in_dim, out_dim, num_basis) * 1.5 + 0.5)   # saturation [0.5, 2.0]
        self.bias = nn.Parameter(torch.randn(in_dim, out_dim, num_basis) * 0.1)
        self.coef = nn.Parameter(torch.randn(in_dim, out_dim, num_basis))

        # State buffers
        self.register_buffer("prev_x", torch.zeros(1, in_dim, out_dim, num_basis))
        self.register_buffer("branch_sign", torch.ones(1, in_dim, out_dim, num_basis))  # +1 upper, -1 lower

    def forward(self, x, return_activations=False):
        if x.dim() > 2:
            x = x.view(x.shape[0], -1)

        x_exp = x.unsqueeze(2).unsqueeze(3).expand(-1, -1, self.out_dim, self.num_basis)
        if self.prev_x is None or self.prev_x.shape != x_exp.shape or self.prev_x.device != x_exp.device or self.prev_x.dtype != x_exp.dtype:
            self.prev_x = x_exp.detach().clone()

        # branch_sign buffer shape fix (start on +1 branch by default)
        if self.branch_sign is None or self.branch_sign.shape != x_exp.shape or self.branch_sign.device != x_exp.device or self.branch_sign.dtype != x_exp.dtype:
            self.branch_sign = torch.ones_like(x_exp).detach()

        # --- IMPORTANT: use SNAPSHOTS for differentiable computation ---
        prev_x_snap = self.prev_x.detach().clone()
        branch_snap = self.branch_sign.detach().clone()

        dx = x_exp - prev_x_snap #x_exp - self.prev_x

         # Direction detection
        is_moving_up = torch.sigmoid(self.gate_slope * dx)

        # Coercive field crossing detection
        crossed_pos_Ec = torch.sigmoid(self.gate_slope * (x_exp - self.Ec))
        crossed_neg_Ec = torch.sigmoid(self.gate_slope * (-x_exp - self.Ec))

        # Branch switching logic
        switch_to_upper = is_moving_up * crossed_pos_Ec
        switch_to_lower = (1 - is_moving_up) * crossed_neg_Ec

        target_sign = switch_to_upper * 1.0 + switch_to_lower * (-1.0) + \
                          (1 - switch_to_upper - switch_to_lower) * branch_snap #* self.branch_sign

        # ---- IMPORTANT: momentum update must NOT overwrite the buffer with a grad tensor ----
        # Smooth update with momentum
        # Use a detached snapshot for the previous sign, and keep the update as a local tensor.
        branch_momomentum = self.alpha * branch_snap + (1.0 - self.alpha) * target_sign  # (B, in_dim, out_dim, K)

        # Use branch_mom for the hysteresis computation (differentiable wrt x, Ec, k, Ps, bias, coef)
        shifted_x = x_exp + self.Ec * branch_momomentum
   
        basis = self.Ps * torch.tanh(self.k * shifted_x) + self.bias

        if self.use_noise:
            noise = torch.randn_like(basis) * self.noise_std
            basis = basis + noise.detach()     
        weighted = basis * self.coef
        output = weighted.sum(dim=(1, 3))
        with torch.no_grad():
            self.prev_x.copy_(x_exp.detach()) 
            self.branch_sign.copy_(target_sign.detach())

        if return_activations:
            return output, basis.detach(), self.coef.detach()
        else:
            return output

    def reset_state(self):
        self.prev_x.zero_()
        self.branch_sign.fill_(1.0)



class TwoDimensionFerroelectricBasis(nn.Module):
    def __init__(self, in_dim, num_basis, use_noise=False, gate_slope=10.0, alpha=0.8, noise_std=0.05):
        super(TwoDimensionFerroelectricBasis, self).__init__()
        self.in_dim = in_dim
        self.num_basis = num_basis
        self.use_noise = use_noise
        self.noise_std = noise_std
        self.gate_slope = gate_slope
        self.alpha = alpha

        # Learnable parameters (4 total)
        self.k = nn.Parameter(torch.rand(in_dim, num_basis) * 2 + 0.5)      # slope [0.5, 2.5]
        self.Ec = nn.Parameter(torch.rand(in_dim, num_basis) * 2 + 0.5)     # coercive field [0.5, 2.5]
        self.Ps = nn.Parameter(torch.rand(in_dim, num_basis) * 1.5 + 0.5)   # saturation [0.5, 2.0]
        self.bias = nn.Parameter(torch.randn(in_dim, num_basis) * 0.1)
        self.coef = nn.Parameter(torch.randn(in_dim, num_basis))

        # State buffers
        self.register_buffer("prev_x", torch.zeros(1, in_dim,  num_basis))
        self.register_buffer("branch_sign", torch.ones(1, in_dim, num_basis))  # +1 upper, -1 lower

    def forward(self, x, return_activations=False):
        if x.dim() > 2:
            x = x.view(x.shape[0], -1)

        x_exp = x.unsqueeze(-1).expand(-1, -1, self.num_basis)
        prev_x = self.prev_x
        if (prev_x is None or
            prev_x.shape != x_exp.shape or
            prev_x.device != x_exp.device or
            prev_x.dtype != x_exp.dtype):
            # reset state for new batch shape
            prev_x = x_exp.detach().clone()
            self.prev_x = prev_x 
        dx = x_exp - self.prev_x

        bs = self.branch_sign
        if (bs is None or bs.shape != x_exp.shape or bs.device != x_exp.device or bs.dtype != x_exp.dtype):
            # choose an initialization that makes sense for your model:
            # +1 means “upper branch”, -1 means “lower branch”.
            self.branch_sign = torch.ones_like(x_exp).detach() 
         # Direction detection
        is_moving_up = torch.sigmoid(self.gate_slope * dx)

        # Coercive field crossing detection
        crossed_pos_Ec = torch.sigmoid(self.gate_slope * (x_exp - self.Ec))
        crossed_neg_Ec = torch.sigmoid(self.gate_slope * (-x_exp - self.Ec))

        # Branch switching logic
        switch_to_upper = is_moving_up * crossed_pos_Ec
        switch_to_lower = (1 - is_moving_up) * crossed_neg_Ec

        target_sign = switch_to_upper * 1.0 + switch_to_lower * (-1.0) + \
                          (1 - switch_to_upper - switch_to_lower) * self.branch_sign

        # Smooth update with momentum
        self.branch_sign = self.alpha * self.branch_sign + (1.0 - self.alpha) * target_sign

        # Simple hysteresis formula
        # P = Ps * tanh(k * (E + Ec * sign))
        shifted_x = x_exp + self.Ec * self.branch_sign
        basis = self.Ps * torch.tanh(self.k * shifted_x) + self.bias

        if self.use_noise:
            noise = torch.randn_like(basis) * self.noise_std
            basis = basis + noise.detach()     
        weighted = basis * self.coef
        with torch.no_grad():
            # If prev_x not initialized or batch size changed, re-init to match x_exp
            if (self.prev_x is None) or (self.prev_x.shape != x_exp.shape):
                # keep it as a buffer so it moves with .to(device)
                self.prev_x = x_exp.detach().clone()
            else:
                self.prev_x.copy_(x_exp.detach())



        return weighted

    def reset_state(self):
        self.prev_x.zero_()
        self.branch_sign.fill_(1.0)


class KANFerroelectricNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_basis, use_noise=False):
        super(KANFerroelectricNet, self).__init__()
        self.layer1 = BatchedFerroelectricBasis(in_dim, hidden_dim, num_basis, use_noise=use_noise)
        self.layer2 = BatchedFerroelectricBasis(hidden_dim, out_dim, num_basis, use_noise=use_noise)

    def forward(self, x, return_all=False):
        if return_all:
            out1, basis1, coef1 = self.layer1(x, return_activations=True)
            out2, basis2, coef2 = self.layer2(out1, return_activations=True)
            return out2, basis1, coef1, basis2, coef2, out1
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            return x

    def reset_state(self):
        self.layer1.reset_state()
        self.layer2.reset_state()


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# -------- Training and Visualization --------
def generate_data(n_samples=200):
    x = np.linspace(-5, 5, n_samples)
    y = np.sin(x) + 0.1 * x ** 2
    return x.reshape(-1, 1), y.reshape(-1, 1)


def train(model, x_train, y_train, epochs=2000, lr=1e-2):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    for epoch in range(epochs + 1):
        model.train()
        model.reset_state()
        optimizer.zero_grad()
        output = model(x_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6f}")
    return model


def run_cycle(layer, cycle, E_max=None):
    """Run hysteresis cycle with warmup."""
    layer.reset_state()
    if E_max is None:
        E_max = 5.0

    warmup = torch.cat([
        torch.linspace(0, E_max, 25),
        torch.linspace(E_max, -E_max, 50),
        torch.linspace(-E_max, 0, 25)
    ])
    with torch.no_grad():
        for val in warmup:
            layer(val.view(1, 1), return_activations=True)

    outputs = []
    with torch.no_grad():
        for val in cycle:
            _, basis, _ = layer(val.view(1, 1), return_activations=True)
            outputs.append(basis[0, 0, 0, 0].item())
    return outputs


def make_cycle(E_max):
    """Create a hysteresis cycle with given amplitude."""
    return torch.cat([
        torch.linspace(0, E_max, 50),
        torch.linspace(E_max, -E_max, 100),
        torch.linspace(-E_max, E_max, 100)
    ])


def test_parameters():
    """Test all 4 parameters."""
    import os
    os.makedirs("hysteresis_plots", exist_ok=True)

    param_configs = {
        'k':    [0.5, 1.0, 2.0, 5.0],
        'Ec':   [0.5, 1.0, 2.0, 3.0],
        'Ps':   [0.5, 1.0, 1.5, 2.0],
        'bias': [-0.3, 0.0, 0.3, 0.6],
    }

    defaults = {'k': 2.0, 'Ec': 1.0, 'Ps': 1.0, 'bias': 0.0}

    fig, axes = plt.subplots(4, 4, figsize=(14, 12))

    for row, (param_name, values) in enumerate(param_configs.items()):
        for col, val in enumerate(values):
            # Determine E_max based on Ec
            if param_name == 'Ec':
                E_max = max(5.0, val * 1.5)
            else:
                E_max = 5.0

            layer = TestedFerroelectricBasis(in_dim=1, out_dim=1, num_basis=1, alpha=0.7, use_noise=False)
            with torch.no_grad():
                layer.k.fill_(defaults['k'])
                layer.Ec.fill_(defaults['Ec'])
                layer.Ps.fill_(defaults['Ps'])
                layer.bias.fill_(defaults['bias'])

                if param_name == 'k':
                    layer.k.fill_(val)
                elif param_name == 'Ec':
                    layer.Ec.fill_(val)
                elif param_name == 'Ps':
                    layer.Ps.fill_(val)
                elif param_name == 'bias':
                    layer.bias.fill_(val)

            cycle = make_cycle(E_max)
            outputs = run_cycle(layer, cycle, E_max)

            ax = axes[row, col]
            ax.plot(cycle.numpy(), outputs, 'b-', linewidth=2)
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
            ax.axvline(x=0, color='k', linestyle='-', alpha=0.2)

            if param_name == 'Ec':
                ax.axvline(x=val, color='r', linestyle='--', alpha=0.5)
                ax.axvline(x=-val, color='r', linestyle='--', alpha=0.5)

            ax.set_xlim(-E_max * 1.1, E_max * 1.1)
            ax.set_ylim(-2.5, 2.5)
            ax.grid(True, alpha=0.2)

            if row == 0:
                ax.set_title(f'{val}', fontsize=11)
            if col == 0:
                ax.set_ylabel(f'{param_name}', fontsize=12, fontweight='bold')
            if row < 3:
                ax.set_xticklabels([])
            if col > 0:
                ax.set_yticklabels([])

    plt.suptitle('Clean Model: 4 Parameters (k, Ec, Ps, bias)\nNo Pr - Simplified', fontsize=14)
    plt.tight_layout()
    plt.savefig('hysteresis_plots/clean_parameters.png', dpi=150)
    plt.close()
    print("Saved: hysteresis_plots/clean_parameters.png")


if __name__ == "__main__":
    print("=" * 50)
    print("Clean Ferroelectric Model (No Pr)")
    print("=" * 50)

    test_parameters()

    print("=" * 50)
    print("Done! Check 'hysteresis_plots/clean_parameters.png'")
    print("=" * 50)

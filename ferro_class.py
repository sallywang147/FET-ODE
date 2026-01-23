import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class OriginalBatchedFerroelectricBasis(nn.Module):
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
        super(OriginalBatchedFerroelectricBasis, self).__init__()
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

        all_basis = []

        for b in range(batch_size):
            x_sample = x_exp[b:b + 1]


            # --- IMPORTANT: use SNAPSHOTS for differentiable computation ---
            if (self.prev_x is None) or (self.prev_x.shape != x_sample.shape) or (self.prev_x.device != x_sample.device) or (self.prev_x.dtype != x_sample.dtype):
                        self.prev_x = x_sample.detach().clone()

            if (self.branch_sign is None) or (self.branch_sign.shape != x_sample.shape) or (self.branch_sign.device != x_sample.device) or (self.branch_sign.dtype != x_sample.dtype):
                        self.branch_sign = torch.ones_like(x_sample).detach()

            prev_x_snap = self.prev_x.detach()
            branch_snap = self.branch_sign.detach()

            #  per-sample dx (NOT x_exp)
            dx = x_sample - prev_x_snap
            # Direction detection
            is_moving_up = torch.sigmoid(self.gate_slope * dx)

            # Coercive field crossing detection
            crossed_pos_Ec = torch.sigmoid(self.gate_slope * (x_sample - self.Ec))
            crossed_neg_Ec = torch.sigmoid(self.gate_slope * (-x_sample - self.Ec))

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
            shifted_x = x_sample + self.Ec * branch_momomentum
    


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





class NoisyBatchedFerroelectricBasis(nn.Module):
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

    def __init__(self, in_dim, out_dim, num_basis, use_noise=True, gate_slope=10.0, alpha=0.8, noise_std=0.2):
        super(NoisyBatchedFerroelectricBasis, self).__init__()
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

        all_basis = []

        for b in range(batch_size):
            x_sample = x_exp[b:b + 1]


           # --- IMPORTANT: use SNAPSHOTS for differentiable computation ---
            if (self.prev_x is None) or (self.prev_x.shape != x_sample.shape) or (self.prev_x.device != x_sample.device) or (self.prev_x.dtype != x_sample.dtype):
                        self.prev_x = x_sample.detach().clone()

            if (self.branch_sign is None) or (self.branch_sign.shape != x_sample.shape) or (self.branch_sign.device != x_sample.device) or (self.branch_sign.dtype != x_sample.dtype):
                        self.branch_sign = torch.ones_like(x_sample).detach()

            prev_x_snap = self.prev_x.detach()
            branch_snap = self.branch_sign.detach()

            #  per-sample dx (NOT x_exp)
            dx = x_sample - prev_x_snap

            # Direction detection
            is_moving_up = torch.sigmoid(self.gate_slope * dx)

            # Coercive field crossing detection
            crossed_pos_Ec = torch.sigmoid(self.gate_slope * (x_sample - self.Ec))
            crossed_neg_Ec = torch.sigmoid(self.gate_slope * (-x_sample - self.Ec))

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
            shifted_x = x_sample + self.Ec * branch_momomentum
    


            basis = self.Ps * torch.tanh(self.k * shifted_x) + self.bias
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

        #assumption to check: branch_sign buffer shape fix (start on +1 branch by default)
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
        self.prev_x.copy_(x_exp.detach())
        if self.use_noise:
            noise = torch.randn_like(basis) * self.noise_std
            basis = basis + noise.detach()     
        weighted = basis * self.coef
        output = weighted.sum(dim=(1, 3))
        

        if return_activations:
            return output, basis.detach(), self.coef.detach()
        else:
            return output

    def reset_state(self):
        self.prev_x.zero_()
        self.branch_sign.fill_(1.0)


class NoisyFerroelectricBasis(nn.Module):
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

    def __init__(self, in_dim, out_dim, num_basis, use_noise=True, gate_slope=10.0, alpha=0.8, noise_std=0.2):
        super(NoisyFerroelectricBasis, self).__init__()
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
        
        # --- now take consistent snapshots ---
        prev_x_snap  = self.prev_x.detach().clone()
        branch_snap  = self.branch_sign.detach().clone()
        dx = x_exp - prev_x_snap

        # Direction detection
        is_moving_up = torch.sigmoid(self.gate_slope * dx)

        # Coercive field crossing detection
        crossed_pos_Ec = torch.sigmoid(self.gate_slope * (x_exp - self.Ec))
        crossed_neg_Ec = torch.sigmoid(self.gate_slope * (-x_exp - self.Ec))

        # Branch switching logic
        switch_to_upper = is_moving_up * crossed_pos_Ec
        switch_to_lower = (1 - is_moving_up) * crossed_neg_Ec

        target_sign = switch_to_upper * 1.0 + switch_to_lower * (-1.0) + \
                          (1 - switch_to_upper - switch_to_lower) *  branch_snap #self.branch_sign

        branch_momomentum = self.alpha * branch_snap + (1.0 - self.alpha) * target_sign  # (B, in_dim, out_dim, K)

        # Use branch_mom for the hysteresis computation (differentiable wrt x, Ec, k, Ps, bias, coef)
        shifted_x = x_exp + self.Ec * branch_momomentum

        # Simple hysteresis formula
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




class FerroelectricBasisConv2d(nn.Module):
    """
    Ferroelectric hysteresis basis in Conv2d form.

    Keeps the SAME core formulas as your FerroelectricBasis.forward():

      dx = x_exp - prev_x
      is_moving_up = sigmoid(gate_slope * dx)
      crossed_pos_Ec = sigmoid(gate_slope * (x_exp - Ec))
      crossed_neg_Ec = sigmoid(gate_slope * (-x_exp - Ec))
      switch_to_upper = is_moving_up * crossed_pos_Ec
      switch_to_lower = (1 - is_moving_up) * crossed_neg_Ec
      target_sign = switch_to_upper*(+1) + switch_to_lower*(-1) + (1 - switch_to_upper - switch_to_lower)*branch_prev
      branch_mom = alpha * branch_prev + (1-alpha) * target_sign
      shifted_x = x_exp + Ec * branch_mom
      basis = Ps * tanh(k * shifted_x) + bias
      weighted = basis * coef
      out = sum over (Cin, K, kH, kW)  -> (B, Cout, L) -> fold to (B, Cout, Hout, Wout)

    Notes on state:
    - Images are not time series, so by default we use a per-forward "stateless" mode:
        prev_x = x_exp (so dx=0) and branch_prev = +1
      That keeps the formula intact but avoids cross-batch memory effects.
    - If you want true hysteresis across multiple forwards (e.g., iterative refinement),
      set stateful=True and call reset_state() between sequences/batches.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        num_basis: int = 3,
        stride: int = 1,
        padding: int = 0,
        use_noise: bool = False,
        noise_std: float = 0.2,
        gate_slope: float = 10.0,
        alpha: float = 0.8,
        stateful: bool = False,   # <--- important default for classification
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.num_basis = num_basis
        self.stride = stride
        self.padding = padding

        self.use_noise = use_noise
        self.noise_std = noise_std
        self.gate_slope = gate_slope
        self.alpha = alpha
        self.stateful = stateful

        kH, kW = self.kernel_size

        # Parameters (same roles as your MLP version, but now per (Cout,Cin,K,kH,kW))
        # Shapes:
        #   (Cout, Cin, K, kH, kW)
        self.k    = nn.Parameter(torch.rand(out_channels, in_channels, num_basis, kH, kW) * 2.0 + 0.5)   # [0.5, 2.5]
        self.Ec   = nn.Parameter(torch.rand(out_channels, in_channels, num_basis, kH, kW) * 2.0 + 0.5)   # [0.5, 2.5]
        self.Ps   = nn.Parameter(torch.rand(out_channels, in_channels, num_basis, kH, kW) * 1.5 + 0.5)   # [0.5, 2.0]
        self.bias = nn.Parameter(torch.randn(out_channels, in_channels, num_basis, kH, kW) * 0.1)
        self.coef = nn.Parameter(torch.randn(out_channels, in_channels, num_basis, kH, kW))

        # Output bias (like normal Conv2d)
        self.out_bias = nn.Parameter(torch.zeros(out_channels))

        # Persistent buffers for true hysteresis across multiple forward calls
        # We keep them as registered buffers but we *resize dynamically* on first use.
        self.register_buffer("_prev_x", torch.zeros(1))       # placeholder
        self.register_buffer("_branch_sign", torch.ones(1))   # placeholder
        self._state_ready = False

    def reset_state(self):
        self._state_ready = False
        # buffers will be re-initialized on next forward

    def _ensure_state(self, x_exp_7d: torch.Tensor):
        """
        x_exp_7d shape:
          (B, Cout, Cin, K, kH, kW, L)
        """
        if (not self._state_ready) or (self._prev_x.shape != x_exp_7d.shape) or (self._prev_x.device != x_exp_7d.device) or (self._prev_x.dtype != x_exp_7d.dtype):
            self._prev_x = x_exp_7d.detach().clone()
            self._branch_sign = torch.ones_like(x_exp_7d).detach()  # +1 upper branch
            self._state_ready = True

    def forward(self, x: torch.Tensor, return_activations: bool = False):
        """
        x: (B, Cin, H, W)
        returns: (B, Cout, Hout, Wout)
        """
        B, Cin, H, W = x.shape
        kH, kW = self.kernel_size

        # 1) Unfold into patches: (B, Cin*kH*kW, L)
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        L = x_unfold.shape[-1]

        # 2) Reshape to (B, Cin, kH, kW, L)
        x_unfold = x_unfold.view(B, Cin, kH, kW, L)

        # 3) Expand input for basis:
        #    (B, Cin, 1, kH, kW, L) -> (B, Cin, K, kH, kW, L)
        x_exp = x_unfold.unsqueeze(2).expand(-1, -1, self.num_basis, -1, -1, -1)

        # 4) Expand params to align with patches and output channels
        # Params: (Cout, Cin, K, kH, kW) -> (1, Cout, Cin, K, kH, kW, 1)
        k    = self.k.unsqueeze(0).unsqueeze(-1)
        Ec   = self.Ec.unsqueeze(0).unsqueeze(-1)
        Ps   = self.Ps.unsqueeze(0).unsqueeze(-1)
        bias = self.bias.unsqueeze(0).unsqueeze(-1)
        coef = self.coef.unsqueeze(0).unsqueeze(-1)

        # Expand x to include Cout dim:
        # x_exp: (B, Cin, K, kH, kW, L) -> (B, 1, Cin, K, kH, kW, L) -> broadcast to Cout
        x_exp = x_exp.unsqueeze(1)

        # 5) State handling
        if self.stateful:
            # true hysteresis across forward calls
            self._ensure_state(x_exp)
            prev_x_snap = self._prev_x.detach().clone()
            branch_snap = self._branch_sign.detach().clone()
        else:
            # stateless: keep formulas but avoid cross-batch memory
            prev_x_snap = x_exp.detach().clone()                 # dx = 0
            branch_snap = torch.ones_like(x_exp).detach()        # +1 branch

        # ======== SAME CORE FORMULAS AS YOUR ORIGINAL FORWARD() ========

        dx = x_exp - prev_x_snap
        is_moving_up = torch.sigmoid(self.gate_slope * dx)

        crossed_pos_Ec = torch.sigmoid(self.gate_slope * (x_exp - Ec))
        crossed_neg_Ec = torch.sigmoid(self.gate_slope * (-x_exp - Ec))

        switch_to_upper = is_moving_up * crossed_pos_Ec
        switch_to_lower = (1.0 - is_moving_up) * crossed_neg_Ec

        target_sign = (
            switch_to_upper * 1.0
            + switch_to_lower * (-1.0)
            + (1.0 - switch_to_upper - switch_to_lower) * branch_snap
        )

        branch_momentum = self.alpha * branch_snap + (1.0 - self.alpha) * target_sign

        shifted_x = x_exp + Ec * branch_momentum
        basis = Ps * torch.tanh(k * shifted_x) + bias

        if self.use_noise:
            noise = torch.randn_like(basis) * self.noise_std
            basis = basis + noise.detach()

        weighted = basis * coef
        out = weighted.sum(dim=(2, 3, 4, 5))  # sum over (Cin, K, kH, kW) -> (B, Cout, L)

        # ======== END CORE FORMULAS ========

        if self.stateful:
            # update buffers WITHOUT grads
            self._prev_x.copy_(x_exp.detach())
            # optional: store smoothed sign (still detached)
            self._branch_sign.copy_(branch_momentum.detach())

        # 6) Fold back to image
        Hout = (H + 2 * self.padding - kH) // self.stride + 1
        Wout = (W + 2 * self.padding - kW) // self.stride + 1
        out = out.view(B, self.out_channels, Hout, Wout)

        out = out + self.out_bias.view(1, -1, 1, 1)

        if return_activations:
            # returning basis/coef can be huge; return detached snapshots
            return out, basis.detach(), coef.detach()
        return out



class MemEfficient_FerroelectricBasisConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        num_basis=3,
        stride=1,
        padding=0,
        use_noise=False,
        noise_std=0.05,
        gate_slope=10.0,
        alpha=0.8,
        stateful=False,
        out_chunk=8,          # <-- NEW: process Cout in chunks
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.num_basis = num_basis
        self.stride = stride
        self.padding = padding

        self.use_noise = use_noise
        self.noise_std = noise_std
        self.gate_slope = gate_slope
        self.alpha = alpha
        self.stateful = stateful
        self.out_chunk = out_chunk

        kH, kW = self.kernel_size

        # (Cout, Cin, K, kH, kW)
        self.k    = nn.Parameter(torch.rand(out_channels, in_channels, num_basis, kH, kW) * 2.0 + 0.5)
        self.Ec   = nn.Parameter(torch.rand(out_channels, in_channels, num_basis, kH, kW) * 2.0 + 0.5)
        self.Ps   = nn.Parameter(torch.rand(out_channels, in_channels, num_basis, kH, kW) * 1.5 + 0.5)
        self.bias = nn.Parameter(torch.randn(out_channels, in_channels, num_basis, kH, kW) * 0.1)
        self.coef = nn.Parameter(torch.randn(out_channels, in_channels, num_basis, kH, kW))

        self.out_bias = nn.Parameter(torch.zeros(out_channels))

        # state buffers (optional)
        self.register_buffer("_prev_x", torch.zeros(1))
        self.register_buffer("_branch_sign", torch.ones(1))
        self._state_ready = False

    def reset_state(self):
        self._state_ready = False

    def _ensure_state(self, x_exp_6d):
        # x_exp_6d: (B, Cin, K, kH, kW, L)  (NOTE: no Cout dimension!)
        if (not self._state_ready) or (self._prev_x.shape != x_exp_6d.shape) or (self._prev_x.device != x_exp_6d.device) or (self._prev_x.dtype != x_exp_6d.dtype):
            self._prev_x = x_exp_6d.detach().clone()
            self._branch_sign = torch.ones_like(x_exp_6d).detach()
            self._state_ready = True

    def forward(self, x, return_activations=False):
        """
        x: (B, Cin, H, W)
        returns: (B, Cout, Hout, Wout)
        """
        B, Cin, H, W = x.shape
        kH, kW = self.kernel_size

        # 1) unfold: (B, Cin*kH*kW, L)
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        L = x_unfold.shape[-1]

        # 2) reshape: (B, Cin, kH, kW, L)
        x_unfold = x_unfold.view(B, Cin, kH, kW, L)

        # 3) expand basis: (B, Cin, K, kH, kW, L)
        x_exp = x_unfold.unsqueeze(2).expand(-1, -1, self.num_basis, -1, -1, -1)

        # 4) hysteresis state (NO Cout here!)
        if self.stateful:
            self._ensure_state(x_exp)
            prev_x_snap = self._prev_x.detach()
            branch_snap = self._branch_sign.detach()
        else:
            prev_x_snap = x_exp.detach()                     # dx=0 but formula preserved
            branch_snap = torch.ones_like(x_exp).detach()    # +1

        # ===== SAME CORE FORMULAS (computed without Cout) =====
        dx = x_exp - prev_x_snap
        is_moving_up = torch.sigmoid(self.gate_slope * dx)

        # crossed needs Ec, which *does* depend on Cout -> handle inside chunk loop below
        # So we only compute direction features here; crossed_* uses x_exp and Ec chunk.

        # ======================================================

        # Allocate output in patch-space: (B, Cout, L)
        out_patches = x_exp.new_zeros((B, self.out_channels, L))

        # Process Cout in chunks to avoid (B,Cout,Cin,K,kH,kW,L) tensors
        for oc0 in range(0, self.out_channels, self.out_chunk):
            oc1 = min(self.out_channels, oc0 + self.out_chunk)

            k_c    = self.k[oc0:oc1]     # (Cchunk, Cin, K, kH, kW)
            Ec_c   = self.Ec[oc0:oc1]
            Ps_c   = self.Ps[oc0:oc1]
            bias_c = self.bias[oc0:oc1]
            coef_c = self.coef[oc0:oc1]

            # add patch dim: (1, Cchunk, Cin, K, kH, kW, 1)
            k_c    = k_c.unsqueeze(0).unsqueeze(-1)
            Ec_c   = Ec_c.unsqueeze(0).unsqueeze(-1)
            Ps_c   = Ps_c.unsqueeze(0).unsqueeze(-1)
            bias_c = bias_c.unsqueeze(0).unsqueeze(-1)
            coef_c = coef_c.unsqueeze(0).unsqueeze(-1)

            # x_exp -> (B, 1, Cin, K, kH, kW, L) broadcasting to Cchunk only
            x7 = x_exp.unsqueeze(1)                 # (B,1,Cin,K,kH,kW,L)
            dx7 = dx.unsqueeze(1)
            is_up7 = is_moving_up.unsqueeze(1)
            branch7 = branch_snap.unsqueeze(1)

            crossed_pos_Ec = torch.sigmoid(self.gate_slope * (x7 - Ec_c))
            crossed_neg_Ec = torch.sigmoid(self.gate_slope * (-x7 - Ec_c))

            switch_to_upper = is_up7 * crossed_pos_Ec
            switch_to_lower = (1.0 - is_up7) * crossed_neg_Ec

            target_sign = (
                switch_to_upper * 1.0
                + switch_to_lower * (-1.0)
                + (1.0 - switch_to_upper - switch_to_lower) * branch7
            )

            branch_momentum = self.alpha * branch7 + (1.0 - self.alpha) * target_sign

            shifted_x = x7 + Ec_c * branch_momentum
            basis = Ps_c * torch.tanh(k_c * shifted_x) + bias_c

            if self.use_noise:
                noise = torch.randn_like(basis) * self.noise_std
                basis = basis + noise.detach()

            weighted = basis * coef_c
            # sum over Cin, K, kH, kW -> (B, Cchunk, L)
            out_chunk = weighted.sum(dim=(2, 3, 4, 5))
            out_patches[:, oc0:oc1, :] = out_chunk

        # Update state if stateful
        if self.stateful:
            self._prev_x.copy_(x_exp.detach())
            # store smoothed sign without Cout (average across Cout is undefined) -> keep branch_snap update off
            # If you really want stateful across conv, we can store per-pixel sign per (Cin,K,kH,kW,L). For now:
            self._branch_sign.copy_(branch_snap.detach())

        # Fold back to image
        Hout = (H + 2 * self.padding - kH) // self.stride + 1
        Wout = (W + 2 * self.padding - kW) // self.stride + 1
        out = out_patches.view(B, self.out_channels, Hout, Wout)
        out = out + self.out_bias.view(1, -1, 1, 1)

        if return_activations:
            # returning basis is extremely memory heavy; avoid in diffusion training
            return out, None, None
        return out

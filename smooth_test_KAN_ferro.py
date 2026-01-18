import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt



class FerroelectricBasis(nn.Module):
    """
    Ferroelectric-like hysteresis basis function for KAN networks.
    Each basis function has independent hysteresis behavior.
    """
    def __init__(self, in_dim, out_dim, num_basis, use_noise=False, branch_breaking_point=0.5, gate_slope: float = 5.0, noise_std=0.05):
        super(FerroelectricBasis, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_basis = num_basis
        self.use_noise = use_noise
        self.noise_std = noise_std
        self.branch_breaking_point=branch_breaking_point
        self.gate_slope = gate_slope

        # Learnable parameters for each basis function
        self.k = nn.Parameter(torch.rand(in_dim, out_dim, num_basis) * 2 + 0.5)  # slope [0.5, 2.5]
        self.Ec = nn.Parameter(torch.rand(in_dim, out_dim, num_basis) * 2 + 0.5)  # coercive field [0.5, 2.5]
        self.Ps = nn.Parameter(torch.rand(in_dim, out_dim, num_basis) * 1.5 + 0.5)  # saturation [0.5, 2.0]
        self.bias = nn.Parameter(torch.randn(in_dim, out_dim, num_basis) * 0.1)  # small bias
        self.coef = nn.Parameter(torch.randn(in_dim, out_dim, num_basis))  # weights

        # State buffers for hysteresis (persistent memory)
        # Shape: (1, in_dim, out_dim, num_basis) - includes batch dimension for broadcasting
        self.register_buffer("prev_x", torch.zeros(1, in_dim, out_dim, num_basis))
        self.register_buffer("branch_state", torch.ones(1, in_dim, out_dim, num_basis))  # +1 = up, -1 = dxfown

    def forward(self, x, return_activations=False, ode_mode=False):
        """
        Args:
            x: input tensor of shape (batch, in_dim)
        Returns:
            output: tensor of shape (batch, out_dim)
        """
        # Ensure x is 2D (batch, in_dim)
        if x.dim() > 2:
            x = x.view(x.shape[0], -1)

        batch_size = x.shape[0]

        # Expand x to (batch, in_dim, out_dim, num_basis)
        x_exp = x.unsqueeze(2).unsqueeze(3)  # (batch, in_dim, 1, 1)
        x_exp = x_exp.expand(-1, -1, self.out_dim, self.num_basis)

        # Sort data by x values for sequential processing (important for hysteresis)
        # We'll process each input dimension independently
        outputs = []
        all_basis = []

        for b in range(batch_size):
            x_sample = x_exp[b:b+1]  # (1, in_dim, out_dim, num_basis)

            # Compute change in field
            dx = x_sample - self.prev_x
            g = torch.sigmoid(self.gate_slope * dx) 
            self.branch_state =  (g > self.branch_breaking_point).float()

            # Linear interpolation between up and down branches
            # Define two logistic branches (up and down)
            P_up = self.Ps * (1 / (1 + torch.exp(-self.k * (x_sample - self.Ec)))) * 2 - self.Ps
            P_down = self.Ps * (1 / (1 + torch.exp(-self.k * (x_sample + self.Ec)))) * 2 - self.Ps

            # Select active branch based on sweep direction
            basis = self.branch_state * P_up + (1.0 - self.branch_state) * P_down + self.bias
            # Add noise if enabled
            if self.use_noise:
                noise = torch.randn_like(basis) * self.noise_std
                basis = basis + noise.detach()

            # Update previous state
            self.prev_x.copy_(x_sample.detach())
            all_basis.append(basis)

        # Stack all basis activations
        basis_all = torch.cat(all_basis, dim=0)  # (batch, in_dim, out_dim, num_basis)

        # Apply weights and sum over input dimension and basis dimension
        weighted = basis_all * self.coef  # (batch, in_dim, out_dim, num_basis)
        
        # Sum over in_dim (dim=1) and num_basis (dim=-1) simultaneously to get (batch, out_dim)
        output = weighted.sum(dim=(1, 3))  # Sum over in_dim and num_basis (explicit dim indices)

        if return_activations:
            return output, basis_all.detach(), self.coef.detach()
        else:
            return output

    def reset_state(self):
        """Reset hysteresis state (useful between training runs)"""
        self.prev_x.zero_()
        self.branch_state.fill_(1.0)


class KANFerroelectricNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_basis, use_noise=False):
        super(KANFerroelectricNet, self).__init__()
        self.layer1 = FerroelectricBasis(in_dim, hidden_dim, num_basis, use_noise=use_noise)
        self.layer2 = FerroelectricBasis(hidden_dim, out_dim, num_basis, use_noise=use_noise)

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
        """Reset all hysteresis states"""
        self.layer1.reset_state()
        self.layer2.reset_state()


# -------- Generate Symbolic Regression Dataset --------
def generate_data(n_samples=200):
    x = np.linspace(-5, 5, n_samples)
    y = np.sin(x) + 0.1 * x**2
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    return x, y


# -------- Training --------
def train(model, x_train, y_train, epochs=4000, lr=1e-2):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Sort training data by x for better hysteresis behavior
    x_train_sorted, sort_idx = np.sort(x_train, axis=0), np.argsort(x_train, axis=0).squeeze()
    y_train_sorted = y_train[sort_idx]

    x_train_tensor = torch.tensor(x_train_sorted, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_sorted, dtype=torch.float32)

    for epoch in range(epochs):
        model.train()
        model.reset_state()  # Reset hysteresis state at start of each epoch

        optimizer.zero_grad()
        output = model(x_train_tensor)

        # Debug: print shapes on first epoch
        if epoch == 0:
            print(f"Output shape: {output.shape}, Target shape: {y_train_tensor.shape}")

        # L1 regularization for pruning basis functions
        reg = torch.sum(torch.abs(model.layer1.coef)) + torch.sum(torch.abs(model.layer2.coef))
        loss = criterion(output, y_train_tensor) + 1e-3 * reg

        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 200 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch} | Loss: {loss.item():.6f} | LR: {current_lr:.6f}")

    return model


# -------- Hysteresis Loop Visualization (P-E Curve Style) --------
def visualize_hysteresis_loop(model, layer_idx=1, field_range=(-5, 5), n_points=100,
                                basis_idx=0, in_dim_idx=0, out_dim_idx=0):
    """
    Visualize actual hysteresis loop (P-E curve) for ferroelectric basis functions.

    Args:
        model: Trained KANFerroelectricNet model
        layer_idx: Which layer to visualize (1 or 2)
        field_range: Tuple of (min, max) field values
        n_points: Number of points in each sweep direction
        basis_idx: Which basis function to visualize
        in_dim_idx: Which input dimension to use
        out_dim_idx: Which output dimension to use
    """
    model.eval()

    # Select the layer
    if layer_idx == 1:
        layer = model.layer1
    else:
        layer = model.layer2

    # Create field sweep: up then down
    E_up = torch.linspace(field_range[0], field_range[1], n_points)
    E_down = torch.linspace(field_range[1], field_range[0], n_points)

    P_up = []
    P_down = []

    # Reset state before starting
    layer.reset_state()

    # Up sweep
    for e in E_up:
        # Create input with proper shape for the layer
        if layer_idx == 1:
            x_in = torch.tensor([[e.item()]], dtype=torch.float32)
        else:
            # For layer 2, we need hidden_dim inputs (from layer1 output)
            # W'll just use repeated values for visualization
            x_in = torch.tensor([[e.item()] * layer.in_dim], dtype=torch.float32)

        # Get basis activations
        with torch.no_grad():
            _, basis, _ = layer(x_in, return_activations=True)

        # Extract the specific basis function output
        p_val = basis[0, in_dim_idx, out_dim_idx, basis_idx].item()
        P_up.append(p_val)

    # Down sweep (continue from previous state - this is key for hysteresis!)
    for e in E_down:
        if layer_idx == 1:
            x_in = torch.tensor([[e.item()]], dtype=torch.float32)
        else:
            x_in = torch.tensor([[e.item()] * layer.in_dim], dtype=torch.float32)

        with torch.no_grad():
            _, basis, _ = layer(x_in, return_activations=True)

        p_val = basis[0, in_dim_idx, out_dim_idx, basis_idx].item()
        P_down.append(p_val)

    # Plot the hysteresis loop
    plt.figure(figsize=(8, 6))
    plt.plot(E_up.numpy(), P_up, 'b-', linewidth=2.5, label='Up sweep', marker='o',
             markersize=3, markevery=10)
    plt.plot(E_down.numpy(), P_down, 'r-', linewidth=2.5, label='Down sweep', marker='s',
             markersize=3, markevery=10)

    plt.xlabel('Electric Field (E)', fontsize=14)
    plt.ylabel('Polarization (P)', fontsize=14)
    plt.title(f'Ferroelectric Hysteresis Loop - Layer {layer_idx}\n' +
              f'Basis {basis_idx}, In-dim {in_dim_idx}, Out-dim {out_dim_idx}', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Calculate coercive field (where P crosses zero)
    E_up_np = E_up.numpy()
    P_up_np = np.array(P_up)
    P_down_np = np.array(P_down)

    # Find zero crossings
    try:
        up_cross_idx = np.where(np.diff(np.sign(P_up_np)))[0]
        down_cross_idx = np.where(np.diff(np.sign(P_down_np)))[0]

        if len(up_cross_idx) > 0 and len(down_cross_idx) > 0:
            Ec_up = E_up_np[up_cross_idx[0]]
            Ec_down = E_down.numpy()[down_cross_idx[0]]
            print(f"Coercive field (up): {Ec_up:.3f}")
            print(f"Coercive field (down): {Ec_down:.3f}")
    except:
        pass


# -------- Multiple Basis Functions Hysteresis Visualization --------
def visualize_multiple_hysteresis_loops(model, layer_idx=1, field_range=(-5, 5),
                                        n_points=100, max_basis=4):
    """
    Visualize hysteresis loops for multiple basis functions in a grid.

    Args:
        model: Trained KANFerroelectricNet model
        layer_idx: Which layer to visualize (1 or 2)
        field_range: Tuple of (min, max) field values
        n_points: Number of points in each sweep direction
        max_basis: Maximum number of basis functions to show
    """
    model.eval()

    # Select the layer
    if layer_idx == 1:
        layer = model.layer1
    else:
        layer = model.layer2

    num_basis_to_show = min(layer.num_basis, max_basis)

    # Create subplot grid
    n_cols = min(2, num_basis_to_show)
    n_rows = (num_basis_to_show + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
    if num_basis_to_show == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Create field sweep
    E_up = torch.linspace(field_range[0], field_range[1], n_points)
    E_down = torch.linspace(field_range[1], field_range[0], n_points)

    for basis_idx in range(num_basis_to_show):
        ax = axes[basis_idx]

        P_up = []
        P_down = []

        # Reset state
        layer.reset_state()

        # Up sweep
        for e in E_up:
            if layer_idx == 1:
                x_in = torch.tensor([[e.item()]], dtype=torch.float32)
            else:
                x_in = torch.tensor([[e.item()] * layer.in_dim], dtype=torch.float32)

            with torch.no_grad():
                _, basis, coef = layer(x_in, return_activations=True)

            # Use first in_dim and out_dim for simplicity
            p_val = basis[0, 0, 0, basis_idx].item()
            P_up.append(p_val)

        # Down sweep
        for e in E_down:
            if layer_idx == 1:
                x_in = torch.tensor([[e.item()]], dtype=torch.float32)
            else:
                x_in = torch.tensor([[e.item()] * layer.in_dim], dtype=torch.float32)

            with torch.no_grad():
                _, basis, coef = layer(x_in, return_activations=True)

            p_val = basis[0, 0, 0, basis_idx].item()
            P_down.append(p_val)

        # Plot
        ax.plot(E_up.numpy(), P_up, 'b-', linewidth=2, label='Up sweep')
        ax.plot(E_down.numpy(), P_down, 'r-', linewidth=2, label='Down sweep')

        # Get weight for this basis
        weight = coef[0, 0, basis_idx].item()

        ax.set_xlabel('Electric Field (E)', fontsize=11)
        ax.set_ylabel('Polarization (P)', fontsize=11)
        ax.set_title(f'Basis {basis_idx} (weight={weight:.3f})', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide extra subplots
    for idx in range(num_basis_to_show, len(axes)):
        axes[idx].set_visible(False)

    # plt.suptitle(f'Ferroelectric Hysteresis Loops - Layer {layer_idx}', fontsize=16, y=1.00)
    plt.suptitle(f'Basis function Loops - Layer {layer_idx}', fontsize=16, y=1.00)

    plt.tight_layout()
    plt.show()


# -------- Original Visualization Function (Weighted Basis Responses) --------
def visualize_basis_responses(basis_vals, coefs, x, layer_name="", max_display=3):
    """Visualize weighted basis function responses (NOT hysteresis loops)"""
    in_dim, out_dim, num_basis = coefs.shape
    x_vals = x.squeeze().numpy()

    # Only visualize first few basis functions to avoid clutter
    num_to_show = min(num_basis, max_display)

    fig, axs = plt.subplots(out_dim, in_dim, figsize=(6 * in_dim, 4 * out_dim), squeeze=False)

    for j in range(out_dim):
        for i in range(in_dim):
            ax = axs[j][i]
            for k in range(num_to_show):
                act = basis_vals[:, i, j, k].numpy()
                weight = coefs[i, j, k].item()
                if abs(weight) > 1e-3:
                    ax.plot(x_vals, act * weight, label=f"basis {k} (w={weight:.2f})")

            ax.set_title(f"{layer_name} - Device response {j} from input {i}")
            ax.set_xlabel("Input")
            ax.set_ylabel("Weighted activation")
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# -------- Run Everything --------
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    # Generate data
    x_train, y_train = generate_data(n_samples=200)

    # Create model with ferroelectric basis
    model = KANFerroelectricNet(in_dim=1, hidden_dim=5, out_dim=1, num_basis=8, use_noise=True)

    # Train model
    print("Training Ferroelectric KAN on symbolic regression task...")
    trained_model = train(model, x_train, y_train, epochs=4000, lr=1e-2)

    # Test on finer grid
    x_test = np.linspace(-5, 5, 300).reshape(-1, 1)
    x_test_sorted = np.sort(x_test, axis=0)
    x_test_tensor = torch.tensor(x_test_sorted, dtype=torch.float32)

    # Get predictions
    trained_model.eval()
    trained_model.reset_state()
    with torch.no_grad():
        y_pred, b1, c1, b2, c2, out1 = trained_model(x_test_tensor, return_all=True)
        y_pred = y_pred.numpy()

    # Ground truth for test set
    y_test_gt = np.sin(x_test_sorted.squeeze()) + 0.1 * x_test_sorted.squeeze()**2

    # Plot prediction vs ground truth
    plt.figure(figsize=(10, 6))
    plt.plot(x_train, y_train, 'b.', markersize=8, label="Training data", alpha=0.6)
    plt.plot(x_test_sorted, y_test_gt, 'g--', linewidth=2, label="Ground truth", alpha=0.7)
    plt.plot(x_test_sorted, y_pred, 'r-', linewidth=2.5, label="Prediction")
    plt.legend(loc="upper left", fontsize=14)
    plt.title("Symbolic Regression", fontsize=18)
    plt.xlabel("x", fontsize=14)
    plt.ylabel("y", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Visualize ACTUAL ferroelectric hysteresis loops (P-E curves)
    print("\n" + "="*60)
    print("HYSTERESIS LOOP VISUALIZATIONS")
    print("="*60)

    print("\nVisualizing Layer 1 hysteresis loops (multiple basis functions)...")
    visualize_multiple_hysteresis_loops(trained_model, layer_idx=1, field_range=(-5, 5),
                                        n_points=100, max_basis=4)

    print("\nVisualizing Layer 2 hysteresis loops (multiple basis functions)...")
    visualize_multiple_hysteresis_loops(trained_model, layer_idx=2, field_range=(-5, 5),
                                        n_points=100, max_basis=4)

    # Optional: Visualize single basis function in detail
    print("\nDetailed hysteresis loop for Layer 1, Basis 0:")
    visualize_hysteresis_loop(trained_model, layer_idx=1, basis_idx=0,
                              field_range=(-5, 5), n_points=150)

    # Save model
    torch.save(model.state_dict(), "KAN_ferro_SR_trained.pth")
    print("\n" + "="*60)
    print("Model saved as KAN_ferro_SR_trained.pth")
    print("="*60)

import sys
from pathlib import Path

# 1. Get the directory of uca.py (/Users/issy/Projects/linear-causal-abstraction/src/models)
current_script_dir = Path(__file__).resolve().parent

# 2. Go up THREE levels to reach /Users/issy/Projects/
projects_root = current_script_dir.parent.parent.parent

# 3. Point to the folder that CONTAINS the 'unsupervised_causal_reduction' package
target_project_dir = projects_root / 'targeted-causal-reduction-unsupervisedAbstraction'

sys.path.append(str(target_project_dir))
import lightning as L
import torch
import numpy as np
from typing import Optional, Tuple
from unsupervised_causal_reduction.causal_model.high_level import HighLevelCausalModel #type: ignore
from unsupervised_causal_reduction.causal_model.low_level import LowLevelCausalModel #type: ignore
from unsupervised_causal_reduction.data_generators.linear import LinearCausalModel #type: ignore
from unsupervised_causal_reduction.abstraction.causal_abstraction import CausalAbstraction #type: ignore
from unsupervised_causal_reduction.lightning.targeted_causal_reduction import UnsupervisedCausalAbstractionModel #type: ignore


class SCMLow(LowLevelCausalModel):
    def __init__(self, A_low: torch.Tensor):
        n_vars = A_low.shape[0]
        super().__init__(n_vars=n_vars)
        self.low_level_scm = A_low
    
    def attributes(self):
        pass

    def sample(self, intervention: np.ndarray, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:

        pass

    def _sample_unmasked_intervention(self) -> np.ndarray:
        sign = np.random.choice([-1,1],size=self.n_vars)
        shift = sign * np.random.uniform(0.5,1.5,size=self.n_vars)
        return shift

    def _sample_intervention(self) -> Tuple[np.ndarray, np.ndarray]:
        scale = np.random.uniform(0.5,1.5,size=self.n_vars)
        sign = np.random.choice([-1,1],size=self.n_vars)
        shift = sign * np.random.uniform(0.5,1.5,size=self.n_vars)
        return scale, shift
    
    def zero_intervention(self) -> np.ndarray:
        return np.zeros(shape=(1,self.n_vars))
    

def uca_model(A_low,k):
    p = A_low.shape[0]
    Model_H = HighLevelCausalModel(n_vars=k,E_macro=None)
    Model_L = LinearCausalModel(n_vars=p,low_level_scm=A_low)
    CA = CausalAbstraction(low_level=Model_L,high_level=Model_H)

    model = UnsupervisedCausalAbstractionModel(abstraction=CA, consistency_reg=1.0, overlap_reg=10, projection_reg=1.0)

    return model

def uca_train(uca_model, num_epochs, verbose=False):
    """Train the abstraction for `num_epochs` gradient steps.

    FIXED 2026-09-24. This previously called `_training_step()` in a bare loop and never
    applied the gradient. `_training_step` only COMPUTES the loss and returns it -- under
    Lightning it is the Trainer that calls `backward()` and `optimizer.step()` -- and there
    is no Trainer here, so the returned abstraction was exactly its random initialisation
    regardless of `num_epochs`. Measured before the fix: T bit-identical after 300 steps,
    `tau_map_weights.weight.grad is None`, loss flat at ~3.5e5. After it: loss 3.5e5 -> 5.3e4
    over 30k steps and T moves.

    The optimiser comes from the model's own `configure_optimizers()`, so the upstream
    learning rate and parameter groups are respected. That call builds a CosineAnnealingLR
    referencing `self.trainer.max_epochs` when `lr_scheduler == "cosine"`; there is no
    trainer here, so it is guarded and falls back to plain Adam at the model's own lr.

    NOTE ON EPOCHS: one "epoch" here is a single gradient step on one sampled intervention,
    NOT a pass over a dataloader as upstream's `max_epochs` means. The two numbers are not
    comparable. Upstream defines no convergence criterion (no EarlyStopping anywhere in the
    package), so this runs a fixed budget; the loss is stochastic across steps, so any
    plateau rule added later needs a windowed mean rather than a per-step comparison.
    """
    try:
        cfg = uca_model.configure_optimizers()
        if isinstance(cfg, dict):
            optimizer = cfg["optimizer"]
        elif isinstance(cfg, (list, tuple)):
            optimizer = cfg[0]
        else:
            optimizer = cfg
    except Exception as e:
        print(f"uca_train: configure_optimizers failed ({type(e).__name__}: {e}); "
              f"falling back to Adam(lr={getattr(uca_model, 'lr', 1e-3)}).")
        optimizer = torch.optim.Adam(uca_model.parameters(),
                                     lr=getattr(uca_model, "lr", 1e-3))

    for epoch in range(num_epochs):
        try:
            loss, results = uca_model._training_step()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        except torch.linalg.LinAlgError:
            print(f"uca_train: singular matrix at epoch {epoch}/{num_epochs}, stopping early with current abstraction.")
            break
        if verbose and epoch % max(1, num_epochs // 10) == 0:
            print(f"uca_train: epoch {epoch}/{num_epochs} loss {float(loss):.4g}")

    return uca_model.abstraction

import torch
from torch.distributions.transforms import PositiveDefiniteTransform

def test_covariance_stability():
    n_vars = 100  # Match your high-level variable count
    
    # 1. Simulate the absolute worst-case scenario: highly negative parameters
    # This causes exp(diagonal) to approach 0 under standard conditions.
    pre_cov = torch.nn.Parameter(torch.ones(n_vars, n_vars) * -50.0)
    
    # 2. Apply the PyTorch Transform
    raw_cov = PositiveDefiniteTransform()(pre_cov)
    
    # 3. Apply the stabilization floor
    eps = 1e-4
    stable_cov = raw_cov + (eps * torch.eye(n_vars, dtype=raw_cov.dtype))
    
    print("--- Test 1: Condition Numbers ---")
    print(f"Raw Covariance Condition Number:    {torch.linalg.cond(raw_cov).item():.2e}")
    print(f"Stabilized Covariance Condition No: {torch.linalg.cond(stable_cov).item():.2e}")
    print("\n*Note: An infinite (inf) or extremely high (>1e+7) condition number will crash the solver.*")
    
    print("\n--- Test 2: Mock Solver Pass ---")
    dummy_mean_diff = torch.ones(n_vars, 1)
    
    # Test Raw (Should fail or produce NaNs if extremely negative)
    try:
        torch.linalg.solve(raw_cov, dummy_mean_diff)
        print("Raw solve: Passed (but potentially unstable)")
    except torch._C._LinAlgError:
        print("Raw solve: FAILED (Expected behavior for singular matrix)")
        
    # Test Stabilized (Must pass)
    try:
        torch.linalg.solve(stable_cov, dummy_mean_diff)
        print("Stabilized solve: PASSED successfully!")
    except torch._C._LinAlgError as e:
        print(f"Stabilized solve: FAILED unexpectedly: {e}")

if __name__ == "__main__":
    test_covariance_stability()

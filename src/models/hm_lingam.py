import sys
import os
sys.path.insert(0, os.path.abspath(os.getcwd()))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from src.utils.helper import log, regularised_surjective_init, get_S, get_A_high

def rbf_kernel(x, sigma=1.0):
    """
    x: (n, 1) tensor
    returns: (n, n) kernel matrix
    """
    sq_dists = (x - x.T) ** 2
    return torch.exp(-sq_dists / (2 * sigma ** 2))


def hsic(X, sigma_x=1.0, sigma_y=1.0):
    assert X.ndim == 2 and X.shape[1] == 2, "X should be a 2D tensor with shape (n,2)."
    n = X.shape[0]
    
    K = rbf_kernel(X[:, 0:1], sigma_x)
    L = rbf_kernel(X[:, 1:2], sigma_y)
    
    # Efficient double-centering without matrix H
    K_c = K - K.mean(dim=0, keepdim=True) - K.mean(dim=1, keepdim=True) + K.mean()
    L_c = L - L.mean(dim=0, keepdim=True) - L.mean(dim=1, keepdim=True) + L.mean()
    
    # Trace of product of symmetric matrices is sum of element-wise products
    hsic_score = torch.sum(K_c * L_c) / ((n - 1) ** 2)
    return hsic_score

def plr_score(x, y, eps=1e-8):
    """
    Pairwise Likelihood Ratio (tanh-based, "sparse" approximation) for
    determining causal direction between two standardized non-Gaussian
    scalar variables, per Hyvarinen & Smith (2013), Eq. 5:

        R_sparse = rho * E[ x*tanh(y) - tanh(x)*y ]

    Interpretation:
        R > 0  =>  evidence for x -> y
        R < 0  =>  evidence for y -> x
        R ~ 0  =>  direction not determined (e.g. x, y Gaussian, or
                   x and y are degenerate/collinear)

    Assumes x, y are (n,) or (n,1) tensors. They will be standardized
    internally (zero mean, unit variance) since the derivation requires
    standardized variables -- do NOT pre-standardize and then also rely
    on this to skip it, but it's safe to call this on raw data directly.

    Note: rho is the *signed* correlation coefficient, not its square --
    getting this sign right matters since flipping it flips R's sign.
    """
    x = x.reshape(-1)
    y = y.reshape(-1)

    x = (x - x.mean()) / (x.std(unbiased=True) + eps)
    y = (y - y.mean()) / (y.std(unbiased=True) + eps)

    rho = (x * y).mean()  # correlation of standardized vars = mean of product

    nonlinear_corr = (x * torch.tanh(y) - torch.tanh(x) * y).mean()

    R = rho * nonlinear_corr
    return R


def plr_matrix(X, eps=1e-8):
    """
    Computes the full pairwise PLR matrix M for a (n, k) tensor of k
    standardized variables, per Eq. 18:

        M = C ⊙ (E[x g(x)^T] - E[g(x) x^T]),  g = tanh

    M[i,j] > 0 is evidence that x_i -> x_j (roughly: row i is "more
    exogenous" relative to column j). For a true root/source variable,
    Hyvarinen & Smith show its entire row should be (approximately)
    non-negative -- this is the basis for DirectLiNGAM-style root-finding
    using this statistic (see their Section 3.2.1).

    Returns:
        M: (k, k) tensor, M[i,j] = pairwise PLR score for (x_i, x_j)
    """
    n, k = X.shape
    Xs = (X - X.mean(dim=0, keepdim=True)) / (X.std(dim=0, unbiased=True, keepdim=True) + eps)

    C = (Xs.T @ Xs) / (n - 1)          # (k,k) correlation matrix
    G = torch.tanh(Xs)                  # (n,k)

    # E[x_i * tanh(x_j)] for all i,j -> (k,k), then transpose-difference
    cross = (Xs.T @ G) / n              # cross[i,j] = E[x_i * tanh(x_j)]
    M = C * (cross - cross.T)

    return M

class GradDiscovery(nn.Module):
    def __init__(self, p, k, logits_init=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.k = k
        logits_init = torch.from_numpy(regularised_surjective_init(p,k)).float() #torch.cat([torch.zeros(p,k-1),torch.ones(p,1)],dim=1) if logits_init is None else logits_init # 
        self.logits = nn.Parameter(logits_init)

    def get_partition(self, temp=1., gumbel=False):
        T_soft = F.gumbel_softmax(self.logits / temp, dim=1, hard=True) if gumbel else F.softmax(self.logits/temp, dim=1)
        indices = torch.argmax(self.logits, dim=1)
        T_hard = F.one_hot(indices, self.k).float()
        T_hard = (T_hard - T_soft).detach() + T_soft 
        
        return T_hard, T_soft
    
    def get_residual(self, X):
        n, k = X.shape
        assert k >= 2, f"The number of variables must be at least 2. Got {k}."

        x0 = X[:, 0:1]
        X_rest = X[:, 1:]
        variables = [x0]
        residuals = [X_rest]          # raw downstream block, x0 excluded

        for _ in range(k - 2):
            x0_centered = x0 - x0.mean(dim=0, keepdim=True)
            X_rest_centered = X_rest - X_rest.mean(dim=0, keepdim=True)

            cov_0j = (x0_centered * X_rest_centered).sum(dim=0) / (n - 1)
            var_0 = x0_centered.var(dim=0, unbiased=True)
            coeff = cov_0j / (var_0 + 1e-8)

            R = X_rest - coeff * x0
            X = R.clone()

            if X.shape[1] == 0:
                break                  # nothing left to deflate

            x0 = X[:, 0:1]
            X_rest = X[:, 1:]
            variables.append(x0)
            residuals.append(X_rest)

        return variables, residuals
        
    
    def forward(self, X):
        T_hard, T_soft = self.get_partition(gumbel=True)
        T = T_soft
        X_abstract = X @ T_hard

        entropy_loss = torch.mean(-torch.sum(T * torch.log(T + 1e-8), dim=1))
        top_k,_ = torch.topk(T,k=2,dim=0) 
        col_deficit = F.relu(0.5 - top_k.mean(dim=0))  
        surjective_loss = (col_deficit * (-torch.log(T + 1e-8).mean(dim=0))).sum()


        variables, residuals = self.get_residual(X_abstract)

        indp_loss = 0.
        for x0, R in zip(variables, residuals):
            scores = torch.stack([plr_score(x0, R[:, j:j+1]) for j in range(R.shape[1])])
            indp_loss += torch.sum(F.leaky_relu(-scores))
        

        return 100* indp_loss, entropy_loss, surjective_loss
    
    def train(self, X, num_epochs=1000, batch_size=None, keep_history=True, lambda_entropy=1e-2, lambda_surj=1.):
        n,p = X.shape
        assert p == self.p, f"Expected {self.p} variables. Got {p} instead. Instantiate another model with the required number of variables."

        history = {'indp_loss':[], 'entropy_loss':[], 'surjective_loss':[], 'T':[]}
        optimiser = torch.optim.Adam(self.parameters())

        batch_size = n if batch_size is None else batch_size
        loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)
        for epoch in range(num_epochs):
            batch_history = {'indp_loss':[], 'entropy_loss':[], 'surjective_loss':[]}
            for (X_batch,) in loader:
                optimiser.zero_grad()
                indp_loss, entropy_loss, surjective_loss = self.forward(X_batch)
                total_loss = indp_loss + lambda_entropy * entropy_loss + lambda_surj * surjective_loss
                total_loss.backward()
                optimiser.step()

                batch_history['indp_loss'].append(indp_loss.item())
                batch_history['entropy_loss'].append(entropy_loss.item())
                batch_history['surjective_loss'].append(surjective_loss.item())
            
            #import pdb;pdb.set_trace()
            
            avg_loss = np.mean(batch_history['indp_loss'])
            avg_entropy = np.mean(batch_history['entropy_loss'])
            avg_surjective = np.mean(batch_history['surjective_loss'])
            if epoch % 100 == 0 or epoch == num_epochs - 1:
                    log(msg=f"Epoch {epoch + 1}/{num_epochs}: Loss = {avg_loss}, Entropy = {avg_entropy}, Surjective = {avg_surjective}")
            if keep_history:
                history['indp_loss'].append(avg_loss)
                history['entropy_loss'].append(np.mean(avg_entropy))
                history['surjective_loss'].append(avg_surjective)
                history['T'].append(self.get_partition()[1].cpu().detach().numpy())

        return history


if __name__ == "__main__":
    from src.utils.plots import plot_matrix, visualize_causal_graph, plot_probability_path, plot_assignment_raster, animate_matrices
    from src.utils.generate_data import sample_low_level_graph, sample_gennorm_lingam_population_standardized
    from src.utils.losses import intervention_loss, observation_loss, abstraction_residual, graph_loss, topo_loss, intervention_loss, reconstruction_loss
    from src.utils.helper import enforce_dag, is_dag, reconstruct_adj
    import matplotlib.pyplot as plt
    from pathlib import Path
    from src.models import hierarchical_clustering
    import time

    # Graph Sampling
    seed = None #44
    n_low = 10 #number of low-level variables
    n_high = 2  #number of high-level variables
    k = n_high
    density = 0.4
    internal_density = 0.
    blockiness = 0.1
    heteregoneity = 1.
    A_low, T_gt, S_gt, A_high_gt = sample_low_level_graph(n_high,n_low,density,seed=seed,shuffle=True,blockiness=blockiness, heterogeneity=heteregoneity, internal_density=internal_density)
    visualize_causal_graph(A_low, A_high_gt, T_gt, S_gt, title='Ground Truth Causal Abstraction',threshold=1e-3)

    # Data Sampling
    n_train = 500
    n_test = 100
    gennorm = 1.8

    X_train = sample_gennorm_lingam_population_standardized(A_low=A_low, n_samples=n_train, gennorm_param=gennorm)
    X_test = sample_gennorm_lingam_population_standardized(A_low=A_low, n_samples=n_test, gennorm_param=gennorm)

    #import pdb;pdb.set_trace()
    model = GradDiscovery(p=n_low, k=k)
    T_init = model.get_partition()[0].cpu().detach().numpy()
    history = model.train(torch.from_numpy(X_train).float(),batch_size=100,num_epochs=5000)

    animate_matrices(history['T'],title='LiNGAM_abstraction',cmap='Reds')

    T_pred, T_soft = model.get_partition(gumbel=False)
    T_pred = T_pred.cpu().detach().numpy()
    T_soft = T_soft.cpu().detach().numpy()

    S_pred = get_S(T=T_pred, A_low=A_low)
    A_high_pred = get_A_high(T=T_pred, A_low=A_low)

    visualize_causal_graph(A_low, A_high_pred, T_pred, S_pred, title='Learned Causal Abstraction',threshold=1e-3)
    plot_matrix(
        matrix_list=[T_init,T_pred, T_soft],
        title_list=['Initial', 'Learned', 'Soft'],
        fig_title='Abstraction Matrices'
    )

    fig,ax = plt.subplots(figsize=(10,8))
    ax.plot(range(0,len(history['indp_loss']),10), history['indp_loss'][::10])
    ax.plot(range(0,len(history['entropy_loss']),10), history['entropy_loss'][::10])
    ax.plot(range(0,len(history['surjective_loss']),10), history['surjective_loss'][::10])
   
    ax.legend(['Loss', 'Entropy','Surjective'])
    #ax.set_yscale('log')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    plt.show()
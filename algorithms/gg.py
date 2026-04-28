import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import time
# Instead of a formal package import, ensure your local folder is hit
import sys
import os

# Add the current directory to the path to prioritize local files
sys.path.insert(0, os.path.abspath(os.getcwd()))

from utils.plots import plot_matrix, visualize_causal_graph
from utils.generate_data import random_dag, generate_random_T, sample_low_level_graph
from utils.helper import reconstruct_adj
from causabs.sampling import sample_linear_concretization

class CausalLearner(nn.Module):
    def __init__(self,n_low,n_high,threshold=1e-5,init_assignment=None,warm_start=None,T_type='binary',learn_S=True, learn_A_high=False,marginalise=False):
        super().__init__()
        self.n_low = n_low
        self.n_high = n_high
        self.threshold = threshold
        self.T_type = T_type
        self.set_W(init_assignment,warm_start=warm_start,marginalise=marginalise)
        self.set_T(T_type,marginalise)

        assert not (learn_S and learn_A_high), "Cannot learn both S and A_high simultaneously."
        self.learn_S = learn_S
        self.learn_A_high = learn_A_high
        if self.learn_S:
            self.set_S()
        elif self.learn_A_high:
            self.set_A_high()
    
    def set_W(self,init_assignment,warm_start,marginalise):
        if warm_start is not None:
            assert warm_start.shape == (self.n_low, self.n_high), f"Expected shape {self.n_low,self.n_high}. Got {warm_start.shape}."
            if type(warm_start) is np.ndarray:
                self.W = nn.Parameter(torch.from_numpy(warm_start).to(torch.float32))
                W_mask = torch.ones(self.n_low,self.n_high)

        elif marginalise:
            self.W = nn.Parameter(torch.randn(self.n_low,self.n_high),requires_grad=True) #the last column represents the high-level bin node for marginalised low-level variables
            W_mask = torch.ones(self.n_low,self.n_high+1)
        else:
            self.W = nn.Parameter(torch.randn(self.n_low,self.n_high),requires_grad=True)
            W_mask = torch.ones(self.n_low,self.n_high)

        if init_assignment:
            assignment_tensor = torch.from_numpy(init_assignment).long()
            valid_mask = (assignment_tensor < self.n_high) & (assignment_tensor != -1)
            low_level_indices = torch.where(valid_mask)[0]
            high_level_indices = assignment_tensor[valid_mask]
            W_mask[low_level_indices] = 0.0
            W_mask[low_level_indices, high_level_indices] = 1.0


        self.register_buffer('W_mask', W_mask)
    
    def set_A_high(self):
        self.A_high = nn.Parameter(torch.randn(self.n_high,self.n_high),requires_grad=True)
        
        A_high_mask = torch.triu(torch.ones(self.n_high,self.n_high), diagonal=1)
        self.register_buffer('A_high_mask', A_high_mask)

    def set_S(self):
        self.S = nn.Parameter(torch.randn(self.n_low,self.n_high),requires_grad=True)
        A_high_mask = torch.triu(torch.ones(self.n_high,self.n_high), diagonal=1)
        self.register_buffer('A_high_mask', A_high_mask)

    def set_T(self,T_type,marginalise):
        if T_type == 'binary':
            self.T = torch.ones(self.n_low,self.n_high+1) if marginalise else torch.ones(self.n_low,self.n_high)
        elif T_type == 'simplex' or T_type == 'sphere':
            self.T = nn.Parameter(torch.randn(self.n_low,self.n_high+1),requires_grad=True) if marginalise else nn.Parameter(torch.randn(self.n_low,self.n_high),requires_grad=True)
        else:
            raise ValueError(f"Unsupported T_type {T_type}. Choose 'binary', 'simplex' or 'sphere'.")
    
    def get_T(self,temp=None,hard=True):
        if self.training and temp is not None:
            partition_mask = F.gumbel_softmax(self.W * self.W_mask,tau=temp,hard=hard,dim=1)
        else:
            partition_mask = F.one_hot(torch.argmax(self.W * self.W_mask,dim=1),num_classes=self.W.shape[1]).float()

        if self.T_type == 'binary':
            T = partition_mask * self.T
        elif self.T_type == 'simplex':
            logits = self.T.masked_fill(partition_mask == 0, float('-inf'))
            T = F.softmax(logits, dim=0)
            T = torch.nan_to_num(T, nan=0.0)
        elif self.T_type == 'sphere':
            T = F.normalize(partition_mask * self.T, p=2, dim=0, eps=1e-12)

        #threshold_mask = (torch.abs(T)>self.threshold).float()
        #T = (threshold_mask * T - T).detach() + T
        return T, partition_mask

    def get_S(self,partition_mask_full,T,E,E_high,impose_sufficiency=True):
        s_mask = partition_mask_full[:, :self.n_high]
        if not impose_sufficiency: 
            is_marginalised = partition_mask_full[:, -1].unsqueeze(1)
            s_mask = s_mask + is_marginalised * torch.ones_like(T) #this allows marginalised low-level variables to influence all high-level variables.
        
        S = torch.linalg.solve(E_high.T, (E @ T).T).T * s_mask
        #threshold_mask = (torch.abs(S) > self.threshold).float()
        #S = (threshold_mask * S - S).detach() + S
        return S
    
    def get_TS(self,temp=None,hard=True):
        assert self.learn_S, "get_TS should only be called when learn_S is True"
        T, partition_mask = self.get_T(temp,hard)
        S = partition_mask * self.S
        return T, S, partition_mask

    def get_entropy(self):
        probs = torch.softmax(self.W*self.W_mask, dim=1)
        entropy = -1 * torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        return entropy.mean()
    
    def get_dag_penalty(self, E_high=None, A_high=None):
        assert E_high is not None or A_high is not None, "Either E_high or A_high must be provided for DAG penalty calculation."
        if A_high is None:
            A_high = torch.eye(self.n_high, device=E_high.device) - torch.linalg.inv(E_high)

        M = A_high * A_high
        return torch.trace(torch.matrix_exp(M)) - self.n_high
    
    def get_reconstruction_penalty(self,partition_mask_full,T,E,E_high):
        S = self.get_S(partition_mask_full,T,E,E_high)
        E_recon = S @ E_high @ torch.linalg.pinv(T)
        recon_error = torch.mean((E_recon - E)**2)
        #baseline = (self.n_low - self.n_high)/(self.n_low ** 2)
        return recon_error 

    def get_adjacency_reconstruction_penalty(self,T,A_low,E_high=None,A_high=None):
        if A_high is None:
            A_high = torch.eye(self.n_high, device=E_high.device) - torch.linalg.inv(E_high)
        if self.T_type == 'binary':
            column_norms = torch.sum(T,dim=0)
        elif self.T_type == 'simplex':
            column_norms = torch.sum(T, dim=0)
        elif self.T_type == 'sphere':
            column_norms = torch.sum(T**2, dim=0)
        
        T_norm = T / (column_norms + 1e-12)
        A_recon = T @ A_high @ T_norm.T

        adj_recon_loss = torch.mean((A_recon - A_low)**2)
        return adj_recon_loss

    def get_marginalization_penalty(self,partition_mask_full,marg_frac=0.):
        if self.W_mask.shape[1] == self.n_high + 1: #check if we have a marginalisation column
            marg = torch.mean(partition_mask_full[:,-1])
            return F.relu(marg - marg_frac)
        return torch.tensor(0.0, device=partition_mask_full.device)

    def get_surjectivity_penalty(self,partition_mask_full,target_norm=1.):
        col_sums = partition_mask_full.sum(dim=0)
        target_sums = torch.ones_like(col_sums) * target_norm
        if self.W_mask.shape[1] == self.n_high + 1: #check if we have a marginalisation column
            target_sums[-1] = 0

        return torch.sum(F.relu(target_sums - col_sums))
    
    @torch.no_grad()
    def boost_lucky_state(self, lucky_mask, scale=1.0):
        """
        Boost the logits of the lucky state.

        lucky_mask: (n_low, n_high + 1) - The hard binary mask that gave zero loss.
        scale: How much to 'boost' the lucky logits.
        """
        target_logits = (lucky_mask * scale) - ((1 - lucky_mask) * scale)
        self.W.add_(target_logits)

    
    def forward(self,E,temp,hard=False,loss_type='gram'):
        T_full, partition_mask_full = self.get_T(temp=temp, hard=hard)
        
        # EXCLUDE the bin column
        T = T_full[:, :self.n_high]  # (n_low, n_high)
        partition_mask = partition_mask_full[:, :self.n_high] # (n_low, n_high)
        device = E.device
        I = torch.eye(self.n_low,device=device)
        F_pi = torch.diag_embed(partition_mask.T) #(n_high, n_low, n_low)
    
        term = I + F_pi @ (E - I)
        ET = E @ T
        M = torch.tensor(0.)
        #M = (E - I) @ inv_term @ F_pi
        I_high = torch.eye(self.n_high, device=device)
        A_high = None

        if self.learn_S:
            T, S, partition_mask = self.get_TS(temp=temp,hard=hard) #T and S have the same partition mask
            T_batch = T.unsqueeze(0).expand(self.n_high, -1, -1) 
            S_batch = S.unsqueeze(0).expand(self.n_high, -1, -1)
            lhs = torch.bmm(F_pi, S_batch)
            rhs = term @ torch.bmm(F_pi, T_batch)
            ET = E @ T
            A_high = torch.linalg.lstsq(ET, ET - S).solution * self.A_high_mask
            A_high_thresh = ((torch.abs(A_high) > self.threshold).float() - A_high).detach() + A_high
            A_high = A_high * A_high_thresh

            intv_loss = torch.mean((lhs-rhs)**2)
            obs_loss = torch.mean((S - ET@(I_high - A_high))**2)#self.get_dag_penalty(A_high=A_high)**2
            E_high = None

        elif self.learn_A_high:
            A_high = self.A_high * self.A_high_mask
            E_batch = E.unsqueeze(0).expand(self.n_high,-1,-1)
            lhs = torch.einsum('ikl,lj->ijk', M @ E, T @ (I_high - A_high))
            rhs = torch.einsum('kl,ilm,mj->ijk',(E-I), F_pi, T)

            intv_loss = torch.mean((lhs - rhs)**2) * self.n_high
            #dag_loss = torch.trace(torch.matrix_exp(A_high * A_high)) - self.n_high

            S = E @ T @ (I_high - A_high) 
            S_masked = S * partition_mask
            obs_loss = torch.mean((S - S_masked.detach())**2) #+ dag_loss

            E_high = torch.linalg.inv(I_high - A_high).detach()
            E_high.fill_diagonal_(1.0)

        else:
            #lhs = torch.einsum('ikl,lj->ijk', M @ E, T)
            #rhs = torch.einsum('ki,j -> ijk',(E-I) @ T, torch.ones(self.n_high, device=device))
            ET_batch = ET.unsqueeze(0).expand(self.n_high,-1,-1)
            T_batch = T.unsqueeze(0).expand(self.n_high, -1, -1) 
            lhs = torch.bmm(F_pi,ET_batch)
            term_T = torch.bmm(term, T_batch)  # (n_high, n_low, n_high)
            # extract column i from each batch i: term[i] @ T[:,i]
            rhs_col = term_T[torch.arange(self.n_high), :, torch.arange(self.n_high)]  # (n_high, n_low)
            rhs = rhs_col.unsqueeze(-1).expand(-1, -1, self.n_high)  # (n_high, n_low, n_high)
    
            

            dot_prod = torch.sum(lhs * rhs, dim=1)
            a_norm = torch.sum(lhs * lhs, dim=1)
            b_norm = torch.sum(rhs * rhs, dim=1)
            E_high_ij = dot_prod/(b_norm + 1e-6)

            E_high_mask = torch.ones_like(E_high_ij) - torch.eye(self.n_high, device=device)
            #E_high_mask = torch.triu(torch.ones_like(E_high_ij), diagonal=1)
            E_high = E_high_ij * E_high_mask + I_high
            
            cluster_sizes = partition_mask.sum(dim=0)  # (n_high,)
            nonempty = (cluster_sizes > 0).float()
            nonempty_ij = nonempty.unsqueeze(1) * nonempty.unsqueeze(0)  # (n_high, n_high)

            valid_mask = E_high_mask * nonempty_ij
            topo_bias = torch.sum(torch.tril(E_high_ij,diagonal=-1)**2)

            if loss_type == 'gram':
                gram_loss = (a_norm * b_norm) - dot_prod ** 2
                intv_loss = torch.relu(gram_loss * valid_mask).mean()

            elif loss_type == 'cosine':
                loss_ij = torch.ones(self.n_high, self.n_high) - dot_prod**2 /(a_norm * b_norm + 1e-8)
                intv_loss = torch.relu(loss_ij).mean()

            elif loss_type == 'mse':
                intv_loss = torch.mean((lhs - E_high.unsqueeze(-1) * rhs)**2)
        

              
            S = self.get_S(partition_mask_full,T,E,E_high)
            obs_loss = torch.mean((S @ E_high - E @T)**2)

        return obs_loss, intv_loss, T, E_high, A_high, partition_mask_full

@dataclass
class TrainingConfig:
    alignment_epochs: int = 10000
    lr: float = 0.01
    acceptable_loss: float = 1e-6
    marginalization_pen: float = 1.
    abstraction_pen: float = 10.
    surjectivity_pen: float = 1.
    dag_pen: float = 1.
    verbose: bool = False
    threshold: float = 1e-8
    temp: float = 5.0
    temp_min: float = 0.1
    anneal_rate: float = 0.0005
    hard: bool = True
    fine_tune_epochs: int = 2000

def calculate_anneal_rate(t_start, t_min, max_epochs):
    """
    Calculates the multiplicative decay rate needed to reach t_min
    from t_start in exactly max_epochs.
    """
    if max_epochs <= 0:
        return 0.0
    
    # Using the derived formula: r = 1 - (t_min / t_start)**(1 / max_epochs)
    rate = 1 - (t_min / t_start)**(1 / max_epochs)
    
    return rate

def log(msg,verbose=True):
        if verbose: print(msg)

def get_reconstruction_loss(A_low, T, A_high=None, E_high=None, T_type='binary'):
    if A_high is None:
        n_high = E_high.shape[0]
        A_high = torch.eye(n_high, device=E_high.device) - torch.linalg.inv(E_high)
    if T_type == 'binary':
        column_norms = torch.sum(T,dim=0)
    elif T_type == 'simplex':
        column_norms = torch.sum(T, dim=0)
    elif T_type == 'sphere':
        column_norms = torch.sqrt(torch.sum(T**2, dim=0))
        
    T_norm = T / (column_norms + 1e-6)
    A_recon = T @ A_high @ T_norm.T

    adj_recon_loss = torch.mean((A_recon - A_low)**2)
    return adj_recon_loss


def gumbel_grad(E,n_high,T_type='binary',init_assignment=None, warm_start=None, train_config=TrainingConfig(),keep_history=True):
    n_low = E.shape[0]
    A_low = torch.eye(n_low, device=E.device) - torch.linalg.inv(E)
    model = CausalLearner(n_low, n_high, T_type=T_type, init_assignment=init_assignment, warm_start=warm_start)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)
    temp = train_config.temp
    history = {'obs_loss': [], 'interv_loss': [], 'entropy': [], 'surjectivity' : [], 'reconstruction': [], 'marginalization': [], 'dag': []}
    best_val_loss = float('inf')
    best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
    
    log("Starting Gumbel-Softmax training...",train_config.verbose)
    epoch = 0
    identity_count = 0
    alignment_epochs = int(50 * n_low)
    anneal_rate = calculate_anneal_rate(train_config.temp, train_config.temp_min, alignment_epochs)
    while epoch < alignment_epochs:
        model.eval() # Temporarily switch to argmax mode
        with torch.no_grad():
            val_obs_loss, val_intv_loss, T , E_high , A_high, partition_mask_full = model(E, temp, hard=True)
            val_reconstruction_loss = get_reconstruction_loss(A_low=A_low, T=T, A_high=A_high, E_high=E_high, T_type=T_type)
            val_surjectivity_loss = model.get_surjectivity_penalty(partition_mask_full)
            val_marginalization_loss = 0 #train_config.marginalization_pen * model.get_marginalization_penalty(partition_mask_full)
            val_dag_loss = 0 #train_config.dag_pen * model.get_dag_penalty(E_high=E_high, A_high=A_high)
            current_val_loss = max(val_obs_loss.item(), val_intv_loss.item())
        model.train()
        
        is_almost_valid = (val_obs_loss < train_config.acceptable_loss) and (val_intv_loss < train_config.acceptable_loss)
        constraints_met = (val_surjectivity_loss == 0.0 and 
                           val_marginalization_loss == 0.0 and 
                           val_dag_loss == 0.0)

        if constraints_met:
            if is_almost_valid:
                current_val_loss = val_reconstruction_loss.item()
                model.boost_lucky_state(partition_mask_full.detach(), scale=1.0) 
            else:
                current_val_loss = max(val_obs_loss.item(), val_intv_loss.item()) + val_reconstruction_loss.item()

            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        if keep_history:
            history['obs_loss'].append(obs_loss.item())
            history['interv_loss'].append(intv_loss.item())
            history['entropy'].append(model.get_entropy().item())
            history['marginalization'].append(marginalization_loss.item())
            history['surjectivity'].append(surjectivity_loss.item())
            history['reconstruction'].append(reconstruction_loss.item())
            history['dag'].append(dag_loss.item())

        optimizer.zero_grad()
        obs_loss, intv_loss, T, E_high, A_high, partition_mask_full = model(E, temp, hard=train_config.hard)

        reconstruction_loss = get_reconstruction_loss(A_low=A_low, T=T, A_high=A_high, E_high=E_high, T_type=T_type)
        surjectivity_loss = model.get_surjectivity_penalty(partition_mask_full)
        marginalization_loss = torch.tensor(0.) #train_config.marginalization_pen * model.get_marginalization_penalty(partition_mask_full)
        dag_loss = torch.tensor(0.) #train_config.dag_pen * model.get_dag_penalty(E_high=E_high,A_high=A_high)

        abstraction_loss = intv_loss + obs_loss #torch.log(intv_loss + 1e-6) + torch.log(obs_loss + 1e-6)
        train_loss = (reconstruction_loss 
                      + train_config.abstraction_pen * abstraction_loss
                      + train_config.surjectivity_pen * surjectivity_loss 
                      + train_config.marginalization_pen * marginalization_loss 
                      + train_config.dag_pen * dag_loss)
        train_loss.backward()
        optimizer.step()

        temp = max(train_config.temp_min, temp * (1 - anneal_rate))
        if epoch % 100 == 0:
            log(f"Epoch {epoch}, Observed Loss: {obs_loss.item()}, Interventional Loss: {intv_loss.item()}, ReconstructionLoss: {reconstruction_loss.item()}, Temp: {temp}",train_config.verbose)
        
        epoch += 1
    
    best_model = CausalLearner(n_low,n_high)
    best_model.load_state_dict(best_state)
    best_model.eval()
    
    log(f"Training completed. Best Validation Loss: {best_val_loss} \t Identity Count: {identity_count}/{train_config.alignment_epochs}",train_config.verbose)
    return best_model,history

def gg_output(model,E_low):
    E_low_tensor = torch.from_numpy(E_low).to(torch.float32)
    A_low_tensor = torch.eye(E_low_tensor.shape[0], device=E_low_tensor.device) - torch.linalg.inv(E_low_tensor)
    final_obs_loss, final_intv_loss, final_T, final_E_high, final_A_high, final_partition_mask_full = model(E_low_tensor,None)
    if model.learn_S:
        final_T, final_S, final_partition_mask_full = model.get_TS(temp=None, hard=True)
    else:
        final_S = model.get_S(final_partition_mask_full,final_T,E_low_tensor,final_E_high)

    T_pred = final_T.detach().cpu().numpy()
    partition_mask = final_partition_mask_full[:,:-1].detach().cpu().numpy()

    if final_E_high is not None:
        E_high = final_E_high.detach().cpu().numpy()

    if final_A_high is not None:
        A_high = final_A_high.detach().cpu().numpy()
        E_high = np.linalg.inv(np.eye(A_high.shape[0]) - A_high)
    else:
        A_high = np.eye(model.n_high) - np.linalg.inv(E_high)

    S_pred = final_S.detach().cpu().numpy()
    #S_pred = np.round(S_pred, decimals=3)

    #reconstruction_loss = model.get_reconstruction_penalty(final_partition_mask_full,final_T,E_low_tensor,final_E_high)
    reconstruction_loss = model.get_adjacency_reconstruction_penalty(final_T,A_low_tensor,A_high=final_A_high,E_high=final_E_high)
    surjectivity_loss = model.get_surjectivity_penalty(final_partition_mask_full)
    marginalization_loss = model.get_marginalization_penalty(final_partition_mask_full)
    dag_loss = model.get_dag_penalty(E_high=final_E_high, A_high=final_A_high)

    final_losses = {
        "reconstruction": reconstruction_loss.item(),
        "surjectivity": surjectivity_loss.item(),
        "marginalization": marginalization_loss.item(),
        "dag": dag_loss.item(),
        "obs_loss": final_obs_loss.item(),
        "intv_loss": final_intv_loss.item()
    }

    return partition_mask, T_pred, S_pred, E_high, A_high, final_losses

if __name__ == "__main__":
    seed = None #23 sometimes gives an example of zero loss but graphical violations
    n_low = 8
    n_high = 3
    marg_prob = 0.
    T_lower = (-2,1)
    T_upper = (-1,2)
    edge_density = 0.4

    A_low,T_gt,S_gt,A_high_gt = sample_low_level_graph(n_high=n_high,n_low=n_low,density=edge_density,T_lower=T_lower,T_upper=T_upper,marg_prob=marg_prob,seed=seed,shuffle=True,internal=False)
    E_low = np.linalg.inv(np.eye(n_low) - A_low)
    E_low_tensor = torch.from_numpy(E_low).to(torch.float32)
    
    plot_matrix(
        matrix_list=[A_high_gt,T_gt,S_gt],
        title_list=['High-Level Adjacency','Endogenous Abstraction T','Exogenous Abstraction S'],
        fig_title='Ground Truth Matrices'
    )
    visualize_causal_graph(A_low, A_high_gt, T_gt, S_gt, title='Ground Truth Causal Abstraction')

    ORACLE = True
    alignment_epochs = int(100 * n_low**2)
  
    train_config = TrainingConfig(alignment_epochs=alignment_epochs, lr=0.01, verbose=True, threshold=1e-8, temp=5.0, temp_min=0.1, anneal_rate=0.0005, hard=True)
    start_time = time.time()
    model,history = gumbel_grad(E_low_tensor,n_high=n_high,train_config=train_config)
    print(f"Completed in {time.time()-start_time:.4f}s")

    partition_mask, T_pred, S_pred, E_high, A_high, final_losses = gg_output(model,E_low)
    active_nodes = (partition_mask.sum(axis=0) > 0).astype(float)
    observed_loss = np.linalg.norm(S_pred@E_high - E_low @ T_pred, ord=1)

    print(f"Final Observed Loss: {final_losses['obs_loss']}")
    print(f"Final Interventional Loss: {final_losses['intv_loss']}")
    print(f"Final Reconstruction Loss: {final_losses['reconstruction']}")
    print(f"Final DAG Loss: {final_losses['dag']}")
    print(f"Final Surjectivity Loss: {final_losses['surjectivity']}")
    print(f"Final Marginalization Loss: {final_losses['marginalization']}")
    
    
    print(f"A_low:\n{A_low}")
    print(f"T_pred:\n{T_pred}")
    print(f"S_pred:\n{S_pred}")
    print(f"A_high:\n{A_high}")
    
    visualize_causal_graph(A_low, A_high, T_pred, S_pred, title='Learned Causal Abstraction')

    
    plot_matrix(
        matrix_list=[A_high,T_pred,S_pred],
        title_list=['High-Level Adjacency','Abstraction Matrix T','Exogenous Abstraction S'],
        fig_title='Learned Matrices'
    )


    #Plot a graph of loss vs epochs
    fig, ax = plt.subplots(1,2, figsize=(10,8))
    ax[0].plot(range(0,len(history['obs_loss']),10), history['obs_loss'][::10])
    ax[0].plot(range(0,len(history['interv_loss']),10), history['interv_loss'][::10])
    ax[0].plot(range(0,len(history['reconstruction']),10), history['reconstruction'][::10])
    ax[0].plot(range(0,len(history['dag']),10), history['dag'][::10])
    ax[0].legend(['Observed Loss','Interventional Loss','Reconstruction Loss','DAG Loss'])
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[1].plot(range(0,len(history['entropy']),10), history['entropy'][::10])
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Entropy")
    plt.show()
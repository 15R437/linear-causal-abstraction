import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from dataclasses import dataclass
# Instead of a formal package import, ensure your local folder is hit
import sys
import os

# Add the current directory to the path to prioritize local files
sys.path.insert(0, os.path.abspath(os.getcwd()))

from utils.plots import plot_matrix, visualize_causal_graph
from utils.generate_data import random_dag, generate_random_T, sample_low_level_graph
from causabs.sampling import sample_linear_concretization

class CausalLearner(nn.Module):
    def __init__(self,n_low,n_high,threshold=1e-8,W=None,T=None,E_high=None,E_high_mask=None):
        super().__init__()
        self.n_low = n_low
        self.n_high = n_high
        self.threshold = threshold
        self.set_W(W)
        self.set_T(T)
        self.set_E_high(E_high)

        if E_high_mask is None:
            E_high_mask = torch.triu(torch.ones(n_high, n_high), diagonal=1)
            E_high_mask = torch.concatenate([E_high_mask,torch.zeros(n_high,1)],dim=1)
            E_high_mask = torch.concatenate([E_high_mask,torch.zeros(1,n_high+1)],dim=0)
        else:
            assert(E_high_mask.shape == (self.n_high,self.n_high)),f"Expected shape {(self.n_high,self.n_high)}, got {E_high_mask.shape}." 
            E_high_mask = torch.concatenate([E_high_mask,torch.zeros(n_high,1)],dim=1)
            E_high_mask = torch.concatenate([E_high_mask,torch.zeros(1,n_high+1)],dim=0)
            E_high_mask.fill_diagonal_(0)

        self.register_buffer('mask', E_high_mask)
        self.register_buffer('identity', torch.eye(n_high+1))
    

    def set_W(self,W=None,requires_grad=True):
        if W is None:
            self.W = nn.Parameter(torch.randn(self.n_low,self.n_high+1),requires_grad=requires_grad) #the last column represents the high-level bin node for marginalised low-level variables
        else:
            assert(W.shape == (self.n_low,self.n_high)),f"Expected shape {(self.n_low,self.n_high)}, got {W.shape}."
            marg_col = torch.all(W == 0,dim=1).view(-1,1)
            W = torch.concatenate([W,marg_col],dim=1)
            self.W = nn.Parameter(W,requires_grad=requires_grad)
    
    def set_T(self,T=None,requires_grad=True):
        if T is None:
            self.T = nn.Parameter(torch.randn(self.n_low,self.n_high+1),requires_grad=requires_grad)
        else:
            assert(T.shape == (self.n_low,self.n_high)),f"Expected shape {(self.n_low,self.n_high)}, got {T.shape}."
            marg_col = torch.all(T == 0,dim=1).view(-1,1)
            T = torch.concatenate([T,marg_col],dim=1)
            self.T = nn.Parameter(T,requires_grad=requires_grad)

    def set_E_high(self,E_high=None,requires_grad=True):
        if E_high is None:
            self.E_high = nn.Parameter(torch.randn(self.n_high+1,self.n_high+1),requires_grad=requires_grad)
        else:
            assert(E_high.shape == (self.n_high,self.n_high)),f"Expected shape {(self.n_high,self.n_high)}, got {E_high.shape}."
            E_high = torch.concatenate([E_high,torch.zeros(self.n_high,1)],dim=1)
            E_high = torch.concatenate([E_high,torch.zeros(1,self.n_high+1)],dim=0)
            self.E_high = nn.Parameter(E_high,requires_grad=requires_grad)
    
    def get_assignment_entropy(self):
        probs = torch.softmax(self.W, dim=1)
        entropy = -1 * torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        return entropy.mean()
    
    def get_T(self,temp=None,hard=True):
        if self.training and temp is not None:
            partition_mask = F.gumbel_softmax(self.W,tau=temp,hard=hard,dim=1)
        else:
            partition_mask = F.one_hot(torch.argmax(self.W,dim=1),num_classes=self.n_high+1).float()

        T = partition_mask * self.T
        threshold_mask = (torch.abs(T)>self.threshold).float()
        return (threshold_mask * T - T).detach() + T, partition_mask
    
    def get_E_high(self,partition_mask=None):
        if partition_mask is None:
            partition_mask = F.one_hot(torch.argmax(self.W,dim=1),num_classes=self.n_high+1).float()
        
        active_nodes = (partition_mask.sum(dim=0) > 0)
        interaction_mask = active_nodes.unsqueeze(0) * active_nodes.unsqueeze(1)
        
        E_high_masked = (self.E_high * self.mask) * interaction_mask + self.identity
        threshold_mask = (torch.abs(E_high_masked)>self.threshold).float()
        #import pdb; pdb.set_trace()
        return (threshold_mask * E_high_masked - E_high_masked).detach() + E_high_masked
    
    def get_S(self,E,T,partition_mask):
        E_high = self.get_E_high(partition_mask)
        is_marginalised = partition_mask[:,-1] == 1
        s_mask = partition_mask.clone()
        s_mask[is_marginalised, :-1] = 1

        S = (E @ T @ torch.linalg.inv(E_high)) * s_mask
        threshold_mask = (torch.abs(S) > self.threshold).float()
        return (threshold_mask * S - S).detach() + S
    
    def get_surjectivity_penalty(self,target_norm=1.):
        hard_mask = F.one_hot(torch.argmax(self.W,dim=1), num_classes=self.n_high+1).float()
        soft_probs = F.softmax(self.W, dim=1)
        effective_mask = (hard_mask - soft_probs).detach() + soft_probs
        
        col_sums = effective_mask.sum(dim=0)
        target_sums = torch.ones_like(col_sums) * target_norm
        target_sums[-1] = 0

        return torch.sum(F.relu(target_sums - col_sums))
    
    def get_scale_penalty(self):
        hard_mask = F.one_hot(torch.argmax(self.W,dim=1), num_classes=self.n_high+1).float()
        soft_probs = F.softmax(self.W, dim=1)
        effective_mask = (hard_mask - soft_probs).detach() + soft_probs
        T_masked = self.T * effective_mask

        col_norms = torch.norm(T_masked,p=2,dim=0)/self.n_low
        active_nodes = (effective_mask.sum(dim=0)>0).float()
        target_norms = torch.ones_like(col_norms) * 0.1
        target_norms[-1] = 0

        return torch.sum(active_nodes * F.relu(target_norms - col_norms))
    
    def forward(self,E,temp,hard=False):
        T,partition_mask = self.get_T(temp=temp,hard=hard)
        E_tilde = self.get_E_high(partition_mask)
        S = self.get_S(E,T,partition_mask)

        obs_loss = self.n_high * torch.norm((S@E_tilde -E@T), p=2)**2 # how far is S from S * partition_mask?
        interv_loss = 0
        I_low = torch.eye(self.n_low,device=E.device)
        e_j_batch = torch.eye(self.n_high + 1, device=E.device)
        F_ss_batch = torch.diag_embed(torch.einsum('ij,kj->ik',e_j_batch,partition_mask))
        F_jj_batch = torch.diag_embed(e_j_batch)
        lhs = torch.linalg.solve(I_low + F_ss_batch @ (E - I_low),F_ss_batch @ E @ T)
        rhs = T @ F_jj_batch @ E_tilde
        residual = (E - I_low) @ (lhs - rhs) 
        interv_loss = torch.norm(residual,p=2)**2

        return obs_loss , interv_loss

    
@dataclass
class TrainingConfig:
    alignment_epochs: int = 10000
    lr: float = 0.01
    acceptable_loss: float = 0.
    entropy_pen: float = 0.
    surjectivity_pen: float = 1.
    scale_pen: float = 1.
    verbose: bool = True
    threshold: float = 1e-8
    temp: float = 5.0
    temp_min: float = 0.1
    anneal_rate: float = 0.0005
    hard: bool = True
    fine_tune_epochs: int = 2000

def log(msg,verbose=True):
        if verbose: print(msg)

def gumbel_grad(E,n_high,train_config=TrainingConfig(),W=None,E_high_mask=None,**kwds):
    n_low = E.shape[0]
    model = CausalLearner(n_low, n_high, threshold=train_config.threshold, W=W, E_high_mask=E_high_mask)
    model.set_W(W, requires_grad=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)
    temp = train_config.temp
    history = {'obs_loss': [], 'interv_loss': [], 'entropy': [], 'surjectivity' : [], 'scale': []}
    best_val_loss = float('inf')
    best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
    
    log("Starting Gumbel-Softmax training...",train_config.verbose)
    epoch = 0
    while epoch < train_config.alignment_epochs:
        optimizer.zero_grad()
        obs_loss, interv_loss = model(E, temp, hard=train_config.hard)
        loss = obs_loss + interv_loss

        entropy_loss = train_config.entropy_pen  *  model.get_assignment_entropy()
        surjectivity_loss = train_config.surjectivity_pen * model.get_surjectivity_penalty()
        scale_loss = train_config.scale_pen * temp * model.get_scale_penalty()

        total_loss = loss + entropy_loss + surjectivity_loss + scale_loss
        model.eval() # Temporarily switch to argmax mode
        with torch.no_grad():
            current_obs_loss, current_interv_loss = model(E, temp, hard=True)
            current_val_loss = current_obs_loss + current_interv_loss 
        model.train() # Switch back

        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}

        history['obs_loss'].append(obs_loss.item())
        history['interv_loss'].append(interv_loss.item())
        history['entropy'].append(entropy_loss.item())
        history['surjectivity'].append(surjectivity_loss.item())
        history['scale'].append(model.get_scale_penalty().item())
        if best_val_loss <= train_config.acceptable_loss:
            log(f"Early stopping at epoch {epoch} with loss {best_val_loss}",train_config.verbose)
            max_epochs = epoch
            break
        
        total_loss.backward()
        optimizer.step()
        temp = max(train_config.temp_min, temp* (1 - train_config.anneal_rate))
        if epoch % 100 == 0:
            log(f"Epoch {epoch}, ObsLoss: {obs_loss.item()}, IntervLoss: {interv_loss.item()}, SurjectivityLoss: {surjectivity_loss.item()}, Temp: {temp}",train_config.verbose)
        
        epoch += 1

    alignment_epochs = epoch

    best_model = CausalLearner(n_low, n_high, threshold=train_config.threshold)
    best_model.load_state_dict(best_state)
    best_model.eval()
    best_model.W.requires_grad = False
    #freeze W and fine-tune T and E_high
    log("Starting fine-tuning of T and E_high...",verbose=train_config.verbose)
    optimizer = torch.optim.Adam([best_model.T, best_model.E_high], lr=train_config.lr)
    while epoch < alignment_epochs + train_config.fine_tune_epochs:
        optimizer.zero_grad()
        obs_loss, interv_loss = best_model(E, temp=0.1, hard=True)
        loss = obs_loss + interv_loss

        scale_loss = train_config.scale_pen * train_config.temp_min * best_model.get_scale_penalty()


        total_loss = loss + scale_loss
        if loss.item() < best_val_loss:
            best_val_loss = loss.item()
            best_state = {k:v.cpu().clone() for k,v in best_model.state_dict().items()}
        history['obs_loss'].append(obs_loss.item())
        history['interv_loss'].append(interv_loss.item())
        history['entropy'].append(train_config.entropy_pen  *  model.get_assignment_entropy().item())
        history['surjectivity'].append(train_config.surjectivity_pen * model.get_surjectivity_penalty().item())
        history['scale'].append(model.get_scale_penalty().item())

        if best_val_loss <= train_config.acceptable_loss:
            log(f"Early stopping at epoch {epoch} with loss {best_val_loss}",verbose=train_config.verbose)
            break

        total_loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            log(f"Fine-tuning Epoch {epoch}, ObsLoss: {obs_loss.item()}, IntervLoss: {interv_loss.item()}, ScaleLoss: {scale_loss.item()}",train_config.verbose)
        
        epoch += 1
    
    best_model.load_state_dict(best_state)
    best_model.eval()
    with torch.no_grad():
        final_obs_loss, final_interv_loss = best_model(E, temp=0.1, hard=True)
        final_loss = final_obs_loss + final_interv_loss
        final = {'obs_loss':final_obs_loss, 'interv_loss':final_interv_loss, 'alignment_epochs':alignment_epochs, 'total_epochs':epoch}
        log(f"Final Loss after fine-tuning: {final_loss.item()}",train_config.verbose)

    return best_model, history, final

def gg_output(model,E_low):
    T_pred_og,partition_mask_og = model.get_T()
    T_pred = T_pred_og[:,:-1].detach().cpu().numpy()
    partition_mask = partition_mask_og[:,:-1].detach().cpu().numpy()

    E_high_og = model.get_E_high()
    E_high = E_high_og[:-1,:-1].detach().cpu().numpy()
    S_pred_og = model.get_S(E_low,T_pred_og,partition_mask_og).detach().cpu().numpy() #(E_low @ T_pred @ np.linalg.inv(E_high)) * partition_mask
    S_pred = S_pred_og[:,:-1]

    return partition_mask, T_pred, S_pred, E_high

if __name__ == "__main__":
    seed = None #23 sometimes gives an example of zero loss but graphical violations
    n_low = 5
    n_high = 3
    marg_prob = 0.2
    T_lower = (-2,1)
    T_upper = (-1,2)
    edge_density = 0.4

    A_low,T_gt,S_gt,A_high_gt = sample_low_level_graph(n_high=n_high,n_low=n_low,density=edge_density,T_lower=T_lower,T_upper=T_upper,marg_prob=marg_prob,seed=seed,shuffle=True,internal=True)
    E_low = np.linalg.inv(np.eye(n_low) - A_low)
    E_low_tensor = torch.from_numpy(E_low).to(torch.float32)
    
    plot_matrix(
        matrix_list=[A_high_gt,T_gt,S_gt],
        title_list=['High-Level Adjacency','Endogenous Abstraction T','Exogenous Abstraction S'],
        fig_title='Ground Truth Matrices'
    )
    visualize_causal_graph(A_low, A_high_gt, T_gt, S_gt, title='Ground Truth Causal Abstraction')

    ORACLE = True
    fine_tune_epochs = int(500 * n_high * np.log10(n_low))
    alignment_epochs = int(500 * n_low/n_high * np.log10(n_low))
    if ORACLE:
        E_high_gt = torch.linalg.inv(torch.eye(n_high) - A_high_gt)
        train_config = TrainingConfig(alignment_epochs=0, fine_tune_epochs=fine_tune_epochs, lr=0.05, acceptable_loss=0, entropy_pen=0., scale_pen=1e-6, verbose=True, threshold= 1e-8, temp=10.0, temp_min=0.1, anneal_rate=0.0005, hard=True)
        start_time = time.time()
        model,history,_ = gumbel_grad(E_low_tensor,n_high,train_config,W=torch.tensor(S_gt != 0).float(),E_high_mask=(E_high_gt!=0)) 
    else:
        train_config = TrainingConfig(alignment_epochs=alignment_epochs, fine_tune_epochs=fine_tune_epochs, lr=0.01, acceptable_loss=0, entropy_pen=0., scale_pen=1e-6, verbose=True, threshold=1e-8, temp=10.0, temp_min=0.1, anneal_rate=0.0005, hard=True)
        start_time = time.time()
        model,history,_ = gumbel_grad(E_low_tensor,n_high,train_config,W=None)
    
    print(f"Completed in {time.time()-start_time:.4f}s")
    partition_mask, T_pred, S_pred, E_high = gg_output(model,E_low_tensor)
    active_nodes = (partition_mask.sum(axis=0) > 0).astype(float) # Shape: (n_high,)
    A_high = np.eye(n_high) - np.linalg.inv(E_high)
    observed_loss = np.linalg.norm(S_pred@E_high - E_low @ T_pred, ord=1)

    print(f"Observed Loss (L1 norm): {observed_loss:.4f}")
    print(f"Active nodes (high-level): {active_nodes}")
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
    ax[0].plot(range(1000,len(history['obs_loss']),10), history['obs_loss'][1000::10])
    ax[0].plot(range(1000,len(history['interv_loss']),10), history['interv_loss'][1000::10])
    ax[0].legend(['Observational Loss','Interventional Loss'])
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[1].plot(range(0,len(history['scale']),10), history['scale'][::10])
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Scale")
    plt.show()
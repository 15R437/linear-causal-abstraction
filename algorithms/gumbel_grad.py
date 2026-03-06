import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        S = (E @ T @ torch.linalg.inv(E_high)) * partition_mask
        threshold_mask = (torch.abs(S) > self.threshold).float()
        return (threshold_mask * S - S).detach() + S
    
    def get_surjectivity_penalty(self):
        hard_mask = F.one_hot(torch.argmax(self.W,dim=1), num_classes=self.n_high+1).float()
        soft_probs = F.softmax(self.W, dim=1)
        threshold_mask = (torch.abs(self.T)>self.threshold).float()
        effective_mask = (hard_mask - soft_probs).detach() + soft_probs
        
        col_sums = effective_mask.sum(dim=0)
        target_sums = torch.ones_like(col_sums)
        target_sums[-1] = 0

        return torch.sum(F.relu(target_sums - col_sums))
    
    def forward(self,E,temp,hard=False):
        E = torch.tensor(E).float()
        T,partition_mask = self.get_T(temp=temp,hard=hard)
        E_tilde = self.get_E_high(partition_mask)
        S = self.get_S(E,T,partition_mask)

        obs_loss = self.n_high * torch.norm((S@E_tilde -E@T), p=2)**2 # how far is S from S * partition_mask?
        interv_loss = 0 
        for j in range(self.n_high+1):
            #interventional loss
            e_j = F.one_hot(torch.tensor(j),num_classes=self.n_high+1).float().reshape(self.n_high+1,1)
            F_ss = torch.diag((partition_mask@e_j).squeeze())  #low-level selection matrix
            F_jj = (e_j@e_j.T)  #high-level selection matrix
            I = torch.eye(self.n_low)
            residual = (E-I) @ (torch.linalg.solve((I+F_ss @ (E-I)),I)@F_ss @ E @ T - T @ F_jj @ E_tilde)
            interv_loss += torch.norm(residual, p=2)**2

        return obs_loss , interv_loss
    
@dataclass
class TrainingConfig:
    max_epochs: int = 10000
    lr: float = 0.01
    acceptable_loss: float = 1e-4
    entropy_pen: float = 0.
    surjectivity_pen: float = 10.
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
    history = {'obs_loss': [], 'interv_loss': [], 'entropy': []}
    best_val_loss = float('inf')
    best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
    
    log("Starting Gumbel-Softmax training...")
    max_epochs = train_config.max_epochs
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        obs_loss, interv_loss = model(E, temp, hard=train_config.hard)
        loss = obs_loss + interv_loss

        entropy_loss = train_config.entropy_pen  *  model.get_assignment_entropy().item()
        surjectivity_loss = train_config.surjectivity_pen * model.get_surjectivity_penalty().item()
        total_loss = loss + entropy_loss + surjectivity_loss
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
        history['entropy'].append(model.get_assignment_entropy().item())
        if best_val_loss <= train_config.acceptable_loss:
            log(f"Early stopping at epoch {epoch} with loss {best_val_loss}")
            max_epochs = epoch
            break
        
        total_loss.backward()
        optimizer.step()
        temp = max(train_config.temp_min, temp* (1 - train_config.anneal_rate))
        if epoch % 100 == 0:
            log(f"Epoch {epoch}, ObsLoss: {obs_loss.item()}, IntervLoss: {interv_loss.item()}, SurjectivityLoss: {surjectivity_loss}, Temp: {temp}",train_config.verbose)

    best_model = CausalLearner(n_low, n_high, threshold=train_config.threshold)
    best_model.load_state_dict(best_state)
    best_model.eval()
    best_model.W.requires_grad = False
    #freeze W and fine-tune T and E_high
    log("Starting fine-tuning of T and E_high...")
    optimizer = torch.optim.Adam([best_model.T, best_model.E_high], lr=train_config.lr)
    for epoch in range(max_epochs, max_epochs + train_config.fine_tune_epochs):
        optimizer.zero_grad()
        obs_loss, interv_loss = best_model(E, temp=0.1, hard=True)
        loss = obs_loss + interv_loss

        T,partition_mask = best_model.get_T()
        col_norms = torch.norm(T,p=2,dim=0)/n_low
        active_nodes = (partition_mask.sum(dim=0)>0).float()
        margin = 10 * best_model.threshold
        target_norms = torch.ones_like(col_norms) * 0.1
        target_norms[-1] = 0
        scale_loss = train_config.scale_pen * torch.sum(active_nodes * F.relu(target_norms - col_norms))


        total_loss = loss + scale_loss
        if loss.item() < best_val_loss:
            best_val_loss = loss.item()
            best_state = {k:v.cpu().clone() for k,v in best_model.state_dict().items()}
        history['obs_loss'].append(obs_loss.item())
        history['interv_loss'].append(interv_loss.item())
        history['entropy'].append(best_model.get_assignment_entropy().item())
        total_loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            log(f"Fine-tuning Epoch {epoch}, ObsLoss: {obs_loss.item()}, IntervLoss: {interv_loss.item()}, ScaleLoss: {scale_loss.item()}",train_config.verbose)
    
    best_model.load_state_dict(best_state)
    best_model.eval()
    with torch.no_grad():
        final_obs_loss, final_interv_loss = best_model(E, temp=0.1, hard=True)
        final_loss = final_obs_loss + final_interv_loss
        log(f"Final Loss after fine-tuning: {final_loss.item()}")

    return best_model, history

if __name__ == "__main__":
    seed = None #23 sometimes gives an example of zero loss but graphical violations
    n_low = 5
    n_high = 3
    marg_prob = 0.2
    T_lower = (-2,1)
    T_upper = (-1,2)
    edge_density = 0.2

    A_low,T_gt,S_gt,A_high_gt = sample_low_level_graph(n_high=n_high,n_low=n_low,density=edge_density,T_lower=T_lower,T_upper=T_upper,marg_prob=marg_prob,seed=seed)
    E_low = np.linalg.inv(np.eye(n_low) - A_low)
    
    plot_matrix(
        matrix_list=[A_high_gt,T_gt,S_gt],
        title_list=['High-Level Adjacency','Endogenous Abstraction T','Exogenous Abstraction S'],
        fig_title='Ground Truth Matrices'
    )
    visualize_causal_graph(A_low, A_high_gt, T_gt, S_gt, title='Ground Truth Causal Abstraction')

    ORACLE = False
    if ORACLE:
        E_high_gt = torch.linalg.inv(torch.eye(n_high) - A_high_gt)
        train_config = TrainingConfig(max_epochs=0, lr=0.01, acceptable_loss=0, entropy_pen=0., verbose=True, threshold= 1e-8, temp=10.0, temp_min=0.1, anneal_rate=0.0005, hard=True, fine_tune_epochs=5000)
        model,history = gumbel_grad(E_low,n_high,train_config,W=torch.tensor(S_gt != 0).float(),E_high_mask=(E_high_gt!=0))
    else:
        train_config = TrainingConfig(max_epochs=4000, lr=0.01, acceptable_loss=0, entropy_pen=0., verbose=True, temp=10.0, temp_min=0.1, anneal_rate=0.0005, hard=True, fine_tune_epochs=1000)
        model,history = gumbel_grad(E_low,n_high,train_config,W=None)
    
    
    T_pred,partition_mask = model.get_T()
    T_pred = T_pred[:,:-1].detach().cpu().numpy()
    partition_mask = partition_mask[:,:-1].detach().cpu().numpy()
    active_nodes = (partition_mask.sum(axis=0) > 0).astype(float) # Shape: (n_high,)
    print(f"Active nodes (high-level): {active_nodes}")
    E_high = model.get_E_high()[:-1,:-1].detach().cpu().numpy()
    S_pred = (E_low @ T_pred @ np.linalg.inv(E_high)) * partition_mask
    A_high = np.eye(n_high) - np.linalg.inv(E_high)
    observed_loss = np.linalg.norm(S_pred@E_high - E_low @ T_pred, ord=1)
    print(f"Observed Loss (L1 norm): {observed_loss:.4f}")
    print(f"E_low:\n{E_low}")
    print(f"T_pred:\n{T_pred}")
    print(f"S_pred:\n{S_pred}")
    print(f"E_high:\n{E_high}")
    
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
    ax[1].plot(range(0,len(history['entropy']),10), history['entropy'][::10])
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Entropy")
    plt.show()
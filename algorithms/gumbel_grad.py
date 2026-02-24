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
from utils.generate_data import random_dag, generate_random_T
from causabs.sampling import sample_linear_concretization

class CausalLearner(nn.Module):
    def __init__(self,n_low,n_high,eps=0.1,W=None,T=None,E_high=None):
        super().__init__()
        self.n_low = n_low
        self.n_high = n_high
        self.eps = eps
        self.set_W(W)
        self.set_T(T)
        self.set_E_high(E_high)

        self.register_buffer('mask', torch.triu(torch.ones(n_high, n_high), diagonal=1))
        self.register_buffer('identity', torch.eye(n_high))
    
    def set_W(self,W=None,requires_grad=True):
        if W is None:
            self.W = nn.Parameter(torch.randn(self.n_low,self.n_high),requires_grad=requires_grad)
        else:
            assert(W.shape == (self.n_low,self.n_high)),f"Expected shape {(self.n_low,self.n_high)}, got {W.shape}."
            self.W = nn.Parameter(W,requires_grad=requires_grad)
    
    def set_T(self,T=None,requires_grad=True):
        if T is None:
            self.T = nn.Parameter(torch.randn(self.n_low,self.n_high),requires_grad=requires_grad)
        else:
            assert(T.shape == (self.n_low,self.n_high)),f"Expected shape {(self.n_low,self.n_high)}, got {T.shape}."
            self.T = nn.Parameter(T,requires_grad=requires_grad)

    def set_E_high(self,E_high=None,requires_grad=True):
        if E_high is None:
            self.E_high = nn.Parameter(torch.zeros(self.n_high,self.n_high),requires_grad=requires_grad)
        else:
            assert(E_high.shape == (self.n_high,self.n_high)),f"Expected shape {(self.n_high,self.n_high)}, got {E_high.shape}."
            self.E_high = nn.Parameter(E_high,requires_grad=requires_grad)
    
    def forward(self,E,temp,hard=False):
        if self.training:
            T_mask = F.gumbel_softmax(self.W,tau=temp,hard=hard,dim=1)
        else:
            T_mask = F.one_hot(torch.argmax(self.W,dim=1),num_classes=self.n_high).float()

        E = torch.tensor(E).float()
        T = T_mask*self.T
        E_tilde = self._E_high()

        #impose causal sufficiency
        partition = F.one_hot(torch.argmax(self.W,dim=1),num_classes=self.n_high).float()
        S = E @ T @ torch.linalg.inv(E_tilde)
        obs_loss = self.n_high * torch.norm((S*T_mask)@E_tilde -E@T, p=2)**2 #penalize any contribution from 'inactive' high-level nodes (those with no low-level nodes assigned)
        #obs_loss = self.n_high * torch.norm(T @ E_tilde - E @ T, p=2)**2 #assuming S = T
        interv_loss = 0 
        for j in range(self.n_high):
            #interventional loss
            e_j = F.one_hot(torch.tensor(j),num_classes=self.n_high).float().reshape(self.n_high,1)
            F_ss = torch.diag((T_mask@e_j).squeeze())  #low-level selection matrix
            F_jj = (e_j@e_j.T)  #high-level selection matrix
            I = torch.eye(self.n_low)
            residual = (E-I) @ (torch.linalg.solve((I+F_ss @ (E-I)),I)@F_ss @ E @ T - T @ F_jj @ E_tilde)
            interv_loss += torch.norm(residual, p=2)**2
            #loss += torch.norm(residual, p='fro')**2

        return obs_loss , interv_loss
    
    def get_assignment_entropy(self):
        probs = torch.softmax(self.W, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        return entropy.mean()
    
    def _T(self):
        partition = F.one_hot(torch.argmax(self.W,dim=1),num_classes=self.n_high).float()
        T = partition * self.T
        return T
    
    def _E_high(self):
        T_mask = F.one_hot(torch.argmax(self.W,dim=1),num_classes=self.n_high).float()
        active_nodes = (T_mask.sum(dim=0) > 0).float() # Shape: (n_high,)
        interaction_mask = active_nodes.unsqueeze(0) * active_nodes.unsqueeze(1)
        
        # Apply logic:
        # 1. Start with strictly upper triangular E_high
        # 2. Mask out interactions involving 'inactive' high-level nodes (those with no low-level nodes assigned)
        # 3. Add the FULL identity matrix back (ensures diag is all 1s -> determinant is 1)
        E_high = (self.E_high * self.mask) * interaction_mask + self.identity
        return E_high
@dataclass
class TrainingConfig:
    max_epochs: int = 10000
    lr: float = 0.01
    acceptable_loss: float = 1e-4
    entropy_pen: float = 0.
    verbose: bool = True
    temp: float = 5.0
    temp_min: float = 0.1
    anneal_rate: float = 0.0005
    hard: bool = True
    fine_tune_epochs: int = 2000
def log(msg,verbose=True):
        if verbose: print(msg)

def gumbel_grad(E,n_high,train_config=TrainingConfig(),W=None,**kwds):
    n_low = E.shape[0]
    model = CausalLearner(n_low, n_high)
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
        entropy = model.get_assignment_entropy().item()
        entropy_loss = train_config.entropy_pen  *  entropy
        total_loss = loss + entropy_loss
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
            log(f"Epoch {epoch}, ObsLoss: {obs_loss.item()}, IntervLoss: {interv_loss.item()}, Temp: {temp}",train_config.verbose)

    best_model = CausalLearner(n_low, n_high)
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
        if loss.item() < best_val_loss:
            best_val_loss = loss.item()
            best_state = {k:v.cpu().clone() for k,v in best_model.state_dict().items()}
        history['obs_loss'].append(obs_loss.item())
        history['interv_loss'].append(interv_loss.item())
        history['entropy'].append(best_model.get_assignment_entropy().item())
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            log(f"Fine-tuning Epoch {epoch}, ObsLoss: {obs_loss.item()}, IntervLoss: {interv_loss.item()}")
    
    best_model.load_state_dict(best_state)
    best_model.eval()
    with torch.no_grad():
        final_obs_loss, final_interv_loss = best_model(E, temp=0.1, hard=True)
        final_loss = final_obs_loss + final_interv_loss
        log(f"Final Loss after fine-tuning: {final_loss.item()}")

    return best_model, history

if __name__ == "__main__":
    seed = None
    n_low = 5
    n_high = 3
    #A_low = random_dag(n_low, edge_density=0.5, weight_mean=0, weight_std=1, seed=seed)
    A_high_gt = random_dag(n_high, edge_density=0.5, weight_mean=0, weight_std=2, seed=seed)
    T_gt,block_size = generate_random_T(n_low, n_high, T_range=(-1.5, 1.5), seed=seed)
    A_low,S_gt= sample_linear_concretization(A_high_gt,T_gt,block_size,internal=False)
    E_low = np.linalg.inv(np.eye(n_low) - A_low)
    
    plot_matrix(
        matrix_list=[A_high_gt,T_gt,S_gt],
        title_list=['High-Level Adjacency','Endogenous Abstraction T','Exogenous Abstraction S'],
        fig_title='Ground Truth Matrices'
    )
    visualize_causal_graph(A_low, A_high_gt, T_gt, S_gt, title='Ground Truth Causal Abstraction')

    ORACLE = True
    if ORACLE:
        train_config = TrainingConfig(max_epochs=0, lr=0.01, acceptable_loss=0, entropy_pen=0., verbose=True, temp=10.0, temp_min=0.1, anneal_rate=0.0005, hard=True, fine_tune_epochs=5000)
        model,history = gumbel_grad(E_low,n_high,train_config,W=torch.tensor(T_gt != 0).float())
    else:
        train_config = TrainingConfig(max_epochs=4000, lr=0.01, acceptable_loss=0, entropy_pen=0., verbose=True, temp=10.0, temp_min=0.1, anneal_rate=0.0005, hard=True, fine_tune_epochs=1000)
        model,history = gumbel_grad(E_low,n_high,train_config,W=None)
    
    
    partition = F.one_hot(torch.argmax(model.W,dim=1),num_classes=model.n_high).float().detach().cpu().numpy()
    T_pred = model.T.detach().cpu().numpy() * partition
    active_nodes = (partition.sum(axis=0) > 0).astype(float) # Shape: (n_high,)
    print(f"Active nodes (high-level): {active_nodes}")
    E_high = model._E_high().detach().cpu().numpy()
    S_pred = E_low @ T_pred @ np.linalg.inv(E_high) * partition
    A_high = np.eye(n_high) - np.linalg.inv(E_high)
    observed_loss = np.linalg.norm(S_pred@E_high - E_low @ T_pred, ord=1)
    print(f"Observed Loss (L1 norm): {observed_loss:.4f}")
    print(f"E_low:\n{E_low}")
    print(f"T_pred:\n{T_pred}")
    print(f"E_high:\n{E_high}")
    print(f"S_pred:\n{S_pred}")

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
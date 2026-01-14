import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
            self.E_high = nn.Parameter(torch.randn(self.n_high,self.n_high),requires_grad=requires_grad)
        else:
            assert(E_high.shape == (self.n_high,self.n_high)),f"Expected shape {(self.n_high,self.n_high)}, got {E_high.shape}."
            self.E_high = nn.Parameter(E_high,requires_grad=requires_grad)
    
    def forward(self,E,temp,hard=False):
        T_mask = F.gumbel_softmax(self.W,tau=temp,hard=hard,dim=1)
        T = T_mask*self.T
        loss = 0
        for j in range(self.n_high):
            e_j = F.one_hot(torch.tensor(j),num_classes=self.n_high).float().resize(self.n_high,1) #high-level vector
            F_ss = torch.diag((T_mask@e_j).squeeze())
            F_jj = e_j@e_j.T
            I = torch.eye(self.n_low)
            E_tilde = self.E_high * self.mask + self.identity
            residual = (I-E) @ F_ss @ T -T@F_jj + E @ T @ torch.linalg.inv(E_tilde) @ F_jj @ E_tilde
            loss += torch.norm(residual, p='fro')**2

        return loss
    
    def _T(self):
        T_mask = F.gumbel_softmax(self.W,hard=True,dim=1)
        T = T_mask*self.T
        return T
    
    def _E_high(self):
        return self.E_high * self.mask + self.identity

if __name__ == "__main__":
    n_low = 25
    n_high = 5
    model = CausalLearner(n_low, n_high)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    E = torch.randn(n_low, n_low) * torch.triu(torch.ones(n_low, n_low), diagonal=1) + torch.eye(n_low)
    temp = 5.0
    temp_min = 0.1
    anneal_rate = 0.001
    alternating_schedule = 1
    loss_history = []
    print("Starting training...")
    for epoch in range(1000):
        optimizer.zero_grad()
        loss = model(E, temp, hard=True)
        loss_history.append(loss.item())
        loss.backward()
        optimizer.step()
        temp = max(temp_min, temp* (1 - anneal_rate))
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}, Temp: {temp}")
    
    print("Training completed.")
    print(f"Final Loss: {loss.item():6f}")
    print("Learned T matrix:")
    print(model._T())
    print("Learned E_high matrix:")
    print(model._E_high())
    print("Ground Truth E matrix:")
    #print(E)

    #Plot a graph of loss vs epochs
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(range(0,1000,10), loss_history[::10])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    plt.show()
    #Freeze W and fine-tune T and E_high
    #import pdb;pdb.set_trace()
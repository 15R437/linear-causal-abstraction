import torch
import torch.nn as nn
import torch.nn.functional as F

def stick_breaking(w):
    one_minus_w = 1 - w # [1-w1, 1-w2, ...]
    
    cumprod_one_minus_w = torch.cumprod(one_minus_w, dim=1) # [1-w1, (1-w1)(1-w2), ...]

    shifted_cumprod = torch.cat([torch.ones(w.shape[0], 1,device=w.device), cumprod_one_minus_w[:, :-1]], dim=1) # [1, 1-w1, (1-w1)(1-w2), ...]
    
    return w * shifted_cumprod

class HailMary(nn.Module):
    def __init__(self, n_low, n_high, E, init_assignment=None,threshold=0.5):
        super().__init__()
        self.n_low = n_low
        self.n_high = n_high
        self.E = E
        self.device = E.device
        self.W_param = nn.Parameter(torch.randn(n_low, n_high,device=self.device))
        self.A_param = nn.Parameter(torch.randn(n_low, n_high,device=self.device))
        self.threshold = threshold
        W_mask = torch.ones(n_low, n_high,device=self.device)
        if init_assignment:
            assignment_tensor = torch.tensor(init_assignment, device=self.device).long()
            valid_mask = (assignment_tensor < self.n_high) & (assignment_tensor != -1)
            low_level_indices = torch.where(valid_mask)[0]
            high_level_indices = assignment_tensor[valid_mask]
            W_mask[low_level_indices] = 0.0
            W_mask[low_level_indices, high_level_indices] = 1.0

        self.register_buffer('W_mask', W_mask)
    
    def get_params(self):
        W = torch.sigmoid(self.W_param) * self.W_mask
        A = torch.sigmoid(self.A_param) * self.W_mask

        T = stick_breaking(W)

        return T, W, A

    def get_S(self,E_high,A,T):
        s_mask = (A > self.threshold)
        S = torch.linalg.solve(E_high.T, (self.E @ T).T).T * s_mask
        return S
    
    def forward(self):
        T, W, A = self.get_params() # all (n_low, n_high)
        E = self.E
        constraint_loss = torch.mean((W - F.relu(W + A - self.threshold - 1) - F.relu(W + A - self.threshold))**2)
        regularisation_loss = F.binary_cross_entropy(W,A)
        
        I = torch.eye(self.n_low,device=self.device)
        I_high = torch.eye(self.n_high,device=self.device)
        F_pi = torch.diag_embed(T.T) #(n_high, n_low, n_low)
    
        term = I + F_pi @ (E - I)
        ET = E @ T
        ET_batch = ET.unsqueeze(0).expand(self.n_high,-1,-1)
        T_batch = T.unsqueeze(0).expand(self.n_high, -1, -1)
        term_T = torch.bmm(term, T_batch)  # (n_high, n_low, n_high) 
        
        lhs = torch.bmm(F_pi,ET_batch)
        rhs_col = term_T[torch.arange(self.n_high), :, torch.arange(self.n_high)]  # (n_high, n_low)
        rhs = rhs_col.unsqueeze(-1).expand(-1, -1, self.n_high)  # (n_high, n_low, n_high)

        dot_prod = torch.sum(lhs * rhs, dim=1)
        a_norm = torch.sum(lhs * lhs, dim=1)
        b_norm = torch.sum(rhs * rhs, dim=1)
        safe_b = torch.where(b_norm > 0, b_norm, torch.ones_like(b_norm))
        E_high_ij = torch.where(b_norm > 0, dot_prod / safe_b, torch.zeros_like(dot_prod))

        
        E_high_mask = torch.triu(torch.ones_like(E_high_ij), diagonal=1)
        E_high =  E_high_ij * E_high_mask + I_high
        #E_high = (E_high_ij * E_high_mask + I_high - E_high_ij).detach() + E_high_ij
        #gram_loss = torch.mean((a_norm * b_norm) - dot_prod ** 2)
        mse_loss = torch.mean((lhs - E_high.unsqueeze(-1) * rhs)**2)
        intv_loss = torch.relu(mse_loss * self.n_high) #torch.relu(mse_loss).mean()

        S = self.get_S(E_high=E_high,A=A,T=T)
        obs_loss = torch.mean((S @ E_high - self.E @ T)**2)

        return obs_loss, intv_loss, constraint_loss, regularisation_loss, E_high, S, T, W, A
    
    def train_step(self, optimizer):
        self.train()
        optimizer.zero_grad()
        obs_loss, intv_loss, constraint_loss, regularisation_loss, _, _, _, _, _ = self.forward()
        total_loss = obs_loss + intv_loss + constraint_loss + regularisation_loss
        total_loss.backward()
        optimizer.step()
        return obs_loss.item(), intv_loss.item(), constraint_loss.item(), regularisation_loss.item()
    
    def _log(msg,verbose=True):
        if verbose:
            print(msg)
    
    def train(self, num_epochs, verbose=True):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        for epoch in range(num_epochs):
            obs_loss, intv_loss, constraint_loss, regularisation_loss = self.train_step(optimizer)
            self._log(f'Epoch {epoch+1}/{num_epochs}, Obs Loss: {obs_loss:.4f}, Intv Loss: {intv_loss:.4f}, Constraint Loss: {constraint_loss:.4f}, Regularisation Loss: {regularisation_loss:.4f}', verbose=verbose)
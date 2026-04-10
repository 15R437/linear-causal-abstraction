import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.abspath(os.getcwd()))

from utils.plots import plot_matrix, visualize_causal_graph
from utils.generate_data import sample_low_level_graph

class GradSplit(nn.Module):
    def __init__(self,E):
        super().__init__()
    
    def forward(self):
        
        pass
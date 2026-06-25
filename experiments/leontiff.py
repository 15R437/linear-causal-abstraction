import sys
import os
sys.path.insert(0, os.path.abspath(os.getcwd()))

import pandas as pd
import numpy as np
import pymrio
from src.utils.helper import enforce_dag

try:
    A_low = np.load("data/A_low.npy")
    print(f"A_low loaded. Shape: {A_low.shape}.")
except:
    exio_data = pymrio.parse_exiobase3(path="data/IOT_2024_ixi.zip")
    print(exio_data.extensions)
    exio_data.calc_all()

    available_regions = exio_data.get_regions()
    available_sectors = exio_data.get_sectors()

    A_full = exio_data.A
    S_full = exio_data.factor_inputs.S  # Shape: (Total_K, N)

    target_regions = ['GB']

    idx = pd.IndexSlice

    A_filtered = A_full.loc[idx[target_regions, :], idx[target_regions, :]]

    S_filtered = S_full.loc[:, idx[target_regions, :]]
    target_factors = S_filtered.index.tolist()

    ST = S_filtered.T  # Shape: (N, K)

    K = len(target_factors)
    N = A_filtered.shape[0]

    zeros_bottom_left = pd.DataFrame(
        0.0, index=target_factors, columns=A_filtered.columns
    )
    zeros_bottom_right = pd.DataFrame(
        0.0, index=target_factors, columns=target_factors
    )

    # Horizontal concatenation of top blocks and bottom blocks
    top_block = pd.concat([A_filtered, ST], axis=1)
    bottom_block = pd.concat([zeros_bottom_left, zeros_bottom_right], axis=1)

    # Vertical concatenation to yield the final square weight matrix W
    A_low = pd.concat([top_block, bottom_block], axis=0).to_numpy()  # Shape: (N+K, N+K)
    print(A_low.shape)
    np.save("data/A_low.npy")

A_low = enforce_dag(A_low)

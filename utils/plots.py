import sys
import os
sys.path.insert(0, os.path.abspath(os.getcwd()))

import matplotlib.pyplot as plt
import torch
from algorithms.hail_mary import HailMary
import seaborn as sns
import numpy as np
import networkx as nx
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MultipleLocator,ScalarFormatter,MaxNLocator
import matplotlib.cm as cm
from itertools import combinations

def visualize_causal_graph(A_low, A_high, T_matrix, S_matrix, threshold=0,title='Causal Abstraction Graph'):
    """
    Visualizes low-level nodes, high-level nodes, and the learned abstraction.
    """
    n_low = A_low.shape[0]
    n_high = A_high.shape[0]

    low_nodes_endog = [f"X{i+1}" for i in range(n_low)]
    low_nodes_exog = [f"U{i+1}" for i in range(n_low)]
    high_nodes_endog = [f"Y{i+1}" for i in range(n_high)]
    high_nodes_exog = [f"W{i+1}" for i in range(n_high)]

    #postioning
    # ... inside your function ...

    # 1. Layout for Low-Level Graph
    G_low = nx.DiGraph()
    G_low.add_nodes_from(low_nodes_endog)
    G_low.add_nodes_from(low_nodes_exog)
    pos_low = nx.spring_layout(G_low, seed=42,scale=1.0)
    
    # 2. Layout for High-Level Graph
    G_high = nx.DiGraph()
    G_high.add_nodes_from(high_nodes_endog)
    G_high.add_nodes_from(high_nodes_exog)
    pos_high = nx.spring_layout(G_high, seed=42,scale=1.0)
    for node in pos_high:
        pos_high[node][0] += 3.0  # Shift right

    # Combine positions
    pos = {**pos_low, **pos_high}
    fig,ax = plt.subplots(figsize=(10, 8))
    def draw_level_box(nodes, pos, color, label):
        coords = np.array([pos[node] for node in nodes])
        x_min, y_min = coords.min(axis=0) - 0.2
        x_max, y_max = coords.max(axis=0) + 0.2
        buffer = 0.4
        rect = Rectangle((x_min-buffer, y_min-buffer), x_max - x_min + 2*buffer, y_max - y_min + 2*buffer, 
                         linewidth=2, edgecolor=color, facecolor=color, alpha=0.1, linestyle='--')
        ax.add_patch(rect)
        ax.text(x_min + (x_max-x_min)/2, y_max + buffer+0.05, label, 
                fontsize=12, fontweight='bold', ha='center', color=color)

    draw_level_box(low_nodes_endog + low_nodes_exog, pos, 'blue', 'Low-Level')
    draw_level_box(high_nodes_endog + high_nodes_exog, pos, 'purple', 'High-Level')

    def add_nearby_node(endog_pos, shift_x=-0.3,shift_y=0.3):
        return endog_pos + np.array([shift_x, shift_y]) # Up and Left

    # Map low-level exog to endog
    for i in range(n_low):
        endog_name = low_nodes_endog[i]
        if endog_name in pos:
            pos[low_nodes_exog[i]] = add_nearby_node(pos[endog_name],shift_x=-0.3,shift_y=0.15)
        else:
            # Fallback if node has no edges
            pos[endog_name] = np.array([0, i * 0.5])
            pos[low_nodes_exog[i]] = add_nearby_node(pos[endog_name],shift_x=-0.3,shift_y=0.15)

    # Map high-level exog to endog
    for i in range(n_high):
        endog_name = high_nodes_endog[i]
        if endog_name in pos:
            pos[high_nodes_exog[i]] = add_nearby_node(pos[endog_name],shift_x=0.4,shift_y=-0.2)
        else:
            # Fallback if node has no edges
            pos[endog_name] = np.array([2.0, i * 0.5])
            pos[high_nodes_exog[i]] = add_nearby_node(pos[endog_name],shift_x=0.4,shift_y=-0.2)
    G = nx.DiGraph()
    G.add_nodes_from(low_nodes_endog + low_nodes_exog + high_nodes_endog + high_nodes_exog)
    G.add_edges_from(G_low.edges())
    G.add_edges_from(G_high.edges())
    
    #Colour nodes based on their high-level cluster assignment
    cmap1 = cm.get_cmap('Set1',n_high)
    cmap2 = cm.get_cmap('Pastel1',n_high)
    endogenous_colours = {high_nodes_endog[i]: cmap1(i) for i in range(n_high)}
    exogenous_colours = {high_nodes_exog[i]: cmap2(i) for i in range(n_high)} 
    BIN_COLOUR = '#D3D3D3'
    low_endog_colour = []
    low_exog_colour = []
    for i in range(n_low):
        non_zero = np.where(T_matrix[i, :] != 0)[0]
        if non_zero.size>0:
            clst = np.where(T_matrix[i,:]!=0)[0][0]
            low_endog_colour.append(endogenous_colours[high_nodes_endog[clst]])
        else:
            low_endog_colour.append(BIN_COLOUR)

        non_zero = np.where(S_matrix[i, :] != 0)[0]
        if non_zero.size > 0:
            clst = np.where(S_matrix[i,:]!=0)[0][0]
            low_exog_colour.append(exogenous_colours[high_nodes_exog[np.where(S_matrix[i,:]!=0)[0][0]]])
        else:
            low_exog_colour.append(BIN_COLOUR)
    
    # exogenous edges
    for i in range(n_low):
        ax.annotate("", xy=pos[low_nodes_endog[i]], xytext=pos[low_nodes_exog[i]],
                    arrowprops=dict(arrowstyle="->", color="gray", alpha=0.3))
    
    for i in range(n_high):
        ax.annotate("", xy=pos[high_nodes_endog[i]], xytext=pos[high_nodes_exog[i]],
                    arrowprops=dict(arrowstyle="->", color="gray", alpha=0.3))
    for i in range(n_low):
        latent_confounding = combinations(np.where(S_matrix[i,:]!=0)[0], 2)
        for j,k in latent_confounding:
            ax.annotate("",
                        xy=pos[high_nodes_exog[j]], xycoords='data',
                        xytext=pos[high_nodes_exog[k]], textcoords='data',
                        arrowprops=dict(arrowstyle="-", 
                                        color="gray", 
                                        alpha=0.3, 
                                        ls='--', # Dashed
                                        lw=1.5,
                                        connectionstyle="arc3,rad=0.1"))

    
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='Pos. Causal Edge'),
        Line2D([0], [0], color='navy', lw=2, label='Neg. Causal Edge'),
        Patch(facecolor=BIN_COLOUR, edgecolor='black', label='Binned / Unassigned (X)')
    ]

    # Add mapping colors to legend
    for i in range(n_high):
        legend_elements.append(
            Patch(facecolor=endogenous_colours[high_nodes_endog[i]], 
                  edgecolor='black', label=f'High-Level Endogenous Cluster {i+1}')
        )

    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)

    # Place legend outside the main plot area to avoid overlap
    ax.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=True)
    
    
    # 3. Draw Low-Level Edges with Curves
    max_A_low_val = np.max(np.abs(A_low)) if np.max(np.abs(A_low)) > 0 else 1.0
    max_A_high_val = np.max(np.abs(A_high)) if np.max(np.abs(A_high)) > 0 else 1.0
    max_val = max(max_A_low_val, max_A_high_val)
    for i in range(len(low_nodes_endog)):
        for k in range(len(low_nodes_endog)):
            if abs(A_low[i, k]) > threshold:
                # Calculate "arc" based on distance to avoid overlap
                dist = abs(i - k)
                arc = 0.2 if dist > 1 else 0.0  # Curve if skipping nodes
                intensity = max(abs(A_low[i, k])/max_val, 0.25)
                edge_colour = "red" if A_low[i, k] > 0 else "navy"
                ax.annotate("",
                            xy=pos[low_nodes_endog[k]], xycoords='data',
                            xytext=pos[low_nodes_endog[i]], textcoords='data',
                            arrowprops=dict(arrowstyle="->", color=edge_colour,
                                            alpha=intensity,
                                            connectionstyle=f"arc3,rad={arc}",
                                            mutation_scale=15))
                #add edges from exogenous nodes to endogenous nodes


    # 5. Draw High-Level Edges (Red)
    
    for i in range(len(high_nodes_endog)):
        for k in range(len(high_nodes_endog)):
            if abs(A_high[i, k]) > threshold:
                intensity = max(abs(A_high[i, k])/max_val, 0.25)
                edge_colour = "red" if A_high[i, k] > 0 else "navy"
                ax.annotate("",
                            xy=pos[high_nodes_endog[k]], xycoords='data',
                            xytext=pos[high_nodes_endog[i]], textcoords='data',
                            arrowprops=dict(arrowstyle="->", color=edge_colour,
                                            alpha=intensity,
                                            connectionstyle="arc3,rad=0.2",
                                            mutation_scale=20))

    # Draw Nodes last so they are on top
    binned_low_nodes_endog, binned_low_nodes_exog = [low_nodes_endog[i] for i in range(n_low) if low_endog_colour[i] == BIN_COLOUR], [low_nodes_exog[i] for i in range(n_low) if low_exog_colour[i] == BIN_COLOUR]
    assigned_low_nodes_endog,assigned_low_nodes_exog = [low_nodes_endog[i] for i in range(n_low) if low_endog_colour[i] != BIN_COLOUR], [low_nodes_exog[i] for i in range(n_low) if low_exog_colour[i]!= BIN_COLOUR]
    assigned_colours_endog, assigned_colours_exog = [low_endog_colour[i] for i in range(n_low) if low_endog_colour[i] != BIN_COLOUR], [low_exog_colour[i] for i in range(n_low) if low_exog_colour[i] != BIN_COLOUR]
    nx.draw_networkx_nodes(G, pos, nodelist=assigned_low_nodes_endog, node_color=assigned_colours_endog, edgecolors='black',node_size=400)
    nx.draw_networkx_nodes(G, pos, nodelist=binned_low_nodes_endog, node_color=BIN_COLOUR, alpha = 0.4)

    nx.draw_networkx_nodes(G, pos, nodelist=assigned_low_nodes_exog, node_color=assigned_colours_exog, node_size=200, node_shape='s')
    nx.draw_networkx_nodes(G, pos, nodelist=binned_low_nodes_exog, node_color=BIN_COLOUR, alpha = 0.4)

    nx.draw_networkx_nodes(G, pos, nodelist=high_nodes_endog, node_color=[endogenous_colours[n] for n in high_nodes_endog], edgecolors='black', node_size=600)
    nx.draw_networkx_nodes(G, pos, nodelist=high_nodes_exog, node_color=[exogenous_colours[n] for n in high_nodes_exog], node_size=300, node_shape='s')
    nx.draw_networkx_labels(G, pos, font_size=10)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.show()

def plot_experiment_results(df):
    """
    Plots the metrics (Rand Index, Frobenius Norm, Runtime) from the experiment results.
    Assumes df has columns: ['method', 'repeat', 'metric', 'score']
    """
    # Filter out any None scores if a metric wasn't computed
    df_plot = df.dropna(subset=['score'])
    
    # Set the visual style
    sns.set_theme(style="whitegrid")
    metrics = df_plot['metric'].unique()
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    
    # If there's only one metric, wrap axes in a list to allow iteration
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        metric_data = df_plot[df_plot['metric'] == metric]
        
        # barplot automatically computes mean and 95% confidence interval (error bars)
        sns.barplot(
            data=metric_data, 
            x='method', 
            y='score', 
            hue='method',      # Assign hue to the same variable as x
            legend=False,      # Hide the legend since the x-axis labels cover it
            ax=ax, 
            palette='viridis',
            capsize=.1
        )
        
        ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=10, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_xlabel('Method')
        
        # Improve readability if method names are long
        if len(df_plot['method'].unique()) > 3:
            ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

def plot_matrix(matrix_list,title_list,fig_title=''):
    assert(len(matrix_list)==len(title_list)), f"Expected matrix_list and title_list to have the same length. Got lists of length {len(matrix_list)} and {len(title_list)}."
    n = len(matrix_list)
    fig, ax = plt.subplots(1,n, figsize=(5*n,5),layout='constrained')
    if len(matrix_list) == 1:
         ax = [ax]
    

    global_max = np.max([np.max(np.abs(E)) for E in matrix_list if E is not None])
    for idx,(E,title) in enumerate(zip(matrix_list,title_list)):
        im = ax[idx].imshow(E,cmap='bwr',vmin=-global_max,vmax=global_max)
        ax[idx].set_title(title)

        fig.colorbar(im,ax=ax[idx],fraction=0.046,pad=0.04,shrink=0.8)
    
    fig.suptitle(fig_title,fontsize=16,fontweight='bold')
    plt.show()

def plot_results(results_df,x_axis,y_axis,fixed_vars={},approaches=None,row=None,col=None):
    # 1. Filter data
    df = results_df.copy()
    if approaches:
        df = df[df['Approach'].isin(approaches)]
    
    for c, val in fixed_vars.items():
        if c in df.columns:               
            df = df[df[c] == val]
    
    
    #a grid of plots
    sns.set_theme(style="whitegrid")
    grid = sns.FacetGrid(df, row=row, col=col, hue='Approach', 
                    height=6, aspect=1.5, palette='Set2', sharey=False)

    grid.map_dataframe(sns.lineplot, x=x_axis, y=y_axis, marker='o', errorbar='se')

    if x_axis=='sample_size':
        for ax in grid.axes.flat:
            ax.set_xscale('log', base=2)
    elif x_axis=='k':
        for ax in grid.axes.flat:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=10,integer=True))
    grid.set_axis_labels(x_axis, f"Average {y_axis}")
    grid.add_legend(title="Approach")
    row_tmpl = "{row_name} " + row if row else ""
    col_tmpl = "{col_name} " + col if col else ""
    grid.set_titles(row_template=row_tmpl, col_template=col_tmpl)
    
    title_str = f"Average {y_axis}"
    if row or col:
        title_str += f" grouped by {row if row else ''} {col if col else ''}"
    grid.figure.suptitle(title_str, y=0.98)

    plt.show()

def plot_loss_landscape(A_low, T_gt, idx1, idx2=None, n_points=100, flip_prob=0.):
        """
        Visualises the loss wrt parameters in T.
        
        Params:
            A_low (np.ndarray): low-level adjacency matrix
            T_gt (np.ndarray): ground truth abstraction matrix
            idx1 (int): index of parameter in T to vary
            idx2 (int, optional): index of second parameter in T to vary.
        """

        n_low, n_high = T_gt.shape
        assert n_high == 2, "Only supports 2 high-level varaibles"

        T_gt = (T_gt != 0).astype(float) #convert to binary for visualisation
        W = (T_gt != 0).astype(float)
        flip_mask = (np.random.rand(W.shape[0]) < flip_prob)[:,None] * np.ones_like(W)
        W = (1 - W) * flip_mask + W * (1 - flip_mask)

        t1_values = np.linspace(0,1,n_points)
        model = HailMary(n_low=n_low, n_high=n_high, E=A_low, T_type='binary')
        if idx2 is None:
            losses = []
            for t1 in t1_values:
                W[idx1,0] = t1
                W[idx1,1] = 1-t1
                recon_loss, obs_loss, intv_loss, _, _, _, _ = model(W=torch.from_numpy(W).float())
                losses.append((recon_loss.item(), obs_loss.item(), intv_loss.item()))

            recon_losses, obs_losses, intv_losses = zip(*losses)
            plt.plot(t1_values, recon_losses, label='Reconstruction Loss')
            plt.plot(t1_values, obs_losses, label='Observation Loss')
            plt.plot(t1_values, intv_losses, label='Intervention Loss')
            plt.xlabel(f'T_value for low-level variable {idx1}')
            plt.ylabel('Loss')
            plt.title('Loss Landscape')
            plt.legend()
            plt.show()
        else:
            losses = []
            t2_values = np.linspace(0,1,n_points)
            for t1 in t1_values:
                row_losses = []
                W[idx1,0] = t1
                W[idx1,1] = 1-t1
                for t2 in t2_values:
                    W[idx2,0] = t2
                    W[idx2,1] = 1-t2
                    recon_loss, obs_loss, intv_loss, _, _, _, _ = model(W=torch.from_numpy(W).float())
                    row_losses.append((recon_loss.item(), obs_loss.item(), intv_loss.item()))
                losses.append(row_losses)
            
            fig, ax = plt.subplots(1,3,figsize=(10,8))
            for i, loss_name in enumerate(['Reconstruction Loss','Observation Loss','Intervention Loss']):
                specific_loss = np.array([[loss[i] for loss in row] for row in losses])
                im1 = ax[i].imshow(specific_loss, cmap='Blues', origin='lower')
                ax[i].set_title(loss_name)
                fig.colorbar(im1, ax=ax[i])  # Colorbar for subplot 1

                ax[i].set_xlabel(f'low-level variable {idx2}')
                ax[i].set_ylabel(f'low-level variable {idx1}')

                gt_x = T_gt[idx2, 0] * n_points  # scale from [0,1] to [0,n_points]
                gt_y = T_gt[idx1, 0] * n_points
                ax[i].scatter([gt_x], [gt_y], color='red', marker='*', s=200, label='Ground truth')
                #ax[i].legend()
            
            plt.show()
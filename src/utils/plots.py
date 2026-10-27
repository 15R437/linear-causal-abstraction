import sys
import os
sys.path.insert(0, os.path.abspath(os.getcwd()))

from typing import Literal, Callable
import matplotlib.pyplot as plt
import torch
#from src.models.hail_mary import HailMary
import seaborn as sns
import numpy as np
import networkx as nx
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator
import matplotlib.animation as animation
import matplotlib.cm as cm
from itertools import combinations
from src.utils.helper import get_S, get_A_high, regularised_surjective_init

def visualize_causal_graph(A_low, A_high, T_matrix, S_matrix, threshold=1e-5,title='Causal Abstraction Graph',save_path=None):
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
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()



def plot_matrix(matrix_list,title_list,fig_title=''):
    assert(len(matrix_list)==len(title_list)), f"Expected matrix_list and title_list to have the same length. Got lists of length {len(matrix_list)} and {len(title_list)}."
    n = len(matrix_list)
    if n == 1:
        width, height = 6.5, 8
    else:
        width = min(6 * n + 0.5, 15)
        height = 8
    
    fig, ax = plt.subplots(1, n, figsize=(width, height), layout="tight")
    if n == 1:
         ax = [ax]
    

    global_max = np.max([np.max(np.abs(E)) for E in matrix_list if E is not None])
    for idx,(E,title) in enumerate(zip(matrix_list,title_list)):  
        im = ax[idx].imshow(E,cmap='bwr',vmin=-global_max,vmax=global_max,aspect='auto')
        ax[idx].set_title(title)

        fig.colorbar(im, ax=ax[idx], fraction=0.046, pad=0.04)
    
    fig.suptitle(fig_title,fontsize=16,fontweight='bold')
    plt.show()

def plot_results(results_df,x_axis,y_axis,fixed_vars={},approaches=None,row=None,col=None,ylog=False,xlog=False, save_dir=None, title=None):
    # 1. Filter data
    df = results_df.copy()
    if approaches:
        df = df[df['Approach'].isin(approaches)]
    
    for c, val in fixed_vars.items():
        if c in df.columns:               
            df = df[df[c] == val]
    
    if ylog:
        df[y_axis] += 1e-15
    
    #a grid of plots
    sns.set_theme(style="whitegrid")
    grid = sns.FacetGrid(df, row=row, col=col, hue='Approach', 
                    height=5, aspect=1.3, palette='Set2', sharey=False, sharex=False)

    grid.map_dataframe(sns.lineplot, x=x_axis, y=y_axis, marker='o', estimator=np.mean, errorbar='se')
    if ylog:
        for ax in grid.axes.flat:
            ax.set_yscale('log')
    else:
        grid.set(ylim=(df[y_axis].min() * 0.9, df[y_axis].max() * 1.1))  # Add some padding to y-axis

    if xlog:
        for ax in grid.axes.flat:
            ax.set_xscale('log')
    elif x_axis=='k':
        for ax in grid.axes.flat:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=10,integer=True))
    grid.set_axis_labels(x_axis, y_axis)
    grid.figure.subplots_adjust(left=0.1, right=1-0.05, top=1-0.1)
    grid.add_legend(title="Approach")
    if title is not None:
        grid.set_titles(row_template="", col_template="")
        for ax in grid.axes.flat:
            ax.set_title(title)
    else:
        row_tmpl = row + " = {row_name} " if row else ""
        col_tmpl = col + " = {col_name} " if col else ""
        grid.set_titles(row_template=row_tmpl, col_template=col_tmpl)

    title_str = f""
    if row or col:
        #title_str += f" grouped by {row if row else ''} {col if col else ''}"
        pass
    grid.figure.suptitle(title_str, y=0.98)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/{y_axis}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_dir}/{y_axis}.pdf", bbox_inches='tight')
        plt.close()

    plt.show()

def plot_shared_legend(results_df, approaches=None, save_dir=None, filename="legend",
                       label_map=None, ncol=None, marker='o'):
    """
    Renders only the 'Approach' legend shared by the plot_results panels and saves
    it as a standalone, tightly-cropped PDF (``<save_dir>/<filename>.pdf``).

    The colour/order mapping is taken from an identically-constructed FacetGrid so
    the legend always matches the per-metric figures.
    """
    df = results_df.copy()
    if approaches:
        df = df[df['Approach'].isin(approaches)]

    sns.set_theme(style="whitegrid")
    grid = sns.FacetGrid(df, hue='Approach', palette='Set2')
    hue_names = list(grid.hue_names)
    colors = list(grid._colors)
    plt.close(grid.figure)

    label_map = label_map or {}
    handles = [Line2D([0], [0], color=colors[i], marker=marker, lw=1.5, markersize=6,
                      label=label_map.get(name, name))
               for i, name in enumerate(hue_names)]

    fig = plt.figure(figsize=(0.1, 0.1))
    legend = fig.legend(handles=handles, title="Approach", loc='center',
                        ncol=ncol or len(handles), frameon=True)
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(f"{save_dir}/{filename}.pdf", bbox_inches=bbox)
        plt.close(fig)
    else:
        plt.show()

def plot_probability_path(T_gt, A_low, LOSSES:dict[str,Callable], n_steps=1000,initial_T : Literal['uniform', 'compact','surjective'] = 'uniform'):
    n_plots = len(LOSSES)
    fig, ax = plt.subplots(1,n_plots,figsize=(10,6))
    E_low = np.linalg.inv(np.eye(A_low.shape[0]) - A_low)
    n_low, n_high = T_gt.shape

    if initial_T == 'uniform':
        T_init = np.ones_like(T_gt) / n_high
    elif initial_T == 'compact':
        T_init = np.zeros_like(T_gt)
        T_init[np.arange(n_low), np.argmax(T_gt, axis=1)] = 1
    elif initial_T == 'surjective':
        from scipy.special import softmax
        logits = regularised_surjective_init(n_low, n_high)
        T_init = softmax(logits, axis=-1)
    else:
        raise ValueError(f"Invalid initial_T value: {initial_T}. Expected 'uniform', 'compact', or 'surjective'.")

    t = np.linspace(0,1,n_steps)
    T_list = [t_i * T_gt + (1-t_i) * T_init for t_i in t]
    for i, loss_name in enumerate(LOSSES.keys()):
        loss_fn = LOSSES[loss_name]
        loss_values = []
        for T in T_list:
            S = get_S(T=T, E_low=E_low, A_low=A_low)
            A_high = get_A_high(T=T, A_low=A_low)
            E_high = np.linalg.inv(np.eye(n_high) - A_high)
            try:
                loss_values.append(loss_fn(A_low=A_low, E_low=E_low, T=T, S=S, A_high=A_high, E_high=E_high))
            except np.linalg.LinAlgError:
                loss_values.append(float('nan'))
        ax[i].plot(t, loss_values, label=loss_name)
        ax[i].set_xlabel('t')
        #ax[i].set_yscale('log')
        ax[i].legend()
    fig.suptitle(f'Loss along linear path from {initial_T} initialisation to Ground Truth')
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=n_plots)
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
        #model = HailMary(n_low=n_low, n_high=n_high, A_low=A_low)
        model = None
        if idx2 is None:
            losses = []
            for t1 in t1_values:
                W[idx1,0] = t1
                W[idx1,1] = 1-t1
                try:
                    balance_loss, abs_loss, graph_loss, topo_loss, surjective_loss, entropy_loss = model(T=torch.from_numpy(W).float())
                    losses.append((balance_loss.item(), abs_loss.item(), graph_loss.item(), topo_loss.item(), surjective_loss.item(), entropy_loss.item()))
                except torch._C._LinAlgError:
                    print(f"Failed to run for t ={t1}.")
                    losses.append((float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')))

            balance_losses, abs_losses, graph_losses, topo_losses, surjective_losses,entropy_losses = zip(*losses)
            plt.plot(t1_values, balance_losses, label='Balance Loss')
            plt.plot(t1_values, abs_losses, label='Abstraction Loss')
            plt.plot(t1_values, graph_losses, label='Balance Loss')
            plt.plot(t1_values, topo_losses, label='Balance Loss')
            plt.plot(t1_values, surjective_losses, label='Surjective Loss')
            plt.plot(t1_values, entropy_losses, label='Entropy Loss')
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
                    try:
                        balance_loss, abs_loss, graph_loss, topo_loss, surjective_loss, entropy_loss = model(T=torch.from_numpy(W).float())
                        row_losses.append((balance_loss.item(), abs_loss.item(), graph_loss.item(), topo_loss.item(), surjective_loss.item(), entropy_loss.item()))
                    except torch._C._LinAlgError:
                        print(f"Failed to run for t1 ={t1}, t2={t2}.")
                        row_losses.append((float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')))
                losses.append(row_losses)
            
            fig, ax = plt.subplots(1,6,figsize=(10,8))
            for i, loss_name in enumerate(['Balance Loss','Abstraction Loss','Graph Loss', 'Topo Loss','Surjective Loss','Entropy Loss']):
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

def plot_assignment_raster(history, subsample=10, T_gt=None,soft=True):
    T_list = history['T'][::subsample]  # (n_epochs, n_low, n_high)
    n_epochs = len(T_list)
    n_low, n_high = T_list[0].shape
    
    # Hard assignment at each epoch: (n_epochs, n_low)
    assignments = np.stack([T.argmax(axis=1) for T in T_list], axis=0)
    soft_assignments = np.stack(T_list, axis=0)
    cluster_0_probs = soft_assignments[:, :, 0]

    
    # If ground truth given, find best cluster permutation via Hungarian matching
    # at the final epoch and apply consistently
    if T_gt is not None:
        from scipy.optimize import linear_sum_assignment
        final_assign = assignments[-1]
        gt_assign = T_gt.argmax(axis=1)
        # Cost matrix: how many variables match between predicted cluster i and gt cluster j
        cost = np.zeros((n_high, n_high))
        for i in range(n_high):
            for j in range(n_high):
                cost[i, j] = -np.sum((final_assign == i) & (gt_assign == j))
        _, col_ind = linear_sum_assignment(cost)
        # Remap all assignments
        remap = {i: col_ind[i] for i in range(n_high)}
        assignments = np.vectorize(remap.get)(assignments)
    
    fig, axes = plt.subplots(1, 2 if T_gt is not None else 1, 
                              figsize=(14, 5), sharey=False)
    ax = axes[0] if T_gt is not None else axes

    if soft:
        # Now plot this heatmap
        im = ax.imshow(cluster_0_probs.T, aspect='auto', 
                    cmap='Blues', vmin=0, vmax=1, 
                    interpolation='nearest')
        ax.set_title('Soft Assignment Probability: Cluster 0')
        plt.colorbar(im, ax=ax, label='Probability (0 to 1)')
    else:
        im = ax.imshow(assignments.T, aspect='auto', 
                    cmap='tab20', interpolation='nearest',
                    vmin=0, vmax=n_high-1)
        ax.set_xlabel('Epoch (subsampled)')
        ax.set_ylabel('Variable index')
        ax.set_title('Predicted cluster assignment over training')
        ax.set_xticks(np.linspace(0, n_epochs-1, 6).astype(int))
        ax.set_xticklabels((np.linspace(0, len(history['T'])-1, 6) * subsample).astype(int))
        ax.set_yticks(np.arange(n_high)) # This forces the axis to only show ticks for your cluster count
        #ax.set_yticklabels([f'Node {i}' for i in range(n_high)])
        plt.colorbar(im, ax=ax, label='Cluster')
    
    if T_gt is not None:
        # Ground truth column
        gt_col = T_gt.argmax(axis=1)[:, None] * np.ones((1, n_epochs))
        axes[1].imshow(gt_col.T, aspect='auto', cmap='tab20', 
                       interpolation='nearest', vmin=0, vmax=n_high-1)
        axes[1].set_title('Ground truth assignment')
        axes[1].set_xlabel('(reference)')
    
    plt.tight_layout()
    return fig


def animate_matrices(matrix_list:list[np.ndarray], xlabel:str = '', ylabel:str = '', title: str = '', subsample=10, fps=15, save_dir='results/timelapse/',cmap='Blues'):
    """
    Creates a time-lapse animation of matrix values.
    
    Args:
        matrix_list: List of 2D matrices to display
        subsample: How many epochs to skip between frames to speed up the video
        fps: Frames per second for the playback
    """
    matrix_list = matrix_list[::subsample]
    n_frames = len(matrix_list)
    n_rows, n_cols = matrix_list[0].shape
    
    fig, ax = plt.subplots(figsize=(6, 8))
    
    cax = ax.imshow(matrix_list[0], aspect='auto', cmap=cmap, 
                    vmin=0, vmax=1, interpolation='nearest')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    title_object = ax.set_title(title + ' - Epoch: 1')
    fig.colorbar(cax, ax=ax, label='Value')
    
    ax.set_xticks(np.arange(n_cols))
    #ax.set_xticklabels([f'Cluster {i}' for i in range(n_high)])
    
    # The update function called for each frame
    def update(frame):
        # Update the data in the heatmap
        cax.set_array(matrix_list[frame])
        # Update the title with the correct epoch number
        title_object.set_text(title + f" - Epoch: {frame * subsample + 1}")
        return cax, title_object

    # Build the animation
    ani = animation.FuncAnimation(fig, update, frames=n_frames, 
                                  interval=1000/fps, blit=False)
    
    # Close the static figure so it doesn't print twice in Jupyter
    plt.close(fig)
    filename = save_dir +  f'{title}.gif'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    ani.save(filename, writer='pillow', fps=15)
    

def animate_partition_training(A_low, history, save_path='results/timelapse/partition.gif',
                               title='', as_fraction=True, max_active_frames=45,
                               tail_frames=12, hold=6, figsize=(11, 6.5)):
    """Animate T_soft over training, with the losses and the hard metrics alongside.

    Left   : T_soft, rows sorted by final cluster so the block structure is legible.
    Top-R  : the training losses as stored in `history` (already lambda-weighted by
             HailMary.train), on a log axis.
    Bot-R  : hard metrics recomputed from argmax(T_soft) at each frame. With
             `as_fraction=True` (default) all three are rates in [0, 1] on ONE axis --
             violating nodes / p, clusters used / k offered, intra-cluster edge mass -- which
             keeps them comparable across p and avoids a twin axis. The raw counts still
             appear in the header. Set `as_fraction=False` for counts on a twin axis.
    A moving line marks the current epoch on both plots.

    Frames are sampled densely up to the last cluster REASSIGNMENT and sparsely after, since
    the optimiser typically settles early (in one measured run, epoch 1015 of 6000) and the
    remaining epochs only sharpen T_soft. Each frame is repeated `hold` times because
    matplotlib's pillow writer is pinned to 15fps here, so repetition is what slows playback.

    Args:
        A_low: (p, p) low-level adjacency.
        history: dict returned by `HailMary.train(..., keep_history=True)`. Needs 'T'
            (a T_soft snapshot per epoch) and optionally the loss lists.
        save_path: where to write the .gif. Parent directories are created.
        title: prefix for the figure suptitle.
        as_fraction: plot the bottom-right metrics as rates in [0,1] on a single axis
            (default) rather than counts on a twin axis.
        max_active_frames / tail_frames: frame budget before and after the last reassignment.
        hold: repeat each frame this many times (15/hold = effective fps).

    Returns the path written.
    """
    Ts_all = history['T'] if isinstance(history, dict) else history
    if not len(Ts_all):
        raise ValueError('history contains no T snapshots -- pass keep_history=True to train()')
    A_low = np.asarray(A_low)
    Ab = (A_low != 0)
    absA = np.abs(A_low)
    tot = max(absA.sum(), 1e-12)
    p = A_low.shape[0]
    n_epochs = len(Ts_all)

    def hard_metrics(Tsoft):
        lab = Tsoft.argmax(1)
        groups = [np.where(lab == j)[0] for j in range(Tsoft.shape[1])]
        groups = [g for g in groups if len(g)]
        K = len(groups)
        M = np.zeros((K, K), bool)
        for x in range(K):
            for y in range(K):
                if x != y and Ab[np.ix_(groups[x], groups[y])].any():
                    M[x, y] = True
        bad = set()
        for x in range(K):
            for y in np.where(M[x])[0]:
                for i in groups[x]:
                    if not Ab[i, groups[y]].any():
                        bad.add(int(i))
        T = np.zeros((p, K))
        for j, g in enumerate(groups):
            T[g, j] = 1
        co = T @ T.T
        np.fill_diagonal(co, 0)
        one_child = int(M.sum(1).max() <= 1) if K else 1
        return len(bad), K, float((absA * co).sum() / tot), one_child

    assign = np.array([t.argmax(1) for t in Ts_all])
    changed = (assign[1:] != assign[:-1]).any(1)
    last = int(np.where(changed)[0].max()) + 1 if changed.any() else min(200, n_epochs - 1)
    idx = np.unique(np.concatenate([
        np.linspace(0, last, max_active_frames).astype(int),
        np.linspace(last, n_epochs - 1, tail_frames).astype(int)]))
    order = np.argsort(assign[-1])
    frames = [Ts_all[i][order] for i in idx]
    met = np.array([hard_metrics(Ts_all[i]) for i in idx])

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.5], hspace=0.32, wspace=0.25)
    axT, axL, axM = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])

    im = axT.imshow(frames[0], aspect='auto', cmap='Blues', vmin=0, vmax=1,
                    interpolation='nearest')
    axT.set_xlabel('cluster')
    axT.set_ylabel('low-level node (sorted by final cluster)')
    fig.colorbar(im, ax=axT, fraction=0.046, label='T_soft')

    E = np.arange(n_epochs)
    for key, lab, c in [('graph_loss', 'graph (All-In)', '#d62728'),
                        ('one_child_loss', 'one_child', '#1f77b4'),
                        ('info_loss', 'info', '#2ca02c'),
                        ('recon_loss', 'recon', '#9467bd')]:
        v = history.get(key) if isinstance(history, dict) else None
        if v is None or not len(v) or not np.any(np.asarray(v) > 0):
            continue
        axL.plot(E[:len(v)], np.maximum(np.asarray(v, float), 1e-8), lw=1.2, color=c, label=lab)
    axL.set_yscale('log'); axL.set_xlim(0, n_epochs)
    axL.set_ylabel('weighted loss'); axL.legend(fontsize=7, loc='upper right', framealpha=.9)
    axL.set_title('training losses (as weighted in the objective)', fontsize=9)
    vl1 = axL.axvline(0, color='k', lw=1.1)

    k_offered = Ts_all[0].shape[1]
    if as_fraction:
        plot_v = met[:, 0] / p                  # share of variables that would be marginalised
        plot_k = met[:, 1] / max(k_offered, 1)  # share of the cluster budget in use
        lab_v = 'All-In violating nodes / p'
        lab_k = f'clusters used / {k_offered} offered'
        axM2 = axM
    else:
        plot_v, plot_k = met[:, 0], met[:, 1]
        lab_v, lab_k = 'All-In violating nodes', 'clusters used'
        axM2 = axM.twinx()
        axM2.set_ylabel('intra-cluster mass', color='#2ca02c'); axM2.grid(False)
    axM.plot(idx, plot_v, color='#d62728', lw=1.4, label=lab_v)
    axM.plot(idx, plot_k, color='#7f7f7f', lw=1.4, label=lab_k)
    axM.set_xlim(0, n_epochs); axM.set_xlabel('epoch')
    axM.set_ylabel('fraction' if as_fraction else 'count')
    if as_fraction:
        axM.set_ylim(0, 1.03)
    axM2.plot(idx, met[:, 2], color='#2ca02c', lw=1.4, ls='--', label='intra-cluster mass')
    axM2.set_ylim(0, 1.03)
    # with a twin axis the handles live on two axes, so merge both sets into one legend
    handles = axM.get_legend_handles_labels()[0]
    labels = axM.get_legend_handles_labels()[1]
    if axM2 is not axM:
        handles = handles + axM2.get_legend_handles_labels()[0]
        labels = labels + axM2.get_legend_handles_labels()[1]
    axM.legend(handles, labels, fontsize=7, loc='center right', framealpha=.9)
    axM.set_title('hard metrics from argmax(T_soft)', fontsize=9)
    vl2 = axM.axvline(0, color='k', lw=1.1)
    d1 = axM.plot([idx[0]], [plot_v[0]], 'o', color='#d62728', ms=6)[0]
    d2 = axM.plot([idx[0]], [plot_k[0]], 'o', color='#7f7f7f', ms=6)[0]
    d3 = axM2.plot([idx[0]], [met[0, 2]], 'o', color='#2ca02c', ms=6)[0]
    sup = fig.suptitle('', fontsize=10)

    def update(f):
        i = f // hold
        im.set_array(frames[i])
        e = idx[i]
        vl1.set_xdata([e, e]); vl2.set_xdata([e, e])
        d1.set_data([e], [plot_v[i]]); d2.set_data([e], [plot_k[i]]); d3.set_data([e], [met[i, 2]])
        sup.set_text(f'{title}  epoch {e:5d}   |   violating nodes {int(met[i,0]):3d}   '
                     f'clusters {int(met[i,1]):2d}   intra-mass {met[i,2]:.3f}   '
                     f'one-child {"yes" if met[i,3] else "no"}')
        return im, vl1, vl2, d1, d2, d3, sup

    ani = animation.FuncAnimation(fig, update, frames=len(frames) * hold, blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    ani.save(save_path, writer='pillow', fps=15)
    plt.close(fig)
    return save_path

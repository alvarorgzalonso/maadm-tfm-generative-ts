import numpy as np
from matplotlib import pyplot as plt
import json
import os


## MSM distance

def C(xi, x_prev, yj, c):
    if x_prev <= xi <= yj or x_prev >= xi >= yj:
        return c  # move is the best option
    else:
        return c + min(abs(xi - x_prev), abs(xi - yj))

def MSM_Distance(X, Y, c=0.1, verbose=False, return_cost=False):
    """
    Calculate the Move-Split-Merge (MSM) distance between two time series.
    Parameters:
    X (np.array): The first time series.
    Y (np.array): The second time series.
    c (float): The cost of the operations (move). Default is 0.1.
    return__cost (bool): If True, return the cost matrix. Default is False.
    verbose (bool): If True, log the operations. Default is False.
    Returns:
    tuple: The MSM distance between the two time series.
    """
    m = len(X)
    n = len(Y)
    Cost = np.zeros((m, n))

    # Initialization
    Cost[0, 0] = abs(X[0] - Y[0])
    if verbose:
        print(f"Init: Cost[0,0] = |{X[0]} - {Y[0]}| = {Cost[0, 0]}")

    for i in range(1, m):
        Cost[i, 0] = Cost[i - 1, 0] + C(X[i], X[i - 1], Y[0], c)
        if verbose:
            print((f"Init: Cost[{i},0] = Cost[{i - 1},0] + C({X[i]},"
                   f"{X[i - 1]}, {Y[0]}, {c}) = {Cost[i, 0]}"))

    for j in range(1, n):
        Cost[0, j] = Cost[0, j - 1] + C(Y[j], Y[j - 1], X[0], c)
        if verbose:
            print((f"Init: Cost[0,{j}] = Cost[0,{j - 1}] + C({Y[j]},"
                   f"{Y[j - 1]}, {X[0]}, {c}) = {Cost[0, j]}"))

    # Main Loop
    for i in range(1, m):
        for j in range(1, n):
            # diagonal movement in the cost matrix. This operation considers
            # the cost of matching the ith element in the time series X with
            # the jth element in Y.
            move_cost = Cost[i - 1, j - 1] + abs(X[i] - Y[j])

            # vertical movement in the cost matrix. This operation considers
            # the cost of potentially merging the ith element in the time
            # series X with the (i−1)th element before matching it with the
            # jth element in Y.
            merge_cost = Cost[i - 1, j] + C(X[i], X[i - 1], Y[j], c)

            # horizontal movement in the cost matrix. This operation accounts
            # for the cost of potentially splitting the jth element in the
            # time series Y to align with the ith element in X.
            split_cost = Cost[i, j - 1] + C(Y[j], X[i], Y[j - 1], c)

            Cost[i,j] = min(move_cost, merge_cost, split_cost)

            if verbose:
                print(
                    (f"Cost[{i},{j}]:  min(Move: "
                     f"{move_cost}, Merge: {merge_cost}, Split: "
                     f"{split_cost}) = {Cost[i, j]}")
                )

    if return_cost:
        return Cost[-1, -1], Cost
    else:
        return Cost[-1, -1]

def plot_cost_matrix(cost_matrix, alignment, X, Y):
    n, m = cost_matrix.shape
    
    # Create a color map for the cost matrix
    cmap = plt.cm.get_cmap('viridis')
    
    # Plot the cost matrix
    fig, ax = plt.subplots(figsize=(30, 20))
    im = ax.imshow(cost_matrix, cmap=cmap, aspect='auto')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_label('Cost')
    
    # Add labels and title
    ax.set_xticks(np.arange(m))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(Y)
    ax.set_yticklabels(X)
    ax.set_xlabel('Time Series Y')
    ax.set_ylabel('Time Series X')
    ax.set_title('Cost Matrix with Best Alignment')
    
    # Highlight the best alignment path
    for i, j in alignment:
        ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='red', lw=2))
    
    # Add cost matrix values to each cell
    for i in range(n):
        for j in range(m):
            ax.text(j, i, round(cost_matrix[i, j], 2), ha="center", va="center", color="w")
    
    plt.tight_layout()
    plt.show()

def compute_path(cost_matrix):
    n, m = cost_matrix.shape
    alignment = []
    i, j = n - 1, m - 1
    
    while i > 0 or j > 0:
        alignment.append((i, j))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            min_index = np.argmin((cost_matrix[i-1, j-1], cost_matrix[i-1, j], cost_matrix[i, j-1]))
            if min_index == 0:
                i -= 1
                j -= 1
            elif min_index == 1:
                i -= 1
            else:
                j -= 1
    
    alignment.append((0, 0))
    return list(reversed(alignment))

def msm_barycenter_average(X, max_iters=30, c=1.0, tol=1e-5, verbose=False):
    center = msm_medoids(X, c)
    prev_cost = np.inf
    
    for i in range(max_iters):
        center, cost = msm_ba_update(center, X, c, verbose)
        if verbose:
            print(f"Iteration {i+1}: Cost = {cost}")
        
        if abs(prev_cost - cost) < tol:
            if verbose:
                print(f"Convergence reached at iteration {i+1}")
            break
        
        prev_cost = cost
    
    return center

def msm_ba_update(center, X, c, verbose=False):
    n, m = X.shape[0], center.shape[0]
    num_warps = np.zeros(m)
    alignment = np.zeros(m)
    total_cost = 0
    
    for i in range(n):
        dist, cost_matrix = MSM_Distance(X[i], center, c, return_cost=True)
        curr_alignment = compute_path(cost_matrix)
        for j, k in curr_alignment:
            alignment[k] += X[i, j]
            num_warps[k] += 1
        total_cost += dist
    
    new_center = alignment / num_warps
    avg_cost = total_cost / n
    
    if verbose:
        print(f"Average cost: {avg_cost}")
    
    return new_center, avg_cost

def msm_medoids(X, c):
    n = X.shape[0]
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = MSM_Distance(X[i], X[j], c)
    
    medoid_index = np.argmin(np.sum(dist_matrix, axis=1))
    return X[medoid_index]

def plot_cluster_with_barycenter(X, barycenter):
    plt.figure(figsize=(10, 6))
    
    # Plot the individual time series in the cluster
    for ts in X:
        plt.plot(ts, color='gray', alpha=0.3, linewidth=1)
    
    # Plot the barycenter time series with a thicker line and higher opacity
    plt.plot(barycenter, color='red', alpha=1.0, linewidth=2, label='Barycenter')
    #plot mean
    plt.plot(np.mean(X, axis=0), color='blue', alpha=1.0, linewidth=2, label='Mean')
    
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Cluster with Barycenter')
    plt.legend()
    plt.grid(True)
    plt.show()
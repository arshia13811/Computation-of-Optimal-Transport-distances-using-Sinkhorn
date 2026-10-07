import numpy as np

def compute_sinkhorn_distances(r, C, cost_matrix_M, lmbda, max_iter=20):
    """
    Computes the Sinkhorn distances between a test image 'r' and a matrix of training images 'C'.
    Now includes the relative distance stopping condition.
    """
    K = np.exp(-lmbda * cost_matrix_M)
    KM = K * cost_matrix_M
    N = C.shape[1] 

    # u initialization
    u = np.ones((400, N)) / 400.0 
    r_col = r.reshape(-1, 1)
    
    prev_dist = None

    for _ in range(max_iter):
        # 1. Update v
        K_trans_u = np.dot(K.T, u)
        V = C / (K_trans_u + 1e-15)  # Epsilon added to prevent division by zero
        
        # 2. Update u
        K_v = np.dot(K, V)
        u = r_col / (K_v + 1e-15)
        
        # 3. Compute current transport costs for this iteration
        KMV = np.dot(KM, V)
        current_dist = np.sum(u * KMV, axis=0)
        
        # 4. The relative stopping condition
        if prev_dist is not None:
            # max | (d_k / d_k-1) - 1 | < 10^-4
            max_rel_change = np.max(np.abs((current_dist / (prev_dist + 1e-15)) - 1.0))
            if max_rel_change < 1e-4:
                break  # The distances have settled, stop early
                
        # Store current distances for the next iteration's comparison
        prev_dist = current_dist

    return current_dist

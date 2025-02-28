import torch
import numpy as np

def compute_sigmas(distances, n_neighbors, device):
    """Compute the bandwidth parameter sigma for each point"""
    target = np.log2(n_neighbors)
    
    lo = torch.zeros(distances.shape[0], device=device)
    hi = torch.ones(distances.shape[0], device=device) * 100.0
    mid = (hi + lo) / 2.0
    
    # Normalize distances for numerical stability
    max_distances = distances.max(dim=1, keepdim=True)[0]
    norm_distances = distances / max_distances
    
    for i in range(10):
        rho = distances[:, 0]
        exp_neg_d = torch.exp(-(norm_distances - rho.unsqueeze(1)) / mid.unsqueeze(1))
        sum_exp = exp_neg_d.sum(dim=1)
        
        too_low = sum_exp > target
        too_high = sum_exp < target
        
        hi[too_low] = mid[too_low]
        lo[too_high] = mid[too_high]
        mid = (hi + lo) / 2.0
    
    return mid * max_distances.squeeze()
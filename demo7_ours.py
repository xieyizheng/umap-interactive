# This script extends demo6 by adding continuous graph recalculation option
# Users can choose between manual recalculation or continuous updates

import torch
from torch import nn
import pandas as pd
import polyscope as ps
import polyscope.imgui as psim
from umap.umap_ import find_ab_params
from utils.utils_device import check_device_availability
from utils.utils_ui import UMAPUIController
from utils.utils_fuzzy_simplicial_set import compute_sigmas

# Configuration parameters
N_POINTS = 2000
MIN_DIST = 0.1
EPOCHS = 100000
LEARNING_RATE = 1e-1
N_COMPONENTS = 2
INITIAL_N_NEIGHBORS = 300

class ExtendedUMAPUIController(UMAPUIController):
    def __init__(self, optimizer, min_dist, learning_rate, n_neighbors):
        super().__init__(optimizer, min_dist, learning_rate)
        self.ui_n_neighbors = n_neighbors
        self.recalculate_requested = False
        self.continuous_update = False
        
    def update_callback(self):
        # Call parent class callback first
        super().update_callback()
        
        # Add Graph Control section
        psim.TextUnformatted("Graph Parameters")
        psim.Separator()
        
        # Add n_neighbors slider
        psim.PushItemWidth(200)
        changed, new_n_neighbors = psim.SliderInt("n_neighbors", self.ui_n_neighbors, v_min=5, v_max=500)
        if changed:
            self.ui_n_neighbors = new_n_neighbors
            if self.continuous_update:  # Trigger recalculation if continuous update is enabled
                self.recalculate_requested = True
        
        # Add continuous update toggle
        _, self.continuous_update = psim.Checkbox("Continuous Graph Update", self.continuous_update)
        if self.continuous_update:
            psim.TextColored((1, 1, 0, 1), "Warning: Continuous updates may impact performance")
        
        # Add manual recalculate button (disabled if continuous update is on)
        if not self.continuous_update:
            if psim.Button("Recalculate Graph"):
                self.recalculate_requested = True
                print(f"Graph recalculation requested with n_neighbors={self.ui_n_neighbors}")
        
        psim.PopItemWidth()

def calculate_graph(X_torch, n_neighbors, device):
    """Calculate the UMAP graph with given parameters"""
    print(f"Calculating graph with n_neighbors={n_neighbors}...")
    start_time = time.time()
    
    # Compute distances and get kNN
    distances = torch.cdist(X_torch, X_torch)
    values, indices = torch.topk(distances, k=n_neighbors, dim=1, largest=False)

    # Compute sigmas
    sigmas = compute_sigmas(values, n_neighbors, device)

    # Compute membership strengths
    n_samples = X_torch.shape[0]
    rows = torch.arange(n_samples, dtype=torch.int64, device=device).repeat_interleave(n_neighbors)
    cols = indices.flatten()

    # Calculate weights using the UMAP formula
    rho = values[:, 0].repeat_interleave(n_neighbors)
    dist_expanded = values.flatten()
    sigma_expanded = sigmas.repeat_interleave(n_neighbors)
    weights = torch.exp(-(dist_expanded - rho) / sigma_expanded).to(torch.float32)

    # Create sparse graph
    graph_torch = torch.zeros((n_samples, n_samples), device=device)
    graph_torch[rows, cols] = weights

    # Make the graph symmetric
    graph_torch = torch.maximum(graph_torch, graph_torch.T)
    
    end_time = time.time()
    print(f"Graph calculation completed in {end_time - start_time:.2f} seconds")
    
    return graph_torch

# Load and prepare data
import time
X = pd.read_csv('mammoth_a.csv').sample(n=N_POINTS, random_state=42)[['x', 'y', 'z']].values

# Get device and convert X to tensor
device = check_device_availability()
X_torch = torch.tensor(X, dtype=torch.float32).to(device)

# Calculate initial graph
graph_torch = calculate_graph(X_torch, INITIAL_N_NEIGHBORS, device)

# Initialize random embedding
embedding = nn.Parameter(torch.randn(N_POINTS, N_COMPONENTS, device=device)*5)
optimizer = torch.optim.AdamW([embedding], lr=LEARNING_RATE)

# Setup visualization
ps.init()
ps.set_ground_plane_mode("none")
ui_controller = ExtendedUMAPUIController(optimizer, MIN_DIST, LEARNING_RATE, INITIAL_N_NEIGHBORS)
ps.set_user_callback(ui_controller.update_callback)
cloud = ps.register_point_cloud("umap_embedding", embedding.detach().cpu().numpy(), 
                              enabled=True, color=[0, 0, 0], 
                              transparency=0.5, radius=0.001)

# Training loop
last_update_time = time.time()
MIN_UPDATE_INTERVAL = 1.0  # Minimum time (in seconds) between graph updates

for epoch in range(EPOCHS):
    if not ui_controller.paused:
        current_time = time.time()
        
        # Check if graph recalculation is needed
        should_recalculate = (
            ui_controller.recalculate_requested or
            (ui_controller.continuous_update and 
             current_time - last_update_time >= MIN_UPDATE_INTERVAL)
        )
        
        if should_recalculate:
            graph_torch = calculate_graph(X_torch, ui_controller.ui_n_neighbors, device)
            ui_controller.recalculate_requested = False
            last_update_time = current_time
            print("Graph recalculation completed")
        
        # Update parameters based on current min_dist
        a, b = find_ab_params(1.0, ui_controller.ui_min_dist)
        
        # Compute distances and probabilities
        distance = torch.cdist(embedding, embedding, p=2)
        probability = -torch.log1p(a * distance ** (2 * b))
        log_prob = torch.nn.functional.logsigmoid(probability)
        
        # Calculate loss terms
        attractive_term = -graph_torch * log_prob
        repulsive_term = -(1.0 - graph_torch) * (log_prob - probability)
        
        # Combine terms with weight
        loss = ui_controller.ui_attractive_weight * attractive_term + repulsive_term
        epoch_loss = torch.mean(loss)

        # Optimization step
        optimizer.zero_grad()
        epoch_loss.backward()
        optimizer.step()
        
        # Update UI and logging
        ui_controller.set_epoch_loss(epoch_loss.item())
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}")
    
    # Update visualization
    cloud.update_point_positions(embedding.detach().cpu().numpy())
    ps.frame_tick()

ps.show() 
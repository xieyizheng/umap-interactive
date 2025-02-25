import torch
from torch import nn
import pandas as pd
from umap import UMAP
from umap.umap_ import find_ab_params
import polyscope as ps
from utils.utils_device import check_device_availability
from utils.utils_ui import UMAPUIController

# Configuration parameters
N_POINTS = 2000
MIN_DIST = 0.1
EPOCHS = 100000
LEARNING_RATE = 1e-1
N_COMPONENTS = 3

# Load and prepare data
X = pd.read_csv('mammoth_a.csv').sample(n=N_POINTS)[['x', 'y', 'z']].values

# Get initial embedding and graph from official UMAP
umap = UMAP(
    random_state=42,
    verbose=True,
    n_neighbors=300,
    n_components=N_COMPONENTS,
    n_epochs=0,
    metric='euclidean'
)
init_embedding = umap.fit_transform(X)
graph = umap.graph_

# Setup PyTorch optimization
device = check_device_availability()
graph_torch = torch.tensor(graph.todense(), dtype=torch.float32).to(device)
embedding = nn.Parameter(torch.from_numpy(init_embedding).to(device, dtype=torch.float32))
optimizer = torch.optim.AdamW([embedding], lr=LEARNING_RATE)

# Setup visualization
ps.init()
ps.set_ground_plane_mode("none")
ui_controller = UMAPUIController(optimizer, MIN_DIST, LEARNING_RATE)
ps.set_user_callback(ui_controller.update_callback)
cloud = ps.register_point_cloud("umap_embedding", init_embedding, enabled=True, 
                              color=[0, 0, 0], transparency=0.5, radius=0.001)

# Training loop
for epoch in range(EPOCHS):
    if not ui_controller.paused:
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
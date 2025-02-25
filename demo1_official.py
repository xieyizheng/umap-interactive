# Load mammoth dataset
import pandas as pd
from umap import UMAP
import polyscope as ps
N_POINTS = 2000
# Load and prepare data
X = pd.read_csv('mammoth_a.csv').sample(n=N_POINTS)[['x', 'y', 'z']].values

umap = UMAP(
    random_state=42,
    verbose=True,
    n_neighbors=300,
    n_components=2,
    metric='euclidean'
)
init_embedding = umap.fit_transform(X)

ps.init()
ps.set_ground_plane_mode("none")
cloud = ps.register_point_cloud("umap_embedding", init_embedding, enabled=True, color=[0, 0, 0], transparency=0.5, radius=0.001)
ps.show()





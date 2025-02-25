import polyscope.imgui as psim
import numpy as np

class UMAPUIController:
    def __init__(self, optimizer, min_dist=0.1, learning_rate=1e-1):
        self.ui_min_dist = min_dist
        self.ui_learning_rate = np.log10(learning_rate)
        self.ui_attractive_weight = 50.0
        self.paused = False
        self.optimizer = optimizer
        self.epoch_loss = None

    def update_callback(self):
        psim.PushItemWidth(200)
        
        # Add sliders for parameters
        changed_dist, self.ui_min_dist = psim.SliderFloat("Min Distance", self.ui_min_dist, v_min=0.0, v_max=1.0)
        
        # Learning rate slider (logarithmic scale)
        changed_lr, self.ui_learning_rate = psim.SliderFloat("Log10 Learning Rate", self.ui_learning_rate, v_min=-4.0, v_max=2.0)
        if changed_lr:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 10**self.ui_learning_rate
        
        # Add attractive weight slider
        _, self.ui_attractive_weight = psim.SliderFloat("Attractive Term Weight", self.ui_attractive_weight, v_min=0.0, v_max=500.0)
        
        # Add pause button
        if psim.Button("Pause/Resume"):
            self.paused = not self.paused
        
        # Display current parameters
        psim.TextUnformatted(f"Current Loss: {self.epoch_loss:.4f}" if self.epoch_loss is not None else "Loss: N/A")
        psim.TextUnformatted(f"Current LR: {10**self.ui_learning_rate:.2e}")
        psim.TextUnformatted(f"Attractive Weight: {self.ui_attractive_weight:.2f}")
        
        psim.PopItemWidth()

    def set_epoch_loss(self, loss):
        self.epoch_loss = loss 
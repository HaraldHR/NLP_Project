import matplotlib.pyplot as plt
import torch

class LossPlotter:
    @staticmethod
    def plot_losses(training_loss, validation_loss, timesteps, title="Training vs Validation Loss"):
        # Convert to lists if any input is a tensor
        if isinstance(training_loss, torch.Tensor):
            training_loss = training_loss.tolist()
        if isinstance(validation_loss, torch.Tensor):
            validation_loss = validation_loss.tolist()
        if isinstance(timesteps, torch.Tensor):
            timesteps = timesteps.tolist()

        # Sanity check
        if not (len(training_loss) == len(validation_loss) == len(timesteps)):
            raise ValueError("Lengths of training_loss, validation_loss, and timesteps must be equal.")

        # Plotting
        plt.figure(figsize=(8, 5))
        plt.plot(timesteps, training_loss, label="Training Loss", marker='o')
        plt.plot(timesteps, validation_loss, label="Validation Loss", marker='x')
        plt.xlabel("Timesteps")
        plt.ylabel("Loss")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

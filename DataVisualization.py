import matplotlib.pyplot as plt

class LossPlotter:
    @staticmethod
    def plot_losses(training_loss, validation_loss, epochs, title="Training vs Validation Loss"):
        # Sanity check
        if not (len(training_loss) == len(validation_loss) == len(epochs)):
            raise ValueError("Lengths of training_loss, validation_loss, and epochs must be equal.")

        # Plotting
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, training_loss, label="Training Loss")
        plt.plot(epochs, validation_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

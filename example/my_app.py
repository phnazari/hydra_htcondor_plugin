"""Simple example demonstrating the HTCondor launcher with Hydra."""
import time

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path=".", config_name="config")
def my_app(cfg: DictConfig) -> float:
    """A simple task that simulates some work and returns a result."""
    print(f"Running job with: learning_rate={cfg.learning_rate}, batch_size={cfg.batch_size}")
    print(f"Training for {cfg.epochs} epochs...")

    # Simulate some work
    time.sleep(2)

    # Return a fake "loss" based on hyperparameters
    loss = 1.0 / (cfg.learning_rate * cfg.batch_size)
    print(f"Final loss: {loss:.4f}")

    return loss


if __name__ == "__main__":
    my_app()

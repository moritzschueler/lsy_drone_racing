"""Hyperparameter Random Search for Multi-Agent Drone Racing (IPPO)."""

import random
import time
from pathlib import Path
import fire

from lsy_drone_racing.rl.tasks import get_task
from lsy_drone_racing.utils import load_config

CHECKPOINT_DIR = Path(__file__).parents[3] / "checkpoints"

def sample_hyperparameters() -> dict:
    """Sample a random set of hyperparameters for multi-agent racing."""
    coefficients = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    return {
        # Competition Reward Scaling
        "competition_rank_coef": random.choice(coefficients),
        "competition_segment_lead_coef": random.choice(coefficients),
        "competition_proximity_coef": random.choice(coefficients),
        "competition_victory_coef": random.choice([20.0, 50.0, 100.0]),
        "competition_downwash_coef": random.choice(coefficients),
        "init_checkpoint": "lsy_drone_racing/rl/checkpoints/single_agent_racing/single_agent_racing_20260710-131954_g3.88_best.ckpt"
    }

def main(
    num_trials: int = 20,
    task: str = "multi_agent_racing",
    config: str = "rl_multi_level2.toml",
    wandb_enabled: bool = True,
):
    """Run random search over multi-agent hyperparameters."""
    task_spec = get_task(task)
    
    checkpoint_dir = CHECKPOINT_DIR / task
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Resolve configuration path matching train.py logic
    env_config = load_config(Path(__file__).parents[3] / "config" / config)

    print(f"Starting Random Search for task '{task}' ({num_trials} trials)...")

    for trial in range(num_trials):
        print(f"\n=== Trial {trial + 1}/{num_trials} ===")
        sampled_kwargs = sample_hyperparameters()
        
        print("Sampled Hyperparameters:")
        for k, v in sampled_kwargs.items():
            print(f"  {k}: {v}")
            
        # Create unique run names for tracking
        run_name = f"rs_{task}_trial_{trial + 1}_{int(time.time())}"
        
        # Build Args class instance with randomly sampled overrides
        args = task_spec.args_cls.create(**sampled_kwargs)
        
        try:
            print(f"Launching training run: {run_name}")
            task_spec.train_fn(
                args, 
                task_spec.make_env, 
                env_config, 
                checkpoint_dir, 
                run_name, 
                wandb_enabled
            )
        except Exception as e:
            print(f"Trial {trial + 1} failed with exception: {e}")
            continue

if __name__ == "__main__":
    fire.Fire(main)
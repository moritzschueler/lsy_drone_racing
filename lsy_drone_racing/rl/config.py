"""Training configuration shared across RL tasks."""

from dataclasses import dataclass
from typing import Any


@dataclass
class Args:
    """Class to store configurations."""

    seed: int = 42
    """seed of the experiment"""
    jax_device: str = "cpu"
    """environment and training device"""
    wandb_project_name: str = "maadr"
    """the wandb's project name"""
    wandb_entity: str = "maad-flies"
    """the entity (team) of wandb's project"""

    # Algorithm specific arguments
    total_timesteps: int = 1_500_000
    """total timesteps of the experiments"""
    learning_rate: float = 1.5e-3
    """the learning rate of the optimizer"""
    num_envs: int = 1024
    """the number of parallel game environments"""
    num_steps: int = 8
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.94
    """the discount factor gamma"""
    gae_lambda: float = 0.97
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.26
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.007
    """coefficient of the entropy (initial value when anneal_ent_coef is set)"""
    anneal_ent_coef: bool = False
    """linearly anneal ent_coef from its initial value to 0 over training (like anneal_lr), so the policy explores early and sharpens late instead of growing more stochastic"""
    vf_coef: float = 0.7
    """coefficient of the value function"""
    max_grad_norm: float = 1.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # Graceful early stop of the whole run (saves the best checkpoint, then evaluation/render
    # runs on it). Disabled by default; enabled per-task via the task config.
    early_stop_patience: int = 0
    """iterations with no improvement in the best mean true-start return before stopping the run (0 = disabled). Acts as the floor when early_stop_patience_frac > 0."""
    kl_floor: float = 0.0
    """stalled-policy kill-switch: approx_kl below this for kl_floor_patience consecutive iterations stops the run (0 = disabled)"""
    kl_floor_patience: int = 20
    """consecutive iterations of approx_kl < kl_floor required to trip the stalled-policy kill-switch. Acts as the floor when kl_floor_patience_frac > 0."""
    # Horizon-relative early-stop windows: the absolute patiences above are fixed iteration counts,
    # so a shorter total_timesteps makes them a larger fraction of the run (more trigger-happy). When
    # a *_frac is set, Args.create scales the patience up to that fraction of num_iterations (keeping
    # the absolute value as a floor), so the stop behavior is consistent across training horizons.
    early_stop_patience_frac: float = 0.0
    """no-improvement patience as a fraction of num_iterations (0 = use the absolute early_stop_patience)"""
    kl_floor_patience_frac: float = 0.0
    """stalled-KL patience as a fraction of num_iterations (0 = use the absolute kl_floor_patience)"""
    early_stop_arm_frac: float = 0.0
    """training progress (global_step / total_timesteps) below which both early-stop kill-switches are inert, so the run cannot be cut off while the curriculum is still ramping (0 = armed from the start)"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    # Wrapper settings (observation history + action/angle reward shaping)
    n_obs: int = 2
    rpy_coef: float = 0.06
    d_act_th_coef: float = 0.4
    d_act_xy_coef: float = 1.0
    act_coef: float = 0.02
    """reward coefficients applied by the wrappers during training"""

    # Env (in-step) racing reward coefficients. Only used by the racing task; other tasks
    # compute their reward inside their env class and ignore these.
    progress_coef: float = 1.0
    gate_bonus: float = 2.0
    finish_bonus: float = 10.0
    crash_penalty: float = 5.0
    timeout_penalty: float = 5.0
    """dense racing reward coefficients (computed inside the env step)"""

    @staticmethod
    def create(**kwargs: Any) -> "Args":
        """Create arguments class."""
        args = Args(**kwargs)
        args.batch_size = int(args.num_envs * args.num_steps)
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
        args.num_iterations = args.total_timesteps // args.batch_size
        # Scale the early-stop windows to the horizon: a *_frac sets the patience to that fraction of
        # num_iterations, keeping the absolute value as a floor (so short runs aren't cut off below it
        # and long runs get proportionally more patience).
        if args.early_stop_patience_frac > 0:
            args.early_stop_patience = max(
                args.early_stop_patience, round(args.early_stop_patience_frac * args.num_iterations)
            )
        if args.kl_floor_patience_frac > 0:
            args.kl_floor_patience = max(
                args.kl_floor_patience, round(args.kl_floor_patience_frac * args.num_iterations)
            )
        return args

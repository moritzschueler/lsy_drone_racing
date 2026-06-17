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

    # Filled during runtime
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
    progress_reach: float = 2.0
    """length scale (m) of the progress potential's far field: how far from the gate the term still
    pulls. Sized to the largest gate-to-gate gap so there is no flat dead zone between gates."""
    progress_sharpness: float = 0.3
    """length scale (m) over which the progress potential's near-gate funnel acts; smaller = the
    through-gate funnel is concentrated closer to the opening."""
    exit_scale: float = 3.0
    """entry/exit asymmetry of the progress potential (>= 1). The gate-local along coordinate is
    inflated by this factor on the exit (already-passed) side, so the field decays faster behind the
    gate -- a through-gate funnel that pulls the drone in from the entry side. 1 = symmetric."""
    speed_coef: float = 0.0
    """overall weight of the exponential speed-barrier penalty; 0 disables it."""
    max_speed: float = 3.0
    """speed ceiling (m/s). Soft barrier: the penalty grows exponentially toward this and diverges at
    it (saturated to a finite cap), so the drone effectively cannot exceed max_speed."""
    speed_penalty_slope: float = 0.3
    """slope of the exponential speed barrier: larger = the wall rises earlier/steeper (firmer, lower
    effective ceiling), smaller = the drone can get closer to max_speed before the penalty bites."""

    @staticmethod
    def create(**kwargs: Any) -> "Args":
        """Create arguments class."""
        args = Args(**kwargs)
        args.batch_size = int(args.num_envs * args.num_steps)
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
        args.num_iterations = args.total_timesteps // args.batch_size
        return args

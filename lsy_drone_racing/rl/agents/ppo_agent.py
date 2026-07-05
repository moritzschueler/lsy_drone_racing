"""Actor-critic network and Gaussian-policy helpers for PPO."""

import jax.numpy as jnp
from flax import nnx
from jax import Array

class Agent(nnx.Module):
    """RL Agent implemented as a stateful Flax NNX module.

    Like a PyTorch ``nn.Module``, NNX modules own their weights directly: layers are
    created in ``__init__`` and the parameters live on the instance. Training uses the
    stateful NNX style end to end -- ``nnx.Optimizer`` holds the optimizer state and
    mutates the model in place via ``optimizer.update(model, grads)``, and the jitted
    train/inference functions take the model as an argument (``@nnx.jit``).

    The forward pass returns (action_mean, log_std, value). Separate helper functions
    handle action sampling and log-probability computation.
    """

    def __init__(self, obs_dim: int, action_dim: int, *, rngs: nnx.Rngs):
        """Build the actor and critic heads.

        Args:
            obs_dim: Flattened observation dimension (network input size).
            action_dim: Action dimension (network output size).
            rngs: NNX random state used to initialize the layer weights.
        """
        orth = nnx.initializers.orthogonal
        zeros = nnx.initializers.zeros

        # Critic
        self.critic1 = nnx.Linear(
            obs_dim, 256, kernel_init=orth(jnp.sqrt(2)), bias_init=zeros, rngs=rngs
        )
        self.critic2 = nnx.Linear(256, 128, kernel_init=orth(jnp.sqrt(2)), bias_init=zeros, rngs=rngs)
        self.critic_out = nnx.Linear(128, 1, kernel_init=orth(1.0), bias_init=zeros, rngs=rngs)

        # Actor
        self.actor1 = nnx.Linear(
            obs_dim, 256, kernel_init=orth(jnp.sqrt(2)), bias_init=zeros, rngs=rngs
        )
        self.actor2 = nnx.Linear(256, 128, kernel_init=orth(jnp.sqrt(2)), bias_init=zeros, rngs=rngs)
        self.actor_mean = nnx.Linear(
            128, action_dim, kernel_init=orth(0.01), bias_init=zeros, rngs=rngs
        )

        # Learnable log std: lower for roll/pitch/yaw, higher for thrust (last action dim)
        self.log_std = nnx.Param(jnp.full((1, action_dim), -1.0).at[0, -1].set(1.0))

    def __call__(self, x: Array) -> tuple[Array, Array, Array]:
        """Forward pass.

        Args:
            x: Observation tensor of shape (batch, obs_dim).

        Returns:
            Tuple of (action_mean, log_std, value) with shapes
            (batch, action_dim), (1, action_dim), (batch, 1).
        """
        # Critic
        v = nnx.leaky_relu(self.critic1(x), 0.2)
        v = nnx.leaky_relu(self.critic2(v), 0.2)
        value = self.critic_out(v)

        # Actor
        m = nnx.leaky_relu(self.actor1(x), 0.2)
        m = nnx.leaky_relu(self.actor2(m), 0.2)
        mean = jnp.tanh(self.actor_mean(m))

        return mean, self.log_std[...], value


def _log_prob(action: Array, mean: Array, log_std: Array) -> Array:
    """Compute log probability of actions under a diagonal Gaussian."""
    std = jnp.exp(log_std)
    return jnp.sum(
        -0.5 * ((action - mean) / std) ** 2 - log_std - 0.5 * jnp.log(2.0 * jnp.pi),
        axis=-1,
    )


def _entropy(log_std: Array, batch_size: int) -> Array:
    """Compute entropy of a diagonal Gaussian, broadcast to batch."""
    per_dim = 0.5 * jnp.log(2.0 * jnp.pi * jnp.e) + log_std  # (1, action_dim)
    return jnp.broadcast_to(jnp.sum(per_dim, axis=-1), (batch_size,))

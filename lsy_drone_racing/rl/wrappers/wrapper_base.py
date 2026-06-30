from flax import struct
from gymnasium import spaces
from jax import Array
from typing import Any

@struct.dataclass
class Wrapper(struct.PyTreeNode):
    """Base class for jittable wrappers that delegates common metadata to the wrapped base."""

    base: struct.PyTreeNode = struct.field(pytree_node=True)

    @property
    def single_observation_space(self) -> spaces.Space:
        return getattr(self.base, "single_observation_space")

    @property
    def observation_space(self) -> spaces.Space:
        return getattr(self.base, "observation_space")

    @property
    def single_action_space(self) -> spaces.Space:
        return getattr(self.base, "single_action_space")

    @property
    def action_space(self) -> spaces.Space:
        return getattr(self.base, "action_space")

    @property
    def num_envs(self) -> int:
        return getattr(self.base, "num_envs")

    @property
    def unwrapped(self) -> struct.PyTreeNode:
        return getattr(self.base, "unwrapped", self.base)

    @property
    def steps(self) -> Array:
        return getattr(self.base, "steps")

    @staticmethod
    def recursive_replace(env: struct.PyTreeNode, **kwargs: Any) -> struct.PyTreeNode:
        """Recursively replace fields in the innermost base environment."""
        if isinstance(env, Wrapper):
            new_base = Wrapper.recursive_replace(env.base, **kwargs)
            return env.replace(base=new_base)
        return env.replace(**kwargs)

    def render(self, **kwargs: dict) -> None:
        return self.base.render(**kwargs)

    def close(self, **kwargs: Any) -> None:
        return self.base.close(**kwargs)
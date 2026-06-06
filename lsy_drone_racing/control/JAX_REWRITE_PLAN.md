# JAX Rewrite Plan for `train_rl.py`

## Current Architecture

The script has a mixed JAX/PyTorch split that creates friction:

- **Environment** (JAX): `RandTrajEnv` and all wrappers internally use `jax.numpy`. `FlattenJaxObservation` outputs JAX arrays.
- **JaxToTorch bridge**: The outermost wrapper converts JAX→PyTorch tensors on every step.
- **Agent** (PyTorch): `nn.Module` with PyTorch linear layers, `torch.distributions.Normal`, orthogonal init.
- **Optimizer** (PyTorch): `AdamW` with manual gradient clipping.
- **Training loop** (PyTorch + Python): Python `for` loops for rollout, GAE, minibatch updates. The GAE computation is a Python reverse loop.

**Primary bottlenecks**:
1. `JaxToTorch` — device-transfer or copy on every `step()` call (1024 envs × obs_dim floats per step)
2. Python loop over `num_steps=8` for rollout — prevents the entire rollout from being JIT-compiled
3. Python nested loops over `update_epochs × num_minibatches` — each gradient update is a separate kernel launch

---

## New Dependencies Required

Add to `pyproject.toml` under the `[rl]` extra:

| Package | Purpose | Replaces |
|---|---|---|
| `flax >= 0.10` | Neural network modules | `torch.nn` |
| `optax >= 0.2` | Optimizers + grad clipping | `torch.optim` |
| `orbax-checkpoint` | Model save/load | `torch.save` / `torch.load` |
| `distrax` (optional) | Normal distribution helpers | `torch.distributions.Normal` |

---

## Changes by Component

### 1. Imports & `Args`

- Drop: `torch`, `torch.nn`, `torch.optim`, `torch.distributions`, `Tensor`
- Add: `flax.linen as nn`, `optax`, `orbax.checkpoint as ocp`
- `Args`: remove `cuda`, `torch_deterministic`; `jax_device` already exists
- `set_seeds`: reduce to `random.seed`, `np.random.seed`; JAX PRNG becomes functional key management via `jax.random.PRNGKey(seed)`

### 2. `make_envs` — remove `JaxToTorch`

```python
# Remove this line:
env = JaxToTorch(env, torch_device)
# Remove torch_device parameter from signature entirely
```

The inner wrappers already use JAX arrays end-to-end. `AngleReward.step` uses `.at[].set()` (JAX), and
`FlattenJaxObservation.observations` returns `jp.concatenate(...)`. With `JaxToTorch` removed, actions go
in as JAX arrays and observations come out as JAX arrays with no conversion.

### 3. `Agent` — rewrite in Flax

```python
class Agent(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        # Critic
        v = nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
                     bias_init=nn.initializers.zeros)(x)
        v = nn.tanh(v)
        v = nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
                     bias_init=nn.initializers.zeros)(v)
        v = nn.tanh(v)
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0),
                         bias_init=nn.initializers.zeros)(v)

        # Actor
        m = nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
                     bias_init=nn.initializers.zeros)(x)
        m = nn.tanh(m)
        m = nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
                     bias_init=nn.initializers.zeros)(m)
        m = nn.tanh(m)
        mean = nn.tanh(nn.Dense(self.action_dim,
                                kernel_init=nn.initializers.orthogonal(0.01),
                                bias_init=nn.initializers.zeros)(m))

        log_std = self.param('log_std',
                             lambda rng, s: jnp.array([[-1.0, -1.0, -1.0, 1.0]]),
                             (1, self.action_dim))
        return mean, log_std, value
```

Key differences from PyTorch:
- Parameters are a separate pytree: `params = agent.init(key, sample_obs)['params']`
- Forward call is pure: `mean, log_std, value = agent.apply({'params': params}, obs)`
- `get_action_and_value` becomes a standalone `@jax.jit` function taking `params` as argument:

```python
@jax.jit
def get_action_and_value(params, obs, rng_key, action=None):
    mean, log_std, value = agent.apply({'params': params}, obs)
    std = jnp.exp(log_std)
    if action is None:
        action = mean + std * jax.random.normal(rng_key, mean.shape)
    log_prob = jnp.sum(
        -0.5 * ((action - mean) / std) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi),
        axis=-1,
    )
    entropy = jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e), axis=-1)
    return action, log_prob, entropy, value.squeeze(-1)
```

### 4. Optimizer — switch to Optax

```python
# With LR schedule (replaces the manual anneal_lr block):
schedule = optax.linear_schedule(args.learning_rate, 0.0,
                                  args.num_iterations * args.update_epochs * args.num_minibatches)
optimizer = optax.chain(
    optax.clip_by_global_norm(args.max_grad_norm),
    optax.adamw(schedule, eps=1e-5),
)

# Without LR schedule:
optimizer = optax.chain(
    optax.clip_by_global_norm(args.max_grad_norm),
    optax.adamw(args.learning_rate, eps=1e-5),
)

opt_state = optimizer.init(params)
```

The `anneal_lr` manual override block disappears — handled by the schedule.

### 5. Storage Buffers

Replace `torch.zeros(...)` with `jnp.zeros(...)`. Assignment uses `.at[step].set(...)`:

```python
obs = obs.at[step].set(next_obs)
values = values.at[step].set(value)
```

This creates new arrays but is JIT-friendly. Alternatively, keep Python lists and `jnp.stack` at the
end of the rollout (simpler for the conservative tier).

### 6. GAE Computation — replace Python reverse loop with `lax.scan`

```python
@jax.jit
def compute_gae(rewards, values, dones, next_value, next_done, gamma, gae_lambda):
    def scan_fn(lastgaelam, t):
        reward, value, done, nxt_value, nxt_done = t
        nextnonterminal = 1.0 - nxt_done
        delta = reward + gamma * nxt_value * nextnonterminal - value
        adv = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        return adv, adv

    # Build reversed (T, num_envs) inputs
    all_values = jnp.concatenate([values, next_value[None]], axis=0)  # (T+1, E)
    all_dones  = jnp.concatenate([dones,  next_done[None]],  axis=0)
    inputs = (
        rewards[::-1],
        values[::-1],
        dones[::-1],
        all_values[1:][::-1],
        all_dones[1:][::-1],
    )

    _, advantages_rev = jax.lax.scan(scan_fn, jnp.zeros(rewards.shape[1]), inputs)
    advantages = advantages_rev[::-1]
    return advantages, advantages + values
```

### 7. PPO Loss + Update Step — JIT-compiled

```python
@jax.jit
def ppo_loss(params, obs, actions, log_probs, advantages, returns, b_values):
    _, new_log_probs, entropy, new_values = get_action_and_value(params, obs, None, actions)
    log_ratio = new_log_probs - log_probs
    ratio = jnp.exp(log_ratio)

    # Policy loss (clipped surrogate)
    pg_loss = jnp.mean(jnp.maximum(
        -advantages * ratio,
        -advantages * jnp.clip(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF),
    ))

    # Value loss (clipped)
    v_pred_clipped = b_values + jnp.clip(new_values - b_values, -CLIP_COEF, CLIP_COEF)
    v_loss = 0.5 * jnp.mean(jnp.maximum(
        (new_values - returns) ** 2,
        (v_pred_clipped - returns) ** 2,
    ))

    entropy_loss = jnp.mean(entropy)
    total_loss = pg_loss - ENT_COEF * entropy_loss + VF_COEF * v_loss
    return total_loss, (pg_loss, v_loss, entropy_loss, ratio)


@jax.jit
def update_minibatch(carry, mb_indices):
    params, opt_state, flat_data = carry
    obs, acts, logps, advs, rets, vals = [x[mb_indices] for x in flat_data]
    (loss, aux), grads = jax.value_and_grad(ppo_loss, has_aux=True)(
        params, obs, acts, logps, advs, rets, vals
    )
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return (new_params, new_opt_state, flat_data), (loss, aux)


def update_epoch(params, opt_state, flat_data, rng_key):
    shuffled_inds = jax.random.permutation(rng_key, args.batch_size)
    mb_inds = shuffled_inds.reshape(args.num_minibatches, args.minibatch_size)
    (params, opt_state, _), metrics = jax.lax.scan(
        update_minibatch, (params, opt_state, flat_data), mb_inds
    )
    return params, opt_state, metrics
```

The nested `for epoch / for start` loop becomes `update_epochs` calls to `update_epoch`, each
`jax.jit`-compiled. The entire epoch is one XLA program — no Python overhead per minibatch.

### 8. Model Save/Load — switch to Orbax

```python
# Save
checkpointer = ocp.StandardCheckpointer()
checkpointer.save(model_path, args=ocp.args.StandardSave(params))

# Load
params = checkpointer.restore(model_path, args=ocp.args.StandardRestore(params_structure))
```

Simpler fallback (no extra setup): `pickle.dump(params, f)` / `pickle.load(f)` — params are plain
pytrees and serialize cleanly.

### 9. `evaluate_ppo`

- Drop `torch.load` / `agent.load_state_dict`
- Load params with orbax/pickle
- Replace `agent.get_action_and_value(obs, deterministic=True)` with `get_action_and_value(params, obs, rng_key=None)` using `mean` directly instead of sampling
- `eval_env` no longer needs `JaxToTorch`; remove `torch.device("cpu")` entirely

---

## Implementation Tiers

| Tier | Changes | Expected Speedup |
|---|---|---|
| **1 — Remove bridge** | Remove `JaxToTorch`, keep Python loops, PyTorch agent | ~1.5–2× (eliminates per-step transfers with 1024 envs) |
| **2 — JAX agent + optimizer** | Flax agent, Optax optimizer, JIT loss + update, `lax.scan` GAE | ~3–5× (entire update step is one XLA kernel, no Python grad overhead) |
| **3 — Scan rollout** | Wrap env in a pure functional JAX env, `lax.scan` the `num_steps` loop | ~6–10× (full training iteration is one compiled graph, no Python loop overhead) |

Tier 3 requires refactoring the env to expose a pure `(state, action) → (new_state, obs, reward, done)`
functional API, bypassing the gymnasium wrapper chain entirely. The underlying `DroneEnv` / `crazyflow`
simulation is already JAX-based, so this is feasible but invasive.

---

## Files to Change

| File | Scope |
|---|---|
| `lsy_drone_racing/control/train_rl.py` | Full rewrite of `Agent`, `set_seeds`, `train_ppo`, `evaluate_ppo`, `make_envs` |
| `pyproject.toml` | Add `flax`, `optax`, `orbax-checkpoint` under `[rl]` extra |
| `lsy_drone_racing/control/attitude_rl.py` | If it imports `Agent` from `train_rl`, update the import and param-loading logic |

---

## Open Questions

1. **Tier target**: Tier 1+2 only (conservative, keep gymnasium wrapper chain) or also Tier 3 (functional env wrapper, more invasive)?
2. **Network library**: `flax.linen` (most common for RL) vs `equinox` (PyTorch-style stateful modules, easier migration path)?
3. **Distribution library**: `distrax` (clean API, mirrors `torch.distributions`) vs manual `jax.scipy.stats.norm` (no extra dependency)?
4. **Checkpoint format**: Orbax (production-grade) or pickle (simplest)?

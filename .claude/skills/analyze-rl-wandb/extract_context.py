"""Extract the *live* RL training context for W&B chart analysis.

Reads the repo's config sources directly (no package import -> no jax, no
training, nothing heavy) and prints, for the run that uses repo HEAD:

  1. Effective hyperparameters / reward coefficients (Args defaults merged with
     the racing task overrides), with derived quantities (batch size, iteration
     count) and the curriculum schedule breakpoints in *global steps*.
  2. The live chart inventory -- every metric key the training code logs --
     grouped by prefix, so you can check pasted screenshots against what
     *should* exist.
  3. A PARAMETERS.md coverage check: which effective params have written
      up/down guidance and which are UNDOCUMENTED (analyze those from code +
     first principles, don't guess from a stale doc).

Nothing here is hardcoded to a fixed parameter or metric list: it parses
whatever is currently in the source, so newly added/removed knobs and charts
show up automatically. Run it at the start of every analysis.

    python .claude/skills/analyze-rl-wandb/extract_context.py
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # .claude/skills/analyze-rl-wandb/ -> repo root
CONFIG_PY = ROOT / "lsy_drone_racing/rl/config.py"
RACING_PY = ROOT / "lsy_drone_racing/rl/tasks/single_agent_racing.py"
SPAWN_PY = ROOT / "lsy_drone_racing/envs/segment_spawn.py"
PARAMS_MD = ROOT / "lsy_drone_racing/rl/PARAMETERS.md"
SCAN_DIRS = [ROOT / "lsy_drone_racing/rl", ROOT / "lsy_drone_racing/envs"]


def _literal(node: ast.AST) -> object:
    """Best-effort constant eval of an AST value node; returns a string if not literal."""
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return f"<{ast.unparse(node)}>"


def dataclass_defaults(path: Path, class_name: str) -> dict:
    """Pull `name: type = default` fields out of a @dataclass by name."""
    tree = ast.parse(path.read_text())
    out: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    if stmt.value is not None:
                        out[stmt.target.id] = _literal(stmt.value)
    return out


def return_dict_keys(path: Path, func_name: str) -> list[str]:
    """String-literal keys of the dict a named function returns (e.g. the reward components).

    The racing reward terms are logged as ``rew/<name>`` via an f-string, so the literal chart
    strings never appear in the source for ``chart_inventory``'s regex to find. Instead read them at
    their source of truth: the ``return {...}`` dict of ``racing_reward_components``.
    """
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Dict):
                    return [k.value for k in stmt.value.keys if isinstance(k, ast.Constant)]
    return []


def chart_inventory() -> dict[str, set[str]]:
    """Discover every logged metric key from the source (normalized to chart names)."""
    pat = re.compile(r'"((?:train|losses|charts|diagnostics|reward|rew|max)/[A-Za-z0-9_]+)"')
    keys: set[str] = set()
    for d in SCAN_DIRS:
        for py in d.rglob("*.py"):
            for m in pat.findall(py.read_text()):
                # Normalize to the names that actually appear on W&B (see ppo.py logging):
                if m.startswith("rew/"):
                    keys.add("reward/" + m[len("rew/"):])
                elif m.startswith("max/"):
                    keys.add("diagnostics/" + m[len("max/"):] + "_max")
                else:
                    keys.add(m)
    # The racing reward components are emitted via f-string (f"rew/{name}"), invisible to the regex
    # above -- add them from the racing_reward_components return dict (the single best chart group).
    for name in return_dict_keys(RACING_PY, "racing_reward_components"):
        keys.add(f"reward/{name}")
    groups: dict[str, set[str]] = {}
    for k in keys:
        groups.setdefault(k.split("/", 1)[0], set()).add(k)
    return groups


def main() -> None:
    """Print the effective hyperparameters, curriculum schedule, and chart inventory."""
    args = dataclass_defaults(CONFIG_PY, "Args")
    # The racing task overrides Args via the RacingArgs(Args) dataclass subclass (formerly a
    # RACING_CONFIG dict merged before Args.create). Its declared fields are exactly the overrides.
    racing = dataclass_defaults(RACING_PY, "RacingArgs")
    spawn = dataclass_defaults(SPAWN_PY, "SegmentSpawnConfig")

    effective = {**args, **racing}  # CLI kwargs override both at runtime; HEAD run = these.
    overridden = set(racing) & set(args)
    racing_only = set(racing) - set(args)

    print("=" * 78)
    print("EFFECTIVE HYPERPARAMETERS (Args defaults <- RacingArgs overrides)")
    print("HEAD run uses these unless overridden on the CLI. [*]=task override, [+]=task-only key")
    print("=" * 78)
    for k in sorted(effective):
        tag = "[*]" if k in overridden else ("[+]" if k in racing_only else "   ")
        print(f"  {tag} {k:24s} = {effective[k]}")

    # Derived quantities (mirrors Args.create) + curriculum breakpoints in global steps.
    try:
        T = int(effective.get("total_timesteps", 0))
        bs = int(effective.get("num_envs", 0)) * int(effective.get("num_steps", 0))
        print("\nDERIVED:")
        print(f"   batch_size      = num_envs*num_steps = {bs}")
        if bs:
            print(f"   num_iterations  = total_timesteps//batch_size = {T // bs}")
            print(f"   minibatch_size  = batch_size//num_minibatches  = "
                  f"{bs // int(effective.get('num_minibatches', 1))}")
    except (TypeError, ValueError):
        T = 0

    print("\nCURRICULUM SCHEDULE (tau = global_step / total_timesteps); breakpoints in steps:")
    for label, lo, hi in [
        ("kappa cone-size [a0,a1]", "a0", "a1"),
        ("true-start mix [b0,b1]", "b0", "b1"),
        ("initial speed  [c0,c1]", "c0", "c1"),
    ]:
        a, b = spawn.get(lo), spawn.get(hi)
        steps = (
            f"  ({int(a * T):,} -> {int(b * T):,} steps)"
            if (T and isinstance(a, (int, float)))
            else ""
        )
        print(f"   {label:26s} tau {a} -> {b}{steps}")
    print("   spawn config:", {k: spawn[k] for k in spawn})

    inv = chart_inventory()
    print("\n" + "=" * 78)
    print("LIVE CHART INVENTORY (every metric the code logs -- check screenshots vs this)")
    print("=" * 78)
    for grp in sorted(inv):
        print(f"  {grp}/")
        for k in sorted(inv[grp]):
            print(f"      {k}")

    # PARAMETERS.md coverage: flag effective knobs with no written guidance.
    md = PARAMS_MD.read_text() if PARAMS_MD.exists() else ""
    documented, undocumented = [], []
    skip = {"batch_size", "minibatch_size", "num_iterations", "seed", "jax_device",
            "wandb_project_name", "wandb_entity"}
    for k in sorted(effective):
        if k in skip:
            continue
        (documented if re.search(rf"`{re.escape(k)}`", md) else undocumented).append(k)
    print("\n" + "=" * 78)
    print("PARAMETERS.md COVERAGE")
    print("=" * 78)
    print(f"  documented ({len(documented)}): {', '.join(documented)}")
    print(f"  UNDOCUMENTED ({len(undocumented)}): {', '.join(undocumented) or '(none)'}")
    if undocumented:
        print("  ^ analyze these from code + first principles; PARAMETERS.md has no guidance.")
    print("\nNOTE: PARAMETERS.md is prose guidance and DRIFTS from code. Trust the values above,")
    print("use PARAMETERS.md only for the up/down 'what this knob does' reasoning.")


if __name__ == "__main__":
    main()

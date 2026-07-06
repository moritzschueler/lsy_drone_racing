"""Git provenance for training runs: pin the exact code behind each W&B run to a branch.

On every run we snapshot the working tree (tracked + untracked, minus ``.gitignore``) into a commit
and point a branch ``wandb-runs/<run_name>-<run_id>`` at it, so a chart legend maps straight to
reproducible code::

    git checkout wandb-runs/deep-forest-56-3a7bx9k2
    git diff wandb-runs/rare-energy-49-xxxx wandb-runs/deep-forest-56-yyyy -- lsy_drone_racing/rl/

The run *name* (``deep-forest-56``) is the readable handle from the chart legend; the run *id*
(``3a7bx9k2``, from the run URL) is appended as an immutable, collision-proof suffix -- redundancy
at no cost, so a renamed/reused name can never make a branch ambiguous.

The snapshot is built in a *temporary* index seeded from HEAD, so the user's real index, working
tree, and HEAD are never touched, and the branch is created via ``update-ref`` without switching to
it -- safe to call mid-launch. All git failures are swallowed: provenance must never abort training.

The branch is also pushed to the remote (``origin`` by default) as a best-effort backup, so the
provenance survives loss of the local clone. Set ``LSY_PROVENANCE_REMOTE=""`` to disable the push,
or to another remote name to back up elsewhere. A failed/missing remote never aborts the run.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile

_BRANCH_PREFIX = "wandb-runs"
_DEFAULT_REMOTE = "origin"


def _git(*args: str, env: dict | None = None) -> str:
    """Run a git command at the repo root and return stripped stdout (stderr suppressed)."""
    return subprocess.check_output(
        ["git", *args], text=True, env=env, stderr=subprocess.DEVNULL
    ).strip()


def _sanitize(component: str) -> str:
    """Make a string safe as a single git ref path component (W&B names are tame anyway)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", component).strip("-") or "run"


def _snapshot_tree(head: str, message: str) -> str:
    """Commit the full working tree (tracked + untracked) without touching index/worktree/HEAD.

    Uses a throwaway ``GIT_INDEX_FILE`` seeded from HEAD; ``add -A`` stages every change (still
    honoring ``.gitignore``). Returns the dangling commit's sha with HEAD as its parent.
    """
    with tempfile.NamedTemporaryFile() as tmp:
        env = {**os.environ, "GIT_INDEX_FILE": tmp.name}
        _git("read-tree", head, env=env)
        _git("add", "-A", env=env)
        tree = _git("write-tree", env=env)
        return _git("commit-tree", tree, "-p", head, "-m", message, env=env)


def _push_branch(branch: str) -> bool:
    """Best-effort backup of ``branch`` to the remote; returns whether the push succeeded.

    The remote defaults to ``origin`` and is overridable via ``LSY_PROVENANCE_REMOTE`` (empty
    disables the push). Any failure -- no remote, no network, no credentials -- is swallowed so
    provenance never aborts a training run.
    """
    remote = os.environ.get("LSY_PROVENANCE_REMOTE", _DEFAULT_REMOTE)
    if not remote:
        return False
    try:
        _git("push", "--force", remote, f"refs/heads/{branch}:refs/heads/{branch}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[git_provenance] branch push to '{remote}' failed (kept local): {e}")
        return False


def pin_run_to_branch(run_name: str | None, run_id: str | None) -> dict:
    """Snapshot the code behind a run and point ``wandb-runs/<run_name>-<run_id>`` at it.

    Returns git metadata for ``wandb.config`` (best-effort; ``git_sha`` is None if git is
    unavailable). Never raises -- a provenance failure must not break a training run.
    """
    try:
        head = _git("rev-parse", "HEAD")
        source_branch = _git("rev-parse", "--abbrev-ref", "HEAD")
        dirty = bool(_git("status", "--porcelain"))
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[git_provenance] skipped (not a git repo / git missing): {e}")
        return {
            "git_sha": None,
            "git_head": None,
            "git_branch": None,
            "git_dirty": None,
            "wandb_branch": None,
            "wandb_branch_pushed": False,
        }

    suffix = "-".join(_sanitize(p) for p in (run_name, run_id) if p) or "run"
    branch = f"{_BRANCH_PREFIX}/{suffix}"
    try:
        sha = _snapshot_tree(head, f"wandb run {run_name} ({run_id})") if dirty else head
        # Create-or-update the branch ref without checking it out (HEAD/working tree untouched).
        _git("update-ref", f"refs/heads/{branch}", sha)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[git_provenance] snapshot/branch failed: {e}")
        return {
            "git_sha": head,
            "git_head": head,
            "git_branch": source_branch,
            "git_dirty": dirty,
            "wandb_branch": None,
            "wandb_branch_pushed": False,
        }

    pushed = _push_branch(branch)

    return {
        "git_sha": sha,
        "git_head": head,
        "git_branch": source_branch,
        "git_dirty": dirty,
        "wandb_branch": branch,
        "wandb_branch_pushed": pushed,
    }

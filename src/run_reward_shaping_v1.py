from __future__ import annotations

"""Reward-shaping v1 launcher aligned with the thesis method section.

This script patches the installed VMAS ``navigation.py`` scenario in the current
Python process and then launches ``benchmarl.run``. The coefficients are kept
consistent with the thesis text:

    alpha_p = 0.5    # progress reward
    alpha_s = 0.05   # safety penalty
    alpha_m = 0.005  # smoothness penalty
    d_safe  = 0.15

Recommended usage:
    python src/restore_vmas_navigation.py
    python src/run_reward_shaping_v1.py \
        algorithm=mappo \
        task=vmas/navigation \
        experiment.render=false \
        experiment.evaluation=false \
        experiment.max_n_frames=300000 \
        seed=0
"""

import pathlib
import runpy
import sys


ALPHA_PROGRESS = 0.5
ALPHA_SAFETY = 0.05
ALPHA_SMOOTH = 0.005
SAFE_DISTANCE = 0.15


def _find_navigation_py() -> pathlib.Path:
    candidates = [
        pathlib.Path("/usr/local/lib/python3.12/dist-packages/vmas/scenarios/navigation.py"),
        pathlib.Path("/usr/local/lib/python3.11/dist-packages/vmas/scenarios/navigation.py"),
        pathlib.Path("/usr/local/lib/python3.10/dist-packages/vmas/scenarios/navigation.py"),
        pathlib.Path("/usr/local/lib/python3.9/dist-packages/vmas/scenarios/navigation.py"),
        pathlib.Path("/usr/local/lib/python3.8/dist-packages/vmas/scenarios/navigation.py"),
    ]
    for path in candidates:
        if path.exists():
            return path

    try:
        import vmas  # type: ignore

        root = pathlib.Path(vmas.__file__).resolve().parent
        candidate = root / "scenarios" / "navigation.py"
        if candidate.exists():
            return candidate
    except Exception:
        pass

    raise FileNotFoundError(
        "Could not locate VMAS navigation.py under common site-packages paths."
    )


def patch_navigation_reward() -> pathlib.Path:
    nav_path = _find_navigation_py()
    src = nav_path.read_text(encoding="utf-8")

    # Remove any previous project patches. This is important in Colab because
    # previous APF-AW or reward-shaping runs modify site-packages in place.
    for marker in ["# [apf_aw_patch]", "# [rs_patch]"]:
        if marker in src:
            src = src[: src.index(marker)].rstrip()
            print(f"[reward_shaping_v1] removed old patch marker: {marker}")

    patch = f'''
# [rs_patch]
import torch as _torch

_rs_orig_reward = Scenario.reward
_rs_orig_reset = Scenario.reset_world_at

_RS_ALPHA_PROGRESS = float({ALPHA_PROGRESS})
_RS_ALPHA_SAFETY = float({ALPHA_SAFETY})
_RS_ALPHA_SMOOTH = float({ALPHA_SMOOTH})
_RS_SAFE_DISTANCE = float({SAFE_DISTANCE})


def _rs_reset(self, env_index=None):
    out = _rs_orig_reset(self, env_index)
    if not hasattr(self, "_rs_pd"):
        self._rs_pd = {{}}
        self._rs_pa = {{}}
    if env_index is None:
        self._rs_pd.clear()
        self._rs_pa.clear()
    else:
        for d in (self._rs_pd, self._rs_pa):
            for k, v in list(d.items()):
                if isinstance(v, _torch.Tensor) and v.ndim > 0 and env_index < v.shape[0]:
                    v = v.clone()
                    v[env_index] = 0.0
                    d[k] = v
    return out


def _rs_reward(self, agent):
    base = _rs_orig_reward(self, agent)
    if not isinstance(base, _torch.Tensor):
        return base

    r = base.clone()
    zero = _torch.zeros_like(r)

    if not hasattr(self, "_rs_pd"):
        self._rs_pd = {{}}
        self._rs_pa = {{}}

    key = getattr(agent, "name", str(id(agent)))
    ap = getattr(getattr(agent, "state", None), "pos", None)
    gp = getattr(getattr(getattr(agent, "goal", None), "state", None), "pos", None)
    dist = _torch.linalg.vector_norm(ap - gp, dim=-1) if (ap is not None and gp is not None) else None

    # 1) Progress reward: alpha_p times previous-distance minus current-distance.
    prog = zero
    if dist is not None:
        pd = self._rs_pd.get(key)
        if pd is not None and pd.shape == dist.shape:
            prog = _RS_ALPHA_PROGRESS * (pd - dist)
        self._rs_pd[key] = dist.detach().clone()

    # 2) Smoothness penalty: alpha_m times the L2 norm of action difference.
    smooth = zero
    act = getattr(getattr(agent, "action", None), "u", None)
    if act is not None:
        pa = self._rs_pa.get(key)
        if pa is not None and pa.shape == act.shape:
            smooth = _RS_ALPHA_SMOOTH * _torch.linalg.vector_norm(act - pa, dim=-1)
        self._rs_pa[key] = act.detach().clone()

    # 3) Safety penalty around non-goal landmarks.
    risk = zero
    world = getattr(self, "world", None)
    landmarks = getattr(world, "landmarks", []) if world else []
    mc = None

    ar = 0.0
    try:
        ar = float(getattr(getattr(agent, "shape", None), "radius", 0.0))
    except Exception:
        pass

    for lm in landmarks:
        if lm is getattr(agent, "goal", None):
            continue
        lp = getattr(getattr(lm, "state", None), "pos", None)
        if lp is None or ap is None:
            continue
        lr = 0.0
        try:
            lr = float(getattr(getattr(lm, "shape", None), "radius", 0.0))
        except Exception:
            pass
        cl = _torch.linalg.vector_norm(ap - lp, dim=-1) - ar - lr
        mc = cl if mc is None else _torch.minimum(mc, cl)

    if mc is not None:
        thr = _torch.as_tensor(_RS_SAFE_DISTANCE, dtype=mc.dtype, device=mc.device)
        risk = _RS_ALPHA_SAFETY * _torch.relu(thr - mc)

    return r + prog - smooth - risk


Scenario.reward = _rs_reward
Scenario.reset_world_at = _rs_reset
print(
    "[rs_patch] reward shaping patch applied "
    f"alpha_p={{_RS_ALPHA_PROGRESS}}, alpha_s={{_RS_ALPHA_SAFETY}}, "
    f"alpha_m={{_RS_ALPHA_SMOOTH}}, d_safe={{_RS_SAFE_DISTANCE}}"
)
'''

    src = src + "\n\n" + patch + "\n"
    nav_path.write_text(src, encoding="utf-8")

    pycache_dir = nav_path.parent / "__pycache__"
    if pycache_dir.exists():
        for pyc in pycache_dir.glob("navigation.cpython-*.pyc"):
            try:
                pyc.unlink()
            except Exception:
                pass

    print(f"[reward_shaping_v1] patched {nav_path}")
    return nav_path


def main() -> None:
    patch_navigation_reward()
    sys.argv = ["benchmarl.run"] + sys.argv[1:]
    runpy.run_module("benchmarl.run", run_name="__main__")


if __name__ == "__main__":
    main()

from __future__ import annotations

import pathlib
import runpy
import sys


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
    raise FileNotFoundError(
        "Could not locate VMAS navigation.py under common site-packages paths."
    )


def patch_navigation_reward() -> pathlib.Path:
    nav_path = _find_navigation_py()
    src = nav_path.read_text(encoding="utf-8")

    marker = "# [rs_patch]"
    if marker in src:
        src = src[: src.index(marker)].rstrip()
        print("[reward_shaping_v1] removed old reward-shaping patch")

    patch = r'''
# [rs_patch]
import torch as _torch

_rs_orig_reward = Scenario.reward
_rs_orig_reset = Scenario.reset_world_at


def _rs_reset(self, env_index=None):
    out = _rs_orig_reset(self, env_index)
    if not hasattr(self, "_rs_pd"):
        self._rs_pd = {}
        self._rs_pa = {}
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
        self._rs_pd = {}
        self._rs_pa = {}

    key = getattr(agent, "name", str(id(agent)))
    ap = getattr(getattr(agent, "state", None), "pos", None)
    gp = getattr(getattr(getattr(agent, "goal", None), "state", None), "pos", None)
    dist = _torch.linalg.vector_norm(ap - gp, dim=-1) if (ap is not None and gp is not None) else None

    prog = zero
    if dist is not None:
        pd = self._rs_pd.get(key)
        if pd is not None and pd.shape == dist.shape:
            prog = 0.1 * (pd - dist)
        self._rs_pd[key] = dist.detach().clone()

    smooth = zero
    act = getattr(getattr(agent, "action", None), "u", None)
    if act is not None:
        pa = self._rs_pa.get(key)
        if pa is not None and pa.shape == act.shape:
            smooth = 0.01 * _torch.linalg.vector_norm(act - pa, dim=-1)
        self._rs_pa[key] = act.detach().clone()

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
        thr = _torch.as_tensor(0.15, dtype=mc.dtype, device=mc.device)
        risk = 0.1 * _torch.relu(thr - mc)

    return r + prog - smooth - risk


Scenario.reward = _rs_reward
Scenario.reset_world_at = _rs_reset
print("[rs_patch] reward shaping patch applied")
'''

    src = src + "\n\n" + patch + "\n"
    nav_path.write_text(src, encoding="utf-8")

    pycache_dir = nav_path.parent / "__pycache__"
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

from __future__ import annotations

"""Restore VMAS navigation.py after reward-patching experiments.

The reward-shaping launchers in this project patch the installed VMAS
`navigation.py` file by appending a marker block. This helper removes any known
project patch blocks so a clean baseline can be run in the same Colab runtime.

Usage:
    python src/restore_vmas_navigation.py
"""

import pathlib


def find_navigation_py() -> pathlib.Path:
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

    raise FileNotFoundError("Could not locate vmas/scenarios/navigation.py")


def main() -> None:
    nav_path = find_navigation_py()
    src = nav_path.read_text(encoding="utf-8")
    original = src

    for marker in ["# [apf_aw_patch]", "# [rs_patch]"]:
        if marker in src:
            src = src[: src.index(marker)].rstrip() + "\n"
            print(f"removed patch block: {marker}")

    if src == original:
        print("No project patch marker found. navigation.py already looks clean.")
    else:
        nav_path.write_text(src, encoding="utf-8")
        pycache_dir = nav_path.parent / "__pycache__"
        if pycache_dir.exists():
            for pyc in pycache_dir.glob("navigation.cpython-*.pyc"):
                try:
                    pyc.unlink()
                except Exception:
                    pass
        print(f"Restored clean navigation.py: {nav_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=None)
def load_1a_module(module_name: str):
    repo_root = Path(__file__).resolve().parents[1]
    source_dir = repo_root / "1a_BERTopic"
    module_path = source_dir / f"{module_name}.py"
    if not module_path.exists():
        raise ModuleNotFoundError(f"Could not find source module: {module_path}")

    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

    spec = importlib.util.spec_from_file_location(f"compat_{module_name}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


def export_public_names(module_name: str) -> dict[str, object]:
    module = load_1a_module(module_name)
    exported_names = getattr(module, "__all__", None)
    if exported_names is None:
        exported_names = [name for name in vars(module) if not name.startswith("__")]
    return {name: getattr(module, name) for name in exported_names}

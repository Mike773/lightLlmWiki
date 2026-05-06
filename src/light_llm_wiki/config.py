"""User-config loader for the CLI.

Looks for `lightllm_config.py` in the current working directory and
imports it as an ad-hoc module. The CLI reads attributes from the
returned module: optional `dsn`, required `get_llm()`, optional
`get_embeddings()`.
"""
import importlib.util
from pathlib import Path
from types import ModuleType

CONFIG_FILENAME = "lightllm_config.py"


def load_user_config(path: Path | None = None) -> ModuleType | None:
    target = path or Path.cwd() / CONFIG_FILENAME
    if not target.exists():
        return None
    spec = importlib.util.spec_from_file_location("lightllm_user_config", target)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

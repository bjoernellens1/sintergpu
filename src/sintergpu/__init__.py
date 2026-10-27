"""Versioned portable kernels with explicit runtime ownership."""
from importlib.resources import files
from pathlib import Path

__version__ = files(__package__).joinpath('data/VERSION').read_text().strip()


def kernel_path(operation_id: str) -> Path:
    from .catalog import get_operation, resource_path
    return resource_path(get_operation(operation_id)['path'])


def module_path(module_id: str) -> Path:
    """Locate an importable device-function module from the released package."""
    import json
    from .catalog import resource_path
    modules = json.loads(files(__package__).joinpath('data/modules.json').read_text())
    try:
        return resource_path(modules[module_id]['path'])
    except KeyError:
        raise KeyError(f'unknown device module: {module_id}') from None

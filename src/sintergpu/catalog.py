"""Public operation catalog. Qualification is scoped, never inferred."""
from importlib.resources import files
import hashlib
import json
from pathlib import Path
import re


def resource_path(relative: str) -> Path:
    root = Path(str(files('sintergpu'))).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('resource path escapes package')
    return path


def list_operations() -> list[dict]:
    return json.loads(files('sintergpu').joinpath('data/catalog.json').read_text())['operations']


def get_operation(operation_id: str) -> dict:
    for operation in list_operations():
        if operation['id'] == operation_id:
            return operation
    raise KeyError(f'unknown operation: {operation_id}')


def validate_catalog(operations=None) -> list[str]:
    errors, seen = [], set()
    for op in list_operations() if operations is None else operations:
        try:
            name = op['id']
            if name in seen:
                raise ValueError('duplicate operation')
            seen.add(name)
            for field in ('entrypoint', 'contract', 'requirements', 'qualification', 'variant'):
                if not op.get(field):
                    raise ValueError(f'missing {field}')
            if not re.fullmatch(r'[a-zA-Z0-9_.]+', name):
                raise ValueError('invalid operation id')
            if not re.fullmatch(r'[0-9a-f]{40}', op['source']['revision']):
                raise ValueError('source revision must be an immutable commit')
            if not op['source']['license'] or not op['source']['paths']:
                raise ValueError('missing source provenance')
            path = resource_path(op['path'])
            if hashlib.sha256(path.read_bytes()).hexdigest() != op['sha256']:
                raise ValueError('source hash mismatch')
            if op['qualification']['status'] not in ('experimental', 'pending', 'qualified'):
                raise ValueError('invalid qualification state')
            if op['qualification']['status'] == 'qualified' and not op['qualification'].get('records'):
                raise ValueError('qualified implementation requires scoped evidence records')
        except (KeyError, ValueError, OSError, TypeError) as exc:
            errors.append(f'{op.get("id", "<missing>")}: {exc}')
    return errors

#!/usr/bin/env python3
"""Check imported file bytes against the complete source ledger."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(root=ROOT):
    errors = []
    for ledger in sorted((root/'provenance').glob('*.json')):
        data = json.loads(ledger.read_text())
        records = list(data.get('files', []))
        records += [{'destination': i['path'], 'imported_sha256': i['sha256']}
                    for i in data.get('generation', {}).get('outputs', [])]
        records += [{'destination': p, 'imported_sha256': h}
                    for p, h in data.get('audit_inputs_sha256', {}).items()]
        for item in records:
            destination = item.get('destination')
            digest = item.get('imported_sha256')
            if destination is None or digest is None:
                errors.append(f'{ledger.name}: missing destination/imported_sha256')
                continue
            path = (root/destination).resolve()
            if not path.is_relative_to(root.resolve()):
                errors.append(f'{ledger.name}: path escape')
            elif not path.is_file():
                errors.append(f'{ledger.name}: missing {destination}')
            elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                errors.append(f'{ledger.name}: modified {destination}')
    return errors


if __name__ == '__main__':
    errors = check()
    print(json.dumps({'status':'failed' if errors else 'passed','errors':errors},indent=2))
    raise SystemExit(bool(errors))

"""Explicit test/benchmark harness; never silently substitutes a backend."""
import argparse
import json
import sys
from .catalog import get_operation, list_operations, validate_catalog


def main(argv=None):
    parser = argparse.ArgumentParser(prog='sinter')
    commands = parser.add_subparsers(dest='command', required=True)
    listing = commands.add_parser('list')
    listing.add_argument('--json', action='store_true')
    timings = commands.add_parser('timings')
    timings.add_argument('operation',nargs='?')
    timings.add_argument('--json',action='store_true')
    info = commands.add_parser('info')
    info.add_argument('operation')
    commands.add_parser('validate')
    for name in ('test', 'bench'):
        command = commands.add_parser(name)
        command.add_argument('operation', nargs='?')
        command.add_argument('--all', action='store_true')
        command.add_argument('--backend', choices=('reference', 'vulkan'), default='vulkan')
        command.add_argument('--device', type=int, default=0)
        command.add_argument('--iterations', type=int, default=100)
        command.add_argument('--output')
    args = parser.parse_args(argv)
    try:
        if args.command == 'list':
            ops = list_operations()
            print(json.dumps(ops, indent=2) if args.json else '\n'.join(op['id'] for op in ops))
        elif args.command == 'info':
            print(json.dumps(get_operation(args.operation), indent=2))
        elif args.command == 'timings':
            from importlib.resources import files
            from .timings import build_registry,render_markdown
            stored=json.loads(files('sintergpu').joinpath('data/timings.json').read_text())
            operations=[get_operation(args.operation)] if args.operation else list_operations()
            ids={op['id'] for op in operations}
            report=build_registry(operations,[r for r in stored['measurements'] if r['operation'] in ids])
            if not args.operation:report['additional_pending_targets']=stored.get('additional_pending_targets',[])
            print(json.dumps(report,indent=2,allow_nan=False) if args.json else render_markdown(report),end='\n' if args.json else '')
        elif args.command == 'validate':
            errors = validate_catalog()
            print(json.dumps({'errors': errors, 'operations': len(list_operations())}))
            return bool(errors)
        else:
            if args.all == bool(args.operation):
                parser.error('specify one operation or --all')
            operations = list_operations() if args.all else [get_operation(args.operation)]
            from .validation import run
            report = run(operations, backend=args.backend, device_index=args.device,
                         iterations=args.iterations if args.command == 'bench' else 1)
            payload = json.dumps(report, indent=2, allow_nan=False)
            if args.output:
                from pathlib import Path
                destination = Path(args.output)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(payload + '\n')
            print(payload)
            return 0 if report['status'] == 'passed' else 2
        return 0
    except (KeyError, ValueError, RuntimeError, ImportError, OSError) as exc:
        print(f'sinter: {exc}', file=sys.stderr)
        return 2

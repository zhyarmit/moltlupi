import argparse
import json
from pathlib import Path

from .discovery import discover
from .io import load_windows, read_json
from .pipeline import make_packets, phase2, phase3, resolve_config, run


def main():
    parser = argparse.ArgumentParser(prog='moltlupi')
    subs = parser.add_subparsers(dest='command', required=True)
    for name in ['run', 'phase1', 'packets', 'phase2', 'phase3']:
        sub = subs.add_parser(name)
        sub.add_argument('--config', required=True)
        if name in {'phase2', 'phase3'}:
            sub.add_argument('--output-dir')
        if name == 'phase3':
            sub.add_argument('--annotations')
    args = parser.parse_args()
    try:
        if args.command == 'run':
            result = run(args.config)
        else:
            config = resolve_config(args.config)
            out = Path(config['output_dir'])
            out.mkdir(parents=True, exist_ok=True)
            result = Path(args.output_dir) if getattr(args, 'output_dir', None) else out / args.command
            if args.command == 'phase1':
                discover(config['discovery'], load_windows(config['windows']), result)
            elif args.command == 'packets':
                make_packets(read_json(out / 'phase1' / 'clusters.json'), load_windows(config['windows']), config['posts'], out)
                result = out / 'packets.jsonl'
            elif args.command == 'phase2':
                phase2(config.get('annotation', {}), out / 'packets.jsonl', result)
            elif args.command == 'phase3':
                annotations = Path(args.annotations) if args.annotations else out / 'phase2' / 'post_annotations.jsonl'
                phase3(config.get('assessment', {}), out / 'packets.jsonl', annotations, out / 'phase1' / 'clusters.json', result)
        print(json.dumps({'output': str(result)}))
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, 'Error: ' + str(exc) + '\n')


if __name__ == '__main__':
    main()

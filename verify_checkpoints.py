"""Verify the supplied release files without deserializing neural checkpoints."""
from pathlib import Path
import argparse
import hashlib
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', nargs='?', default='checkpoints')
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name('checkpoint_manifest.json').read_text())
    failed = []
    for entry in manifest:
        path = Path(args.directory) / entry['name']
        good = path.is_file() and path.stat().st_size == entry['bytes']
        if good:
            with path.open('rb') as stream:
                good = hashlib.file_digest(stream, 'sha256').hexdigest() == entry['sha256']
        print(('OK ' if good else 'FAIL ') + entry['name'])
        if not good:
            failed.append(entry['name'])
    raise SystemExit(1 if failed else 0)

if __name__ == '__main__':
    main()

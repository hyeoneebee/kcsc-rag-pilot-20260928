"""Record distributable file hashes; never reads .git or generated caches."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE = {'SHA256SUMS', 'provenance/HANDOFF_VERIFICATION.json'}


def payload_files(root):
    for path in sorted(root.rglob('*')):
        rel = path.relative_to(root)
        if any(part in {'.git', '__pycache__'} or part.startswith('.venv') for part in rel.parts):
            continue
        if path.is_file() and rel.as_posix() not in EXCLUDE:
            yield path


if __name__ == '__main__':
    lines = [f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}' for p in payload_files(ROOT)]
    (ROOT / 'SHA256SUMS').write_text('\n'.join(lines) + '\n')
    print(f'Sealed {len(lines)} files. Report and SHA256SUMS itself are excluded.')

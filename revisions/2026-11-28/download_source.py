"""Download the dated research source release and verify its SHA-256 digest."""
import argparse
import hashlib
from pathlib import Path
import urllib.request
import zipfile

BASE = 'https://github.com/Wangwuyu-hub/saf-predictor-V1/releases/download/v3.1.0-20261128/'
NAME = 'Prediction_Platform_Source_20261128.zip'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('.'))
    parser.add_argument('--download-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checksums = urllib.request.urlopen(BASE + 'SHA256SUMS.txt').read().decode('utf-8')
    expected = next(line.split()[0] for line in checksums.splitlines()
                    if line.split()[-1].lstrip('*') == NAME)
    archive = args.output / NAME
    if not archive.exists() or hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
        temporary = archive.with_suffix('.zip.download')
        with urllib.request.urlopen(BASE + NAME) as response, temporary.open('wb') as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        actual = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Checksum mismatch: expected {expected}; received {actual}')
        temporary.replace(archive)
    print(f'Verified {NAME}: {expected}')
    if not args.download_only:
        destination = args.output / 'Prediction_Platform_Source_20261128'
        if destination.exists():
            raise FileExistsError(f'Refusing to overwrite an existing source directory: {destination}')
        with zipfile.ZipFile(archive) as z:
            root = args.output.resolve()
            for item in z.infolist():
                target = (root / item.filename).resolve()
                if not target.is_relative_to(root):
                    raise ValueError('Unsafe archive member path')
            z.extractall(root)
        print(f'Extracted to {destination.resolve()}')


if __name__ == '__main__':
    main()

"""Verify and assemble the dated source archive using standard-library Python."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_source(root: Path, expected_manifest: bytes) -> int:
    manifest_path = root / 'FILE_MANIFEST_SHA256.json'
    if manifest_path.read_bytes() != expected_manifest:
        raise ValueError('The extracted source manifest differs from the published manifest.')
    expected = json.loads(expected_manifest)['files']
    ignored = {'FILE_MANIFEST_SHA256.json', 'ARCHIVE_CONTENTS_SHA256.json'}
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.relative_to(root).as_posix() not in ignored}
    if actual != set(expected):
        raise ValueError('The extracted file list differs from the published source manifest.')
    for name, record in expected.items():
        path = root / name
        if path.is_symlink() or path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise ValueError('Source file integrity failure: ' + name)
    return len(expected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'source',
                        help='Directory in which to write the complete ZIP and extract its source folder.')
    parser.add_argument('--no-extract', action='store_true', help='Verify and assemble the complete ZIP without extracting it.')
    args = parser.parse_args()
    parts_dir = Path(__file__).resolve().parent
    manifest = json.loads((parts_dir / 'ARCHIVE_PARTS_SHA256.json').read_text(encoding='utf-8'))
    archive = manifest['archive']
    if Path(archive['name']).name != archive['name']:
        raise ValueError('Invalid archive filename.')
    expected_manifest = (parts_dir / 'SOURCE_FILES_SHA256.json').read_bytes()
    if hashlib.sha256(expected_manifest).hexdigest() != manifest['source_manifest_sha256']:
        raise ValueError('Published source manifest integrity failure.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_zip = args.output_dir / archive['name']
    descriptor, temporary_name = tempfile.mkstemp(prefix='.assembling-', suffix='.zip', dir=args.output_dir)
    import os
    os.close(descriptor)
    temporary_zip = Path(temporary_name)
    full_digest = hashlib.sha256()
    full_size = 0
    try:
        with temporary_zip.open('wb') as dest:
            for part in manifest['parts']:
                name = part['name']
                if Path(name).name != name:
                    raise ValueError('Invalid part filename.')
                path = parts_dir / name
                if path.is_symlink() or path.stat().st_size != part['bytes']:
                    raise ValueError('Archive part size failure: ' + name)
                digest = hashlib.sha256()
                with path.open('rb') as source:
                    for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
                        digest.update(block)
                        full_digest.update(block)
                        full_size += len(block)
                        dest.write(block)
                if digest.hexdigest() != part['sha256']:
                    raise ValueError('Archive part integrity failure: ' + name)
        if full_size != archive['bytes'] or full_digest.hexdigest() != archive['sha256']:
            raise ValueError('Combined archive integrity failure.')
        if output_zip.exists():
            if output_zip.stat().st_size != archive['bytes'] or sha(output_zip) != archive['sha256']:
                raise FileExistsError('A different archive already exists at ' + str(output_zip))
            temporary_zip.unlink()
        else:
            temporary_zip.replace(output_zip)
    finally:
        if temporary_zip.exists():
            temporary_zip.unlink()
    result = {'archive': str(output_zip.resolve()), 'archive_sha256': archive['sha256'], 'archive_bytes': archive['bytes'], 'parts_verified': len(manifest['parts'])}
    if not args.no_extract:
        top = archive['top_directory']
        if PurePosixPath(top).name != top:
            raise ValueError('Invalid source directory.')
        source_root = args.output_dir / top
        if source_root.exists():
            count = verify_source(source_root, expected_manifest)
        else:
            temporary_root = Path(tempfile.mkdtemp(prefix='.extracting-', dir=args.output_dir))
            try:
                with zipfile.ZipFile(output_zip) as bundle:
                    names = bundle.namelist()
                    if len(names) != len(set(names)):
                        raise ValueError('Duplicate archive member names.')
                    for member in bundle.infolist():
                        path = PurePosixPath(member.filename)
                        if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] != top or '\\' in member.filename:
                            raise ValueError('Unsafe archive member path.')
                        if stat.S_ISLNK(member.external_attr >> 16):
                            raise ValueError('Symbolic links are not part of the published archive.')
                    if bundle.testzip() is not None:
                        raise ValueError('ZIP CRC verification failed.')
                    bundle.extractall(temporary_root)
                count = verify_source(temporary_root / top, expected_manifest)
                (temporary_root / top).replace(source_root)
            finally:
                shutil.rmtree(temporary_root)
        result.update({'source_directory': str(source_root.resolve()), 'source_files_verified': count})
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

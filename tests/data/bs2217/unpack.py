from __future__ import annotations
from typing import NamedTuple, Literal, cast
from pathlib import Path
import shutil
import tempfile
import zipfile
import json

import numpy as np
from scipy.io import wavfile

HERE = Path(__file__).parent
ZIP_DIR = HERE / 'zipfiles'
WAV_DIR = HERE / 'wavfiles'
NPZ_DIR = HERE / 'npzfiles'
META_FILE = HERE / 'meta.csv'
META_FILE_JSON = HERE / 'meta.json'


class FileMeta(NamedTuple):
    name: str
    num_channels: int
    lkfs_expected: float



class FileMetaFull(NamedTuple):
    name: str
    num_channels: int
    sample_rate: int
    bit_depth: int
    is_float: bool
    lkfs_expected: float

    @classmethod
    def from_meta(cls, meta: FileMeta, sample_rate: int, bit_depth: int, is_float: bool) -> FileMetaFull:
        return cls(
            name=meta.name,
            num_channels=meta.num_channels,
            sample_rate=sample_rate,
            bit_depth=bit_depth,
            is_float=is_float,
            lkfs_expected=meta.lkfs_expected,
        )

    def serialize(self) -> dict[str, str|int|float|bool]:
        return {
            'name': self.name,
            'num_channels': self.num_channels,
            'sample_rate': self.sample_rate,
            'bit_depth': self.bit_depth,
            'is_float': self.is_float,
            'lkfs_expected': self.lkfs_expected,
        }


def load_metadata(meta_file: Path) -> dict[str, FileMeta]:
    metadata = {}
    with meta_file.open('r', encoding='utf-8') as f:
        lines = f.readlines()
    def read_line(line: str) -> list[str]:
        return [item.strip() for item in line.strip().split(',')]
    header = read_line(lines[0])
    assert 'name' in header
    assert 'num_channels' in header
    assert 'lkfs_expected' in header
    keys = cast(list[Literal['name', 'num_channels', 'lkfs_expected']], header)
    for line in lines[1:]:
        if not line.strip():
            continue
        items = read_line(line)
        assert len(items) == len(keys)
        item_dict = dict(zip(keys, items))
        item = FileMeta(
            name=item_dict['name'],
            num_channels=int(item_dict['num_channels']),
            lkfs_expected=float(item_dict['lkfs_expected']),
        )
        metadata[item.name] = item
    return metadata


METADATA = load_metadata(META_FILE)


def unpack_zipfile(zip_filename: Path, wav_dir: Path) -> Path:
    """Unpack the given zip file and place its contents (a single .wav file) into the given directory

    """
    with zipfile.ZipFile(zip_filename, 'r') as zip_file:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            zip_file.extractall(temp_path)
            wav_files = list(temp_path.glob('*.wav'))
            if len(wav_files) != 1:
                raise ValueError(f'Expected exactly one .wav file in the zip archive, found {len(wav_files)}')
            wav_file = wav_files[0]
            base_name = wav_file.stem
            if base_name not in METADATA:
                # check for wav file named with spaces instead of underscores
                # alt_meta_name = base_name.replace('_', ' ').replace('-', ' ')
                alt_base_name = base_name.replace(' ', '_').replace('-', '_')
                for meta_name in METADATA.keys():
                    if meta_name.replace(' ', '_').replace('-', '_') == alt_base_name:
                        base_name = meta_name
                        break
                else:
                    raise ValueError(f'No metadata found for {base_name}')
            wav_file_renamed = wav_file.parent / (base_name + '.wav')
            wav_file.rename(wav_file_renamed)
            wav_file = wav_file_renamed
            dest_file = wav_dir / wav_file.name
            if dest_file.exists():
                return dest_file
            shutil.copy(wav_file, dest_file)
            return dest_file


def unpack_all_zipfiles():
    WAV_DIR.mkdir(exist_ok=True)
    for zip_file in ZIP_DIR.glob('*.zip'):
        unpack_zipfile(zip_file, WAV_DIR)


def load_npz_metadata(npz_file: Path) -> FileMetaFull:
    with np.load(npz_file) as npz:
        name = npz_file.stem
        num_channels = int(npz['num_channels'])
        sample_rate = int(npz['sample_rate'])
        bit_depth = int(npz['bit_depth'])
        is_float = bool(npz['is_float'])
        lkfs_expected = float(npz['lkfs_expected'])
        return FileMetaFull(
            name=name,
            num_channels=num_channels,
            sample_rate=sample_rate,
            bit_depth=bit_depth,
            is_float=is_float,
            lkfs_expected=lkfs_expected,
        )


def convert_wav_to_npz(wav_file: Path, npz_dir: Path, base_meta: FileMeta) -> tuple[Path, FileMetaFull]:
    npz_file = npz_dir / (wav_file.stem + '.npz')
    if npz_file.exists():
        print(f'NPZ file already exists: {npz_file}, loading metadata')
        return npz_file, load_npz_metadata(npz_file)
    sample_rate, data = wavfile.read(wav_file)
    if data.ndim == 1:
        # For mono files, ensure shape is (num_samples, 1)
        assert base_meta.num_channels == 1
        data = np.reshape(data, (data.shape[0], 1))

    assert len(data.shape) == 2
    assert data.shape[1] == base_meta.num_channels, f'{data.shape=}, {base_meta.num_channels=}'

    meta_full = FileMetaFull.from_meta(
        base_meta,
        sample_rate=sample_rate,
        bit_depth=data.dtype.itemsize * 8,
        is_float=np.issubdtype(data.dtype, np.floating),
    )

    np.savez_compressed(
        npz_file,
        sample_rate=sample_rate,
        bit_depth=data.dtype.itemsize * 8,
        is_float=np.issubdtype(data.dtype, np.floating),
        num_channels=meta_full.num_channels,
        lkfs_expected=base_meta.lkfs_expected,
        samples=data,
    )
    return npz_file, meta_full

def convert_all_wav_to_npz():
    NPZ_DIR.mkdir(exist_ok=True)
    metadata = load_metadata(META_FILE)
    all_meta_full: dict[str, FileMetaFull] = {}
    for meta in metadata.values():
        wav_file = WAV_DIR / (meta.name + '.wav')
        if not wav_file.exists():
            raise FileNotFoundError(f'WAV file not found: {wav_file}')
        npz_file, meta_full = convert_wav_to_npz(wav_file, NPZ_DIR, meta)
        all_meta_full[meta.name] = meta_full
    with META_FILE_JSON.open('w', encoding='utf-8') as f:
        json.dump(
            {name: meta.serialize() for name, meta in all_meta_full.items()},
            f,
            indent=4,
        )

if __name__ == '__main__':
    # unpack_all_zipfiles()
    convert_all_wav_to_npz()

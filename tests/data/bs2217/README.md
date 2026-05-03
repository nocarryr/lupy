# ITU-R BS.2217 Compliance Material

This directory contains compliance material from [ITU-R BS.2217] which is the official compliance testing source for [ITU-R BS.1770].


- The `.npz` files were read from the original `.wav` files (48k / 16-bit) without modification
  - `npz` was used primarily for its (lossless) compression.
- The files with channel counts greater than 6 have been omitted as they are not relevant to this project.
- The expected `LKFS` results given in BS.2217 are stored in `meta.csv`.
- `meta.json` contains the same information along with other fields generated when converting from wave to npz.
- The `unpack.py` script is included here for any future reference.


[ITU-R BS.2217]: https://www.itu.int/pub/R-REP-BS.2217
[ITU-R BS.1770]: https://www.itu.int/rec/R-REC-BS.1770

# lupy - Validated Commands

## Install
```bash
uv sync --frozen --no-dev --group test
# or without uv:
pip install -e ".[test]" --break-system-packages
```

## Test
```bash
uv run py.test tests/
# Without uv, without benchmarks:
python3 -m pytest tests/ -k "not benchmark" --override-ini="addopts=" -p no:codspeed
```

## Test with Coverage
```bash
uv run py.test tests/ --cov=src/ --cov-report=xml
# Without uv:
python3 -m pytest tests/ -k "not benchmark" --override-ini="addopts=" -p no:codspeed --cov=src/ --cov-report=term-missing
```

## Type Check
```bash
uv run mypy
```

## Notes
- pyproject.toml addopts: `-n auto --dist=worksteal --doctest-modules`
- pytest-codspeed needed for benchmarks; skip with `-k "not benchmark" -p no:codspeed`
- Coverage uploaded to Codecov via CI

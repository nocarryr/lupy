# Agents

## Installation and Tooling (Agents)

The project uses [uv](https://docs.astral.sh/uv/) for installation and dependency management.

If `uv` is not installed, you can install it using curl (preferred method):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If unavailable, you can also install it using pipx:

```bash
pipx install uv
```

Or using pip:

```bash
pip install --user uv
```


Dependencies are deterministic since they are pinned in `uv.lock`. Always use the `--frozen` flag when installing to ensure that the exact versions specified in `uv.lock` are installed:

```bash
uv sync --frozen
```


## Quick Start

To quickly verify your environment is set up correctly, run:
```bash
uv sync --frozen && uv run pytest -m "not slow" && uv run mypy
```

This ensures dependencies are synced, tests pass, and types are valid in a single step.


## Running Tests

Tests are run using `pytest`. The default test configuration is defined in `pyproject.toml` under the `[tool.pytest.ini_options]` section.

Note that the `--cov` option is not included by default, so modify the
test commands accordingly if you want to include code coverage reporting.

The test paths include both `src/` and `tests/` to enable doctest execution in the source code.


### Test Markers

The `slow` marker is used to indicate tests that are slow to run.
By default, tests marked as `slow` are included in the test suite. To exclude them, use the `-m` flag with `pytest`:

```bash
uv run pytest -m "not slow"
```

This can be used for a quicker test run during development, but the full test suite **must** pass before merging any code changes.


To run all tests, including those marked as `slow`, simply run:

```bash
uv run pytest
```

### Benchmarking

This section should only be relevant for agents that are working on performance-related features or optimizations. If you are not working on performance-related features, you can skip this section.


Benchmarking is done using [pytest-codspeed](https://codspeed.io/docs/reference/pytest-codspeed). To run benchmarks, use the `--codspeed` flag:

```bash
uv run pytest --codspeed
```

After running benchmarks, the results will be available in the `.codspeed/` directory as JSON files. The filename format is `results_<timestamp>.json`.

An example of the structure of the benchmark results JSON file is as follows:

```json
{
  "creator": { ... },
  "python": { ... },
  "instrument": { ... },
  "benchmarks": [
    {
      "name": "test_benchmark_example[param1]",
      "uri": "tests/test_filename.py::test_benchmark_example[param1]",
      "stats": {
        ...
      }
    },
    ...
  ]
}
```

The `benchmarks.stats` field contains various statistics about the benchmark results which can be useful for analysis.


### Type Checking

Type checking is done using `mypy`. To run type checks, use the following command:

```bash
uv run mypy
```

This should **always** pass without any errors or warnings. If you encounter any type errors, please fix them before committing your code.


## Typing Preferences

- Use strict typing as much as possible.
- Always annotate parameters and return types for functions and methods.
- Avoid using type casts unless specifically requested.
- **Never** use `# type: ignore` comments or other mechanisms to suppress type checking errors.
- Follow the existing typing style and conventions used in the project as closely as possible.


## Code Style Preferences

- Follow the existing code style and conventions used in the project as closely as possible, falling back to PEP 8 if there are no clear conventions in the existing code.
- Avoid creating code comments that explain what the code does unless it is strictly necessary for understanding non-obvious logic.
- When splitting long lines for method and function signatures, use an open parenthesis on the first line and align subsequent lines with an additional indentation level.  This also applies to function calls. For example:

```python
def example_function(
    arg1: int,
    arg2: str,
    arg3: Optional[float] = None,
) -> None:
    ...
```


## Python Docstring Style

- The first line of a docstring should be a short, concise summary of the function, method, or class's purpose. It should be written in the imperative mood and fit on a single line if possible, followed by a blank line before any further elaboration or parameter descriptions. No period should be used at the end of this first summary line.
- Keep line lengths to a maximum of 85 characters.
- Do not over-explain in docstrings; provide enough detail to clarify non-obvious behavior but avoid restating what is already clear from the function, method, or class signature.
- Assume that Sphinx is used to generate documentation, so prefer using Sphinx-compatible cross-references and markup in docstrings, such as :class:, :meth:, :attr:, and :term: roles where appropriate.
- Use "Google style" docstring formatting for parameters, return values, and exceptions.
- Prefer to document attributes alongside their declaration in the class body rather than in the class docstring.
- Docstrings for modules, classes, methods, functions and properties should have the closing triple quotes on a line by themselves.
- Docstrings for attributes may have the closing triple quotes on the same line if the body is short and fits comfortably on a single line.


Example of a well-formatted docstring:

```python

class ExampleClass:
    """An example class to demonstrate docstring formatting

    This class serves as an example for how to format docstrings in this project.
    This section is used to provide a description of the class and its purpose.
    It can be multiple lines long if necessary, but should not be overly verbose.

    """

    attribute_a: int
    """An example attribute of the class"""

    attribute_b: str
    """Another example attribute of the class

    This attribute is used to demonstrate an attribute that needs a
    longer description.
    """

    def example_method(self, param1: int, param2: str) -> bool:
        """Demonstrate docstring formatting for a method

        This method serves as an example for how to format docstrings for
        methods in this project.
        The first line is a concise summary of the method's purpose.
        The following lines provide additional details about the method's
        behavior and parameters.

        Paragraphs may be used to separate different sections of the docstring
        for clarity.

        .. tip::

            This can be used to provide additional tips or notes about the
            method that may be helpful for users or developers.

        Arguments:
            param1: An integer parameter that serves as an example.
            param2: A string parameter that serves as another example.

        Returns:
            A boolean value indicating the result of the method's operation.

        Raises:
            ValueError: If the input parameters do not meet certain criteria.
        """
        ...

```

# SSIP

See the [documentation](https://cqcl.github.io/SSIP/api-docs/) for a more thorough description.

## Development

### Prerequisites

- Python >= 3.10
- [Poetry](https://python-poetry.org/docs/#installation)
- [GAP](https://www.gap-system.org)


### Installing

Run the following to setup your virtual environment and install dependencies:

```sh
poetry install
```


You can then activate the virtual environment and work within it with:

```sh
poetry shell
```

Consider using [direnv](https://github.com/direnv/direnv/wiki/Python#poetry) to
automate this when entering and leaving a directory.

To run a single command in the shell, just prefix it with `poetry run`.

Run tests using

```sh
poetry run pytest -v
```

Run tests with code coverage using

```sh
poetry run pytest --cov-report term-missing --cov=ssip tests/
```
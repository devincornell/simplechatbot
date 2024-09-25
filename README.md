# simplechatbot
Basic chatbot built on langchain.

## Installation

`pip install git+ssh://git@github.com/devincornell/simplechatbot.git@main`

When inside the package directory: Basic install: `pip install .`

This package uses buildtools - see `pyproject.toml` for package details.

You can also use `make`.

To install: `make install`

To uninstall: `make uninstall`

## Importing

Basic importing works as you would expect.

`import simplechatbot`

To support backwards compatibility, I also keep old versions in the main module titled `vN` where `N` is the version number. Yeah, it's as sketch as you'd expect.

`import simplechatbot.v4 as simplechatbot`

But it works!


# `simplechatbot` Python Package

For full documentation visit [mkdocs.org](https://www.mkdocs.org).

## Installation

`pip install git+ssh://git@github.com/devincornell/simplechatbot.git@main`

When inside the package directory: Basic install: `pip install .`

This package uses buildtools - see `pyproject.toml` for package details.

### Makefile

You can also use `make`.

To install: `make install`

To uninstall: `make uninstall`

## Importing

Basic importing works as you would expect.

`import simplechatbot`

To support backwards compatibility, I also keep old versions in the main module titled `vN` where `N` is the version number. You can change the imports to look like the following.

`import simplechatbot.v4 as simplechatbot`

But it works!

## Generating Documentation

The Makefile has most of these commands, but including them here jsut in case.

```
	pip install mkdocs
	pip install mkdocs-material
```

[Start Test Server](https://squidfunk.github.io/mkdocs-material/creating-your-site/)

``
`mkdocs serve
```

Build the documentation.

```
	mkdocs build
```

[Publish the documation](https://squidfunk.github.io/mkdocs-material/publishing-your-site/)

```
	mkdocs gh-deploy --force
```


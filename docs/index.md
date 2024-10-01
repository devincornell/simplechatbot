# Introduction

Welcome to the simplechatbot Python package!

## Installation

```
pip install git+ssh://git@github.com/devincornell/simplechatbot.git@main
```

When inside the package directory: Basic install: 

```pip install .```

This package uses buildtools - see `pyproject.toml` for package details.

### Makefile

You can also use `make`.

To install: 

```make install```

To uninstall: 

```make uninstall```

## Importing

Basic importing works as you would expect.

```import simplechatbot```

To support backwards compatibility, I also keep old versions in the main module titled `vN` where `N` is the version number. You can change the imports to look like the following.

```import simplechatbot.v4 as simplechatbot```


## Generating Documentation

The Makefile has most of these commands, but including them here jsut in case.

```
pip install mkdocs
pip install mkdocs-material
```

[Start Test Server](https://squidfunk.github.io/mkdocs-material/creating-your-site/)

```
mkdocs serve
```

Build the documentation.

```
mkdocs build
```

[Publish the documation](https://squidfunk.github.io/mkdocs-material/publishing-your-site/)

```
mkdocs gh-deploy --force
```

### Example Documentation

In the Makefile I included the commands that will take example jupyter notebooks and convert them to markdown so that `mkdocs` can eventually convert them to html for the website. Simply add a notebook to the `site_examples` folder and it will be automatically converted to markdown and placed in the right folder.

```
EXAMPLE_NOTEBOOK_FOLDER = ./site_examples/# this is where example notebooks are stored
EXAMPLE_NOTEBOOK_MARKDOWN_FOLDER = ./docs/examples/# this is where example notebooks are stored

example_notebooks:
	-mkdir $(EXAMPLE_NOTEBOOK_MARKDOWN_FOLDER)
	jupyter nbconvert --to markdown $(EXAMPLE_NOTEBOOK_FOLDER)/*.ipynb
	mv $(EXAMPLE_NOTEBOOK_FOLDER)/*.md $(EXAMPLE_NOTEBOOK_MARKDOWN_FOLDER)
```
PACKAGE_NAME = simplechatbot


install:
	pip install .

uninstall:
	pip uninstall simplechatbot




############### mkdocs-material documentation generator ################
# https://squidfunk.github.io/mkdocs-material/

# https://squidfunk.github.io/mkdocs-material/getting-started/
install_mkdocs:
	pip install mkdocs
	pip install mkdocs-material

# https://squidfunk.github.io/mkdocs-material/creating-your-site/
serve_mkdocs:
	mkdocs serve

build_mkdocs:
	mkdocs build

# https://squidfunk.github.io/mkdocs-material/publishing-your-site/
deploy_mkdocs:
	mkdocs gh-deploy --force


######################## Jupyter notebook to markdown #####################
# these jupyter notebooks will be converted to markdown files for mkdocs to 
# render them as html

EXAMPLE_NOTEBOOK_FOLDER = ./site_examples/# this is where example notebooks are stored
EXAMPLE_NOTEBOOK_MARKDOWN_FOLDER = ./docs/examples/# this is where example notebooks are stored

example_notebooks:
	-mkdir $(EXAMPLE_NOTEBOOK_MARKDOWN_FOLDER)
	jupyter nbconvert --to markdown $(EXAMPLE_NOTEBOOK_FOLDER)/*.ipynb
	mv $(EXAMPLE_NOTEBOOK_FOLDER)/*.md $(EXAMPLE_NOTEBOOK_MARKDOWN_FOLDER)

	

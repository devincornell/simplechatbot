PACKAGE_NAME = simplechatbot


reinstall: uninstall install

install: clean
	pip install .

uninstall:
	pip uninstall simplechatbot

clean:
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info


############### Doc Update Instructions ################
# if you want to update the documentation in the local docs, you can do these commands

# 1. make example_notebooks: will compile notebooks into markdown files and place them in "docs/examples"
# 2. (OPTIONAL) make serve_mkdocs: will serve the mkdocs documentation locally to allow you to see the new files
# 3. make build_mkdocs: will build the mkdocs documentation (markdown -> html)
# 4. make deploy_mkdocs: will deploy the mkdocs documentation to the gh-pages branch. publishes to the live site!


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

	

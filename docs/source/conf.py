# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import tomllib  # please change this if we don't use poetry anymore

# Load the TOML file
with open("../../pyproject.toml", "rb") as file:
    data = tomllib.load(file)

# Access the version key
version = data["tool"]["poetry"]["version"]


project = "createVTLcorpus"
copyright = "2024, Konstantin Sering and Valentin Schmidt"
author = "Konstantin Sering and Valentin Schmidt"
release = version

# add modules to the path
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinxarg.ext",
    "sphinx.ext.autosectionlabel",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/quantling/create_vtl_corpus",  # required
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        }
    ]
}
html_static_path = ["_static"]
html_sidebars = {"**": []}

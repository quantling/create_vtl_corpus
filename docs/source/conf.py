# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "createVTLcorpus"
copyright = "2024, Konstantin Sering and Valentin Schmidt"
author = "Konstantin Sering and Valentin Schmidt"
release = "0.1.0"

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

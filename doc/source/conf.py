# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'lupy'
copyright = '2024, Matthew Reid'
author = 'Matthew Reid'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.extlinks',
]


autodoc_typehints = 'both'
autodoc_typehints_description_target = 'documented'
autodoc_docstring_signature = True

autodoc_default_options = {
    'member-order': 'bysource',
    'show-inheritance': True,
    'special-members': '__call__',
    'ignore-module-all': True,
}

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

intersphinx_mapping = {
    'python':('https://docs.python.org/', None),
    'numpy':('https://numpy.org/doc/stable/', None),
    'scipy':('https://docs.scipy.org/doc/scipy-1.13.1/', None),
}

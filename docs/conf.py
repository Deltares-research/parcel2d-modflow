# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import parcel2d_modflow

# sys.path.insert(0, os.path.abspath(".."))  # isort:skip

# -- Project information -----------------------------------------------------

project = "Parcel2D-Modflow"
copyright = "2026, Deltares"
author = "Deltares"

# The full version, including alpha/beta/rc tags


version = parcel2d_modflow.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "pydata_sphinx_theme",
    "sphinx.ext.napoleon",
    "sphinx_design",
    "myst_nb",
]


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "*bro"]

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "pydata_sphinx_theme"


# # Show members (and docstrings)
# autodoc_default_options = {
#     "members": True,
#     "undoc-members": True,
#     "show-inheritance": True,
# }


# intersphinx-links to external projects
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pathlib": ("https://pathlib.readthedocs.io/en/latest/", None),
    "geopandas": ("https://geopandas.org/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

html_static_path = ["_static"]
html_css_files = ["theme-deltares.css"]
html_theme_options = {
    "show_toc_level": 1,
    "show_nav_level": 1,
    "navbar_align": "left",
    "use_edit_page_button": False,
    "header_links_before_dropdown": 6,
    "pygments_light_style": "tango",
    "pygments_dark_style": "one-dark",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Deltares-research/parcel2d-modflow",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
        # {
        #     "name": "PyPI",
        #     "url": "https://pypi.org/project/lusos",
        #     "icon": "fa-solid fa-cubes",
        #     "type": "fontawesome",
        # },
        {
            "name": "Deltares",
            "url": "https://deltares.nl/en/",
            "icon": "_static/deltares-blue.svg",
            "type": "local",
        },
    ],
    "logo": {
        "text": "Parcel2D-Modflow",
        "image_light": "_static/logo.svg",
        "image_dark": "_static/logo.svg",
    },
}

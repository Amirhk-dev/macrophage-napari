"""Sphinx configuration for the napari-macrophage documentation."""

from __future__ import annotations

import os
import sys
from importlib.metadata import version as _pkg_version

# Make the package importable for autodoc without needing an editable install.
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "napari-macrophage"
author = "Amirhossein Kardoost"
copyright = f"2026, {author}"

try:
    release = _pkg_version("napari-macrophage")
except Exception:
    release = "0.0.0"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Accept both .rst and .md sources.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_static_path = ["_static"]
html_title = f"napari-macrophage {version}"

# -- autodoc / autosummary ---------------------------------------------------

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_class_signature = "separated"


# Accept both Google-style (pre-existing macrophage_mesh / visualize_3d) and
# NumPy-style (newer modules) docstrings.
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
# Render class Attributes sections inline with :ivar: to avoid duplicate
# descriptions when autosummary also lists them.
napoleon_use_ivar = True

# Import-time side effects to avoid during doc builds: napari/qtpy need a Qt
# display. Mock them so autodoc can introspect the plugin without a runtime.
autodoc_mock_imports = [
    "napari",
    "napari.layers",
    "napari.utils",
    "napari.utils.notifications",
    "napari._qt",
    "qtpy",
    "magicgui",
    "tifffile",
    "zarr",
    "onnxruntime",
    "torch",
]

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "skimage": ("https://scikit-image.org/docs/stable/", None),
    "napari": ("https://napari.org/stable/", None),
}

# -- MyST --------------------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "smartquotes",
]

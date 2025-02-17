# Configuration file for the Sphinx documentation builder.
# See https://www.sphinx-doc.org/en/master/usage/configuration.html


project = "SSIP"
copyright = "2024, Quantinuum"
author = "Quantinuum"

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

html_theme = "sphinx_book_theme"

html_title = "SSIP python package API documentation."

html_theme_options = {
    "repository_url": "https://github.com/alexcowtan/SSIP",
    "use_repository_button": True,
    "navigation_with_keys": True,
    "logo": {
        "image_light": "_static/Quantinuum_logo_black.png",
        "image_dark": "_static/Quantinuum_logo_white.png",
    },
}

html_static_path = ["../_static"]
html_css_files = ["custom.css"]

autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
}

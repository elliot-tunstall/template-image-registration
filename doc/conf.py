# dipy documentation build configuration file, created by
# sphinx-quickstart on Thu Feb  4 15:23:20 2010.
#
# This file is execfile()d with the current directory set to its containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import os
import re
import sys
import json

# Doc generation depends on being able to import dipy
try:
    import dipy
except ImportError:
    raise RuntimeError('Cannot import dipy, please investigate')

from packaging.version import Version
import sphinx
if Version(sphinx.__version__) < Version('2'):
    raise RuntimeError('Need sphinx >= 2 for numpydoc to work correctly')

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.append(os.path.abspath('sphinxext'))

# -- General configuration -----------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.doctest',
              'sphinx.ext.intersphinx',
              'sphinx.ext.todo',
              'sphinx.ext.coverage',
              'sphinx.ext.mathjax',
              'sphinx.ext.ifconfig',
              'sphinx.ext.autosummary',
              'prepare_gallery',
              'math_dollar',  # has to go before numpydoc
              'sphinx_gallery.gen_gallery',
              # 'numpydoc',
              'github',
              'sphinx_design',
]

numpydoc_show_class_members = True
numpydoc_class_members_toctree = False

# ghissue config
github_project_url = "https://github.com/dipy/dipy"

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The encoding of source files.
#source_encoding = 'utf-8'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'dipy'
copyright = "Copyright 2008-2023, DIPY developers. Created using Grg Sphinx Theme and PyData Sphinx Theme."

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = dipy.__version__
# The full version, including alpha/beta/rc tags.
release = version

# Include common links
# We don't use this any more because it causes conflicts with the gitwash docs
#rst_epilog = open('links_names.inc', 'rt').read()

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
#today_fmt = '%B %d, %Y'

# List of documents that shouldn't be included in the build.
#unused_docs = []

# List of directories, relative to source directory, that shouldn't be searched
# for source files.
exclude_patterns = ['_build', 'examples', 'examples_revamped']

# The reST default role (used for this markup: `text`) to use for all documents.
#default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
#add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
#modindex_common_prefix = []


# -- Options for HTML output ---------------------------------------------------

# The theme to use for HTML and HTML Help pages.  Major themes that come with
# Sphinx are currently 'default' and 'sphinxdoc'.
html_theme = "grg_sphinx_theme"

# The style sheet to use for HTML and HTML Help pages. A file of that name
# must exist either in Sphinx' static/ path, or in one of the custom paths
# given in html_static_path.
html_style = 'css/dipy.css'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
  "switcher": {"json_url": "https://docs.dipy.org/dev/_static/version_switcher.json",
               "version_match": "1.8.0", },
  "check_switcher": False,
  "show_version_warning_banner": True,
  "navbar_end": ["search-field.html", "version-switcher", "navbar-icon-links.html"],
  "secondary_sidebar_items": ["page-toc"],
  "show_toc_level": 1,
  "navbar_center": ["components/navbar-links.html"],
  "navbar_links": [
     {
        "name": "Docs",
        "children": [
          {
            "name": "Overview",
            "url": "index",
          },
          {
            "name": "Tutorials",
            "url": "examples_built/index",
          },
          {
            "name": "Recipes",
            "url": "recipes/index",
          },
          {
            "name": "CLI / Workflows",
            "url": "interfaces/index",
          },
          {
            "name": "API",
            "url": "reference/index",
          },
          {
            "name": "CLI API",
            "url": "reference_cmd/index",
          }
        ]
     },
     {
        "name": "Workshops",
        "sections": [
          {
            "name": "Latest",
            "children": [
              {
                "name": "DIPY Workshop 2024",
                "url": "https://dipy.org/workshops/dipy-workshop-2024",
                "link_type": "external"
              }
            ]
          },
          {
            "name": "Past",
            "children": [
              {
                "name": "DIPY Workshop 2023",
                "url": "https://dipy.org/workshops/dipy-workshop-2023",
                "link_type": "external"
              },
              {
                "name": "DIPY Workshop 2022",
                "url": "https://dipy.org/workshops/dipy-workshop-2022",
                "link_type": "external"
              },
              {
                "name": "DIPY Workshop 2021",
                "url": "https://dipy.org/workshops/dipy-workshop-2021",
                "link_type": "external"
              },
              {
                "name": "DIPY Workshop 2020",
                "url": "https://dipy.org/workshops/dipy-workshop-2020",
                "link_type": "external"
              },
              {
                "name": "DIPY Workshop 2019",
                "url": "https://dipy.org/workshops/dipy-workshop-2019",
                "link_type": "external"
              },
            ]
          }
        ],
     },
     {
        "name": "Community",
        "sections": [
            {
              "name": "News",
              "children": [
                  {
                    "name": "Calendar",
                    "url": "https://dipy.org/calendar",
                    "link_type": "inter"
                  },
                  {
                    "name": "Newsletters",
                    "url": "https://mail.python.org/mailman3/lists/dipy.python.org/",
                    "link_type": "external"
                  },
                  {
                    "name": "Blog",
                    "url": "https://dipy.org/blog",
                    "link_type": "inter"
                  },
                  {
                    "name": "Youtube",
                    "url": "https://www.youtube.com/c/diffusionimaginginpython",
                    "link_type": "external"
                  }
              ]
            },
            {
              "name": "Help",
              "children": [
                  {
                    "name": "Live Chat (Gitter)",
                    "url": "https://app.gitter.im/#/room/%23dipy_dipy:gitter.im",
                    "link_type": "external"
                  },
                  {
                    "name": "Github Discussions",
                    "url": "https://github.com/dipy/dipy/discussions",
                    "link_type": "external"
                  }
              ]
            }
          ]
     },
     {
        "name": "About",
        "children": [
          {
            "name": "Team",
            "url": "https://dipy.org/team",
            "link_type": "inter"
          },
          {
            "name": "FAQ",
            "url": "faq",
          },
          {
            "name": "Mission Statement",
            "url": "user_guide/mission",
          },
          {
            "name": "Releases",
            "url": "stateoftheart",
          },
          {
            "name": "Cite",
            "url": "cite",
          },
          {
            "name": "Glossary",
            "url": "glossary",
          },
        ]
     },
  ],
  # To remove search icon
  "navbar_persistent": "",
  "icon_links": [
    {
      "name": "GitHub",
      "url": "https://github.com/dipy",
      "icon": "fa-brands fa-github"
    },
    {
      "name": "Twitter/X",
      "url": "https://twitter.com/dipymri",
      "icon": "fa-brands fa-twitter"
    },
    {
      "name": "YouTube",
      "url": "https://www.youtube.com/c/diffusionimaginginpython",
      "icon": "fa-brands fa-youtube"
    },
    {
      "name": "LinkedIn",
      "url": "https://www.linkedin.com/company/dipy/",
      "icon": "fa-brands fa-linkedin"
    },
  ],
  "logo": {
    "image_dark": "_static/images/logos/dipy-logo.png",
    "alt_text": "DIPY",
  },
  "footer_start": ["components/footer-sign-up.html"],
  "footer_signup_data": {
    "heading": "Never miss an update from us!",
    "sub_heading": "Don't worry! we are not going to spam you."
  },
  "footer_end": ["components/footer-sections.html"],
  "footer_links": [
    {
      "title": "About",
      "links": [
        {
          "name": "Developers",
          "link": "https://dipy.org/team",
          "link_type": "inter"
        },
        {
          "name": "Support",
          "link": "https://github.com/dipy/dipy/discussions",
          "link_type": "external"
        },
        {
          "name": "Download",
          "link": "user_guide/installation"
        },
        {
          "name": "Get Started",
          "link": "user_guide/getting_started"
        },
        {
          "name": "Tutorials",
          "link": "examples_built/index"
        },
        {
          "name": "Videos",
          "link": "https://www.youtube.com/c/diffusionimaginginpython",
          "link_type": "external"
        },
      ]
    }, {
      "title": "Friends",
      "links": [
        {
          "name": "Nipy Projects",
          "link": "http://nipy.org/",
          "link_type": "external"
        },
        {
          "name": "FURY",
          "link": "http://fury.gl/",
          "link_type": "external"
        },
        {
          "name": "Nibabel",
          "link": "http://nipy.org/nibabel",
          "link_type": "external"
        },
        {
          "name": "Tortoise",
          "link": "https://tortoise.nibib.nih.gov/",
          "link_type": "external"
        },
      ]
    }, {
      "title": "Support",
      "links": [
        {
          "name": "The department of Intelligent Systems Engineering of Indiana University",
          "link": "https://engineering.indiana.edu/",
          "link_type": "external"
        },
        {
          "name": "The National Institute of Biomedical Imaging and Bioengineering, NIH",
          "link": "https://www.nibib.nih.gov/",
          "link_type": "external"
        },
        {
          "name": "The Gordon and Betty Moore Foundation and the Alfred P. Sloan Foundation, through the University of Washington eScience Institute Data Science Environment",
          "link": "https://escience.washington.edu",
          "link_type": "external"
        },
        {
          "name": "Google supported DIPY through the Google Summer of Code Program during Summer 2015, 2016, 2018",
          "link": "https://summerofcode.withgoogle.com/",
          "link_type": "external"
        },
      ]
    }
  ],
  "footer_copyright": copyright,
}

# Add any paths that contain custom themes here, relative to this directory.
#html_theme_path = []

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
#html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
#html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "_static/images/logos/dipy-logo.png"

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "_static/images/logos/dipy-favicon.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
#html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
#html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
html_sidebars = {'index': []}

# Additional templates that should be rendered to pages, maps page names to
# template names.
#html_additional_pages = {}

# If false, no module index is generated.
# Setting to false fixes double module listing under header
html_use_modindex = False

# If false, no index is generated.
#html_use_index = True

# If true, the index is split into individual pages for each letter.
#html_split_index = False

# If true, links to the reST sources are added to the pages.
#html_show_sourcelink = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
#html_use_opensearch = ''

# If nonempty, this is the file name suffix for HTML files (e.g. ".xhtml").
#html_file_suffix = ''

# Output file base name for HTML help builder.
htmlhelp_basename = 'dipydoc'


# -- Options for LaTeX output --------------------------------------------------

# The paper size ('letter' or 'a4').
#latex_paper_size = 'letter'

# The font size ('10pt', '11pt' or '12pt').
#latex_font_size = '10pt'

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
  ('index', 'dipy.tex', 'dipy Documentation',
   'Eleftherios Garyfallidis, Ian Nimmo-Smith, Matthew Brett', 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#latex_use_parts = False

# Additional stuff for the LaTeX preamble.
latex_preamble = r"""
\usepackage{amsfonts}
"""

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
#latex_use_modindex = True


# -- Options for sphinx gallery -------------------------------------------
from docimage_scrap import ImageFileScraper
from sphinx_gallery.sorting import ExplicitOrder
from prepare_gallery import folder_explicit_order

sc = ImageFileScraper()
ordered_folders = [f'examples_revamped/{f}' for f in folder_explicit_order()]

sphinx_gallery_conf = {
     'doc_module': ('dipy',),
     # path to your examples scripts
     'examples_dirs': ['examples_revamped', ],
     # path where to save gallery generated examples
     'gallery_dirs': ['examples_built', ],
     'subsection_order': ExplicitOrder(ordered_folders),
     'image_scrapers': (sc),
     'backreferences_dir': 'examples_built',
     'reference_url': {'dipy': None, },
     'abort_on_example_error': False,
     'filename_pattern': re.escape(os.sep),
     'default_thumb_file': "_static/images/logos/dipy_full_logo.png",
    #  'pypandoc': {'extra_args': ['--mathjax',]},
}

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {'python': ('https://docs.python.org/3/', None)}


# Retrieve html_theme_options from docs/conf.py
from docs.conf import html_theme_options

html_theme_options["switcher"][
    "json_url"
] = "https://lets-unify.ai/docs/versions/memory.json"

repo_name = "memory"

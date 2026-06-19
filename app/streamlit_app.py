"""World Cup Match Predictor — multipage Streamlit dashboard.

Run from the project root with:

    streamlit run app/streamlit_app.py

The page render functions live in ``pages_views.py``; this file only configures
the page and wires up navigation.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure this app/ directory is importable (covers `streamlit run` and AppTest).
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st

import pages_views as views

st.set_page_config(page_title="World Cup Match Predictor", page_icon="⚽", layout="wide")

pages = [
    st.Page(fn, title=title, icon=icon, default=default)
    for fn, title, icon, default in views.ALL_PAGES
]
st.navigation(pages).run()

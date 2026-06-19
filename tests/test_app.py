"""Smoke tests for the Streamlit dashboard pages.

Each page is executed in Streamlit's AppTest harness via a small bootstrap
script; the test passes if the page renders without raising.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import config  # noqa: E402

APP_DIR = Path(config.PROJECT_ROOT) / "app"
sys.path.insert(0, str(APP_DIR))
import pages_views as views  # noqa: E402

PAGE_NAMES = [fn.__name__ for fn, _, _, _ in views.ALL_PAGES]


def _bootstrap(page_name: str) -> str:
    """A tiny script that imports a page and runs it inside AppTest."""
    return (
        "import sys\n"
        f"sys.path.insert(0, r'{APP_DIR}')\n"
        "import pages_views\n"
        f"pages_views.{page_name}()\n"
    )


@pytest.mark.parametrize("page_name", PAGE_NAMES)
def test_page_renders_without_exception(page_name):
    at = AppTest.from_string(_bootstrap(page_name), default_timeout=240).run()
    assert not at.exception, f"Page '{page_name}' raised: {at.exception}"

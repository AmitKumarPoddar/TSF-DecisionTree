import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests import mock_tesseract  # noqa: E402


@pytest.fixture(scope="session", params=["rs", "olap"])
def mock_api(request):
    """Base URL of a running mock Tesseract server, once per API generation."""
    server, base_url = mock_tesseract.serve(0, request.param)
    yield base_url
    server.shutdown()

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")


@pytest.fixture(scope="session")
def release(tmp_path_factory):
    """Two tiny random-weight models in the same format as a real release."""
    from package_models import build_dummy_release
    return build_dummy_release(tmp_path_factory.mktemp("release"))


@pytest.fixture(scope="session")
def registry(release):
    from src.inference import Registry
    return Registry(release)

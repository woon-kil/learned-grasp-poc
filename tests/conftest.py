from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from learned_grasp.config import ROOT, load_config
from learned_grasp.pose_table import generate


@pytest.fixture(scope="session")
def draft_config():
    return load_config(Path(ROOT) / "config" / "workspace.yaml")


@pytest.fixture(scope="session")
def table(draft_config):
    return generate(draft_config.table)


@pytest.fixture(scope="session")
def ready_config(draft_config):
    """테스트 전용 값. 실기 값이 아니다."""
    return replace(draft_config, park=(90, 120, 30, 30, 90), open_sdk=100, close_sdk=150, verified_g0a=True)

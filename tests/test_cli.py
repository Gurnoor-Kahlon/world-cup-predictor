"""Tests for small CLI helpers (kept fast — no model training)."""

import pytest

import config
from main import resolve_stage


def test_resolve_stage_is_case_insensitive():
    assert resolve_stage("final") == "Final"
    assert resolve_stage("FINAL") == "Final"
    assert resolve_stage("  group ") == "Group"
    assert resolve_stage("quarter-final") == "Quarter-final"


def test_resolve_stage_passes_through_canonical():
    for stage in config.TOURNAMENT_STAGES:
        assert resolve_stage(stage) == stage


def test_resolve_stage_rejects_unknown():
    with pytest.raises(ValueError):
        resolve_stage("kickabout")

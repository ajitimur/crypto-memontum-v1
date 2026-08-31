"""Seam 5: the trials log. Append-only, machine-written, with no delete path."""

import ast
import json
from pathlib import Path

import pytest

import crypto_momentum.trials as trials_module
from crypto_momentum.trials import UnrecordableTrial, append_trial, read_trials


def test_the_first_trial_creates_the_log(tmp_path):
    log = tmp_path / "trials.jsonl"

    append_trial(log, {"config_name": "skeleton", "net_return": 0.1})

    assert json.loads(log.read_text().strip()) == {
        "config_name": "skeleton",
        "net_return": 0.1,
    }


def test_a_later_trial_never_disturbs_an_earlier_one(tmp_path):
    log = tmp_path / "trials.jsonl"
    append_trial(log, {"config_name": "first"})
    append_trial(log, {"config_name": "second"})
    append_trial(log, {"config_name": "third"})

    lines = log.read_text().splitlines()

    assert [json.loads(line)["config_name"] for line in lines] == ["first", "second", "third"]


def test_every_line_stands_alone_as_json(tmp_path):
    log = tmp_path / "trials.jsonl"
    append_trial(log, {"config_name": "with\nnewline", "nested": {"a": [1, 2]}})
    append_trial(log, {"config_name": "second"})

    assert len(log.read_text().splitlines()) == 2
    assert read_trials(log)[0]["config_name"] == "with\nnewline"


def test_the_count_of_configurations_tried_is_the_line_count(tmp_path):
    """`docs/agents/quant-research.md` requires reporting how many configurations
    were tried, which is exactly what this log is for."""
    log = tmp_path / "trials.jsonl"
    for index in range(5):
        append_trial(log, {"config_name": f"trial-{index}"})

    assert len(read_trials(log)) == 5


def test_a_trial_that_cannot_be_serialised_is_rejected_before_anything_is_written(tmp_path):
    log = tmp_path / "trials.jsonl"
    append_trial(log, {"config_name": "good"})

    with pytest.raises(UnrecordableTrial):
        append_trial(log, {"config_name": "bad", "frame": object()})

    assert len(log.read_text().splitlines()) == 1


def test_the_log_module_offers_no_way_to_delete_or_rewrite_a_trial():
    """Append-only is a property of the interface, not of the caller's discipline."""
    tree = ast.parse(Path(trials_module.__file__).read_text())

    modes = {
        keyword.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant)
    }
    assert modes <= {"a", "r"}

    public_names = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    }
    assert public_names == {"append_trial", "read_trials"}

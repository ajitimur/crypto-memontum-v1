"""Every result traces to a commit hash plus a config file."""

import json
import subprocess

import pytest

from crypto_momentum.config import RunConfig
from crypto_momentum.costs import TOKOCRYPTO
from crypto_momentum.provenance import NotAGitRepository, describe_head
from crypto_momentum.results import ResultStore, RunRecord

CONFIG = RunConfig(
    name="skeleton-btcusdt-2021h1",
    venue="binance-spot",
    symbol="BTCUSDT",
    interval="1d",
    start_month="2021-01",
    end_month="2021-02",
    strategy_kind="buy_and_hold",
    cost_model=TOKOCRYPTO,
    slippage_bps_per_side=5.0,
)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    run = lambda *args: subprocess.run(args, cwd=root, check=True, capture_output=True)
    run("git", "init", "-q")
    run("git", "config", "user.email", "test@example.com")
    run("git", "config", "user.name", "Test")
    (root / "tracked.txt").write_text("original\n")
    run("git", "add", ".")
    run("git", "commit", "-qm", "first")
    return root


def record_for(commit: str, dirty: bool, config: RunConfig = CONFIG) -> RunRecord:
    return RunRecord(
        commit=commit,
        working_tree_dirty=dirty,
        run_at_utc="2026-08-31T00:00:00Z",
        config=config,
        config_sha256="0" * 64,
        config_path="configs/skeleton.toml",
        metrics={"net_return": 0.25},
        window={"start_month": config.start_month, "end_month": config.end_month},
    )


def test_head_is_reported_as_a_full_commit_hash(repo):
    provenance = describe_head(repo)

    expected = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    assert provenance.commit == expected
    assert len(provenance.commit) == 40
    assert provenance.working_tree_dirty is False


def test_an_edited_tracked_file_makes_the_working_tree_dirty(repo):
    (repo / "tracked.txt").write_text("edited\n")

    assert describe_head(repo).working_tree_dirty is True


def test_an_untracked_file_makes_the_working_tree_dirty(repo):
    """An untracked module can change a result, so it counts."""
    (repo / "extra.py").write_text("x = 1\n")

    assert describe_head(repo).working_tree_dirty is True


def test_running_outside_a_repository_is_refused(tmp_path):
    outside = tmp_path / "not-a-repo"
    outside.mkdir()

    with pytest.raises(NotAGitRepository):
        describe_head(outside)


def test_a_result_is_keyed_by_commit_hash_and_config_name(tmp_path):
    store = ResultStore(tmp_path / "results")

    path = store.write(record_for("a" * 40, dirty=False))

    assert path.relative_to(store.root).as_posix() == (
        f"{'a' * 40}/skeleton-btcusdt-2021h1.json"
    )
    assert json.loads(path.read_text())["metrics"]["net_return"] == 0.25


def test_two_configs_at_one_commit_do_not_collide(tmp_path):
    store = ResultStore(tmp_path / "results")
    other = RunConfig(**{**CONFIG.__dict__, "name": "skeleton-btcusdt-2021-january"})

    first = store.write(record_for("a" * 40, dirty=False))
    second = store.write(record_for("a" * 40, dirty=False, config=other))

    assert first != second
    assert first.parent == second.parent


def test_one_config_at_two_commits_does_not_collide(tmp_path):
    store = ResultStore(tmp_path / "results")

    first = store.write(record_for("a" * 40, dirty=False))
    second = store.write(record_for("b" * 40, dirty=False))

    assert first.parent != second.parent


def test_a_dirty_run_is_never_filed_as_a_clean_one(tmp_path):
    """A dirty tree means the commit hash does not identify the code that ran."""
    store = ResultStore(tmp_path / "results")

    clean = store.write(record_for("a" * 40, dirty=False))
    dirty = store.write(record_for("a" * 40, dirty=True))

    assert clean != dirty
    assert dirty.parent.name.endswith("-dirty")


def test_the_record_carries_the_config_that_produced_it(tmp_path):
    store = ResultStore(tmp_path / "results")

    path = store.write(record_for("a" * 40, dirty=False))
    written = json.loads(path.read_text())

    assert written["config"]["symbol"] == "BTCUSDT"
    assert written["config"]["cost_model"]["name"] == "tokocrypto"
    assert written["config"]["cost_model"]["tax_bps_per_side"] == 21.0
    assert written["commit"] == "a" * 40
    assert written["config_sha256"] == "0" * 64
    assert written["working_tree_dirty"] is False


def test_the_trial_line_summarises_the_record_without_the_result_body(tmp_path):
    trial = record_for("a" * 40, dirty=False).trial_line()

    assert trial["commit"] == "a" * 40
    assert trial["config_name"] == "skeleton-btcusdt-2021h1"
    assert trial["config_sha256"] == "0" * 64
    assert trial["symbol"] == "BTCUSDT"
    assert trial["net_return"] == 0.25
    assert trial["run_at_utc"] == "2026-08-31T00:00:00Z"

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from btorch.utils.conf import load_config, to_dotlist


@dataclass
class CommonConf:
    # Shared arguments used by every single run (mirrors single.py CommonConf).
    # Think: one specimen, one output folder.
    ephys_dataset_path: str = "dataset.nc"
    id: int = 0
    get_size: bool = False
    overwrite: bool = False
    output_path: Path = Path("outputs")

    def ensure_path(self):
        self.output_path.mkdir(parents=True, exist_ok=True)


@dataclass
class SolverConf:
    # Solver-specific knobs that ride along with the single-task config.
    k_candidates: list[int] = field(default_factory=lambda: [1, 2, 3])
    no_excess_zero_fr: bool = False
    plot_fi: bool = False
    plot_filename: str = "fi.png"


@dataclass
class SingleArgConf:
    # Mimics the per-specimen program in single.py.
    # This is what each worker process receives.
    common: CommonConf = field(default_factory=CommonConf)
    solver: SolverConf = field(default_factory=SolverConf)


@dataclass
class BatchArgConf:
    # Manages the whole sweep; batch.single is forwarded to each worker process.
    # batch.py wraps this, decides which specimen IDs to run, and spawns workers.
    single: SingleArgConf = field(default_factory=SingleArgConf)
    id: list[int] | None = None
    id_select: str | None = None
    timeout: int = 60 * 4 * 10
    ncpu_per_task: int = 1
    max_concurrent: int = 64


def test_load_config_merges_file_and_cli(tmp_path, monkeypatch):
    """Mimics the single.py usage: config file + CLI overrides."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "\n".join(
            [
                "common:",
                "  id: 5",
                "  overwrite: true",
                "  output_path: results",
                "solver:",
                "  plot_fi: true",
                "  plot_filename: from_file.png",
            ]
        )
    )

    # Equivalent CLI call the user would make:
    # python single.py config_path=cfg_path common.id=7 \
    #   solver.k_candidates=[9,10] solver.plot_filename=cli.png

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            f"config_path={cfg_file}",
            "common.id=7",
            "solver.k_candidates=[9,10]",
            "solver.plot_filename=cli.png",
        ],
    )

    cfg = load_config(SingleArgConf)

    assert isinstance(cfg, SingleArgConf)
    assert cfg.common.id == 7  # CLI wins over file/defaults
    assert cfg.common.overwrite is True  # from file
    assert cfg.common.output_path == Path("results")
    assert cfg.solver.k_candidates == [9, 10]  # CLI list parsing
    assert cfg.solver.plot_fi is True  # from file
    assert cfg.solver.plot_filename == "cli.png"


def test_to_dotlist_respects_exclusions(monkeypatch):
    """Mimics the batch.py usage where CLI is forwarded to worker tasks."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "single.common.id=3",
            "single.common.get_size=True",
            "single.common.output_path=/tmp/run",
            "single.solver.plot_fi=True",
            "single.solver.plot_filename=cli.png",
        ],
    )

    args, args_cli = load_config(BatchArgConf, return_cli=True)
    args: BatchArgConf

    assert args.single.common.id == 3
    assert args.single.common.output_path == Path("/tmp/run")
    assert args.single.solver.plot_fi is True

    # batch.py strips off id/get_size (set per worker) before forwarding to children
    dotlist = to_dotlist(
        args_cli.single, use_equal=True, exclude={"common.get_size", "common.id"}
    )

    assert "common.id=3" not in dotlist
    assert "common.get_size=True" not in dotlist
    assert set(dotlist) == {
        "common.output_path=/tmp/run",
        "solver.plot_fi=True",
        "solver.plot_filename=cli.png",
    }

    # Subfield traversal should shift path roots, then include/exclude still works.
    solver_only = to_dotlist(
        args_cli.single,
        use_equal=True,
        subfield="solver",
        include={"plot_fi", "plot_filename"},
        exclude={"plot_filename"},
    )
    assert solver_only == ["plot_fi=True"]

    # Nested list traversal with include/exclude at the selected subfield root.
    nested = OmegaConf.create({"a": {"b": [{"c": 1}, {"c": 2}]}})
    assert to_dotlist(nested, subfield="a.b.1", include={"c"}) == ["c=2"]
    assert to_dotlist(nested, subfield="a.b.1", exclude={"c"}) == []


def test_batch_worker_command_build(monkeypatch, tmp_path):
    """End-to-end shape of batch: config file -> batch.single -> worker arg list."""
    cfg_file = tmp_path / "batch.yaml"
    cfg_file.write_text(
        "\n".join(
            [
                "single:",
                "  common:",
                "    output_path: /data/exp",
                "  solver:",
                "    k_candidates: [4,5]",
            ]
        )
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            f"config_path={cfg_file}",
            "single.common.overwrite=True",
            "single.common.id=12",
            "single.solver.plot_fi=True",
        ],
    )

    # BatchArgConf comes from CLI+file; args_cli is the raw CLI OmegaConf
    args, args_cli = load_config(BatchArgConf, return_cli=True)

    # Pretend batch.py builds the per-worker CLI, filtering id/get_size then
    # appending new id
    base_worker_cli = to_dotlist(
        args_cli.single, use_equal=True, exclude={"common.get_size", "common.id"}
    )
    worker_cmd = base_worker_cli + [f"common.id={99}"]

    assert args.single.common.output_path == Path("/data/exp")
    assert args.single.common.overwrite is True
    assert args.single.solver.k_candidates == [4, 5]
    assert args.single.solver.plot_fi is True

    # to_dotlist should carry everything except filtered keys; caller appends target id
    assert "common.id=12" not in worker_cmd
    assert "common.get_size=True" not in worker_cmd
    # Only CLI-provided keys are forwarded; config-file/default values stay in
    # the worker defaults
    assert set(worker_cmd) == {
        "common.overwrite=True",
        "solver.plot_fi=True",
        "common.id=99",
    }


def test_to_dotlist_subfield_missing_policy():
    cfg = OmegaConf.create({"a": {"b": 1}})

    with pytest.raises(KeyError):
        to_dotlist(cfg, subfield="x.y")

    assert to_dotlist(cfg, subfield="x.y", missing_subfield_policy="empty") == []

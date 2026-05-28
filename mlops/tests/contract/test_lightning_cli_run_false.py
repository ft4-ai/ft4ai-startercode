"""Contract: LightningCLI(run=False, save_config_callback=None) is usable
end-to-end, and cli.config dumps reproducibly across builds.

(a) ft4 constructs LightningCLI with run=False so it can interpose hashing,
hset routing, and path injection between instantiation and trainer.fit().
With save_config_callback=None, ft4 takes responsibility for writing
config.yaml to the resolved hset dir.

(b) ft4 computes config_hash from a YAML dump of cli.config. For hset
routing to be stable across separate `ft4 run` invocations, identical args
must produce identical dumps. We approximate that by building twice in one
process; cross-process stability is a corollary if dict iteration order in
the dumped YAML is deterministic (it is, since PyYAML sorts keys by default
on safe_dump, and jsonargparse's dump is similarly deterministic).

This is the highest-stakes test — if any of the four assertions fail, option
(A) from spec § 4.5 is infeasible and we fall back to (B) or (C).
"""
import sys
import yaml
from lightning.pytorch.cli import LightningCLI

from conftest import TinyModel, TinyDataModule


CLI_ARGS = [
    "--model.hidden=16",
    "--data.batch_size=4",
    "--trainer.max_epochs=1",
    "--trainer.enable_progress_bar=false",
    "--trainer.enable_model_summary=false",
    "--trainer.enable_checkpointing=false",
    "--trainer.logger=false",
]

def _build_cli():
    saved = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        return LightningCLI(
            TinyModel, TinyDataModule,
            run=False, save_config_callback=None,
            args=CLI_ARGS,
        )
    finally:
        sys.argv = saved

def test_lightning_cli_constructs_with_run_false():
    """cli built with run=False exposes model / datamodule / trainer / config."""
    cli = _build_cli()
    assert cli.model is not None
    assert cli.datamodule is not None
    assert cli.trainer is not None
    assert cli.config is not None


def test_trainer_fit_after_run_false():
    """trainer.fit() works on the cli's pre-instantiated objects, without
    going through LightningCLI's subcommand dispatch."""
    cli = _build_cli()
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)


def test_config_dump_is_reproducible():
    """Two builds with the same args produce identical YAML dumps.
    Necessary (but not sufficient) for stable config_hash across sessions."""
    cli1 = _build_cli()
    cli2 = _build_cli()
    dump1 = cli1.parser.dump(cli1.config, format="yaml")
    dump2 = cli2.parser.dump(cli2.config, format="yaml")
    assert dump1 == dump2, (
        "cli.config dumps differ across identical builds; "
        "config_hash will not be stable across sessions.\n\n"
        f"--- build 1 ---\n{dump1}\n\n--- build 2 ---\n{dump2}"
    )


def test_config_dump_yaml_roundtrip():
    """cli.config dump survives a yaml round-trip without semantic change.
    config_hash hashes the parsed dict, so yaml.safe_load(dump) must be stable."""
    cli = _build_cli()
    dump = cli.parser.dump(cli.config, format="yaml")
    parsed = yaml.safe_load(dump)
    redumped = yaml.safe_dump(parsed)
    reparsed = yaml.safe_load(redumped)
    assert parsed == reparsed, (
        "config dict not stable across yaml round-trips; "
        "config_hash will be unstable."
    )

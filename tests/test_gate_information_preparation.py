"""Focused identity, isolation and parity checks for current-parent preparation."""
import json
import zipfile
from pathlib import Path

import pytest

from sheepdog.config import LabConfig
from sheepdog.gate_information_preparation import (
    ARCH,
    ARMS,
    arm_config,
    inspect_parent,
    prepare,
)
from sheepdog.recovery_hyperparam_experiment import sha256_file


def parent_fixture(tmp_path: Path):
    model = tmp_path / "checkpoint.zip"
    archive = {"policy_kwargs": {"net_arch": list(ARCH)}, "n_steps": 2048,
               "batch_size": 1024, "gamma": .997, "gae_lambda": .98,
               "ent_coef": .015, "learning_rate": .0001, "n_epochs": 10}
    with zipfile.ZipFile(model, "w") as z:
        z.writestr("data", json.dumps(archive))
    cfg = LabConfig().to_dict()
    cfg["environment"].update(dogs=3, sheep=6, curriculum_stage=34, max_steps=2320)
    meta = {"policy_state_path": str(model), "environment_config": cfg["environment"],
            "reward_config": cfg["rewards"], "policy_config": {
                "observation_size": 54, "action_size": 9, "hidden_sizes": list(ARCH)}}
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps(meta), encoding="utf-8")
    return model, metadata, meta, archive


def test_parent_identity_before_deserialization(tmp_path):
    model, metadata, _, archive = parent_fixture(tmp_path)
    _, data = inspect_parent(model, metadata, sha256_file(model))
    assert data == archive
    with pytest.raises(ValueError, match="SHA-256"):
        inspect_parent(model, metadata, "0" * 64)


@pytest.mark.parametrize("defect", ["reference", "architecture", "task", "mode"])
def test_rejects_mismatched_parent_metadata(tmp_path, defect):
    model, metadata, meta, _ = parent_fixture(tmp_path)
    if defect == "reference":
        meta["policy_state_path"] = "other.zip"
    elif defect == "architecture":
        meta["policy_config"]["hidden_sizes"] = [32, 32]
    elif defect == "task":
        meta["environment_config"]["sheep"] = 5
    else:
        meta["policy_config"]["observation_mode"] = "emergent"
    metadata.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError):
        inspect_parent(model, metadata, sha256_file(model))


def test_arm_parity_and_archive_hyperparameters(tmp_path):
    _, _, meta, archive = parent_fixture(tmp_path)
    configs = []
    for arm, mode in ARMS.items():
        cfg = arm_config(meta, archive, tmp_path / arm, 101, mode).to_dict()
        assert cfg["environment"] == meta["environment_config"]
        assert cfg["rewards"] == meta["reward_config"]
        assert cfg["training"]["entropy_coef"] == .015
        assert cfg["training"]["learning_rate"] == .0001
        assert cfg["training"]["failure_directed_target_ratio"] == .25
        for key in ("observation_mode", "output_dir", "backup_dir", "web_export_dir"):
            cfg["training"].pop(key)
        configs.append(cfg)
    assert configs[0] == configs[1]


def test_existing_directory_rejected_without_touching_it(tmp_path):
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        prepare(tmp_path, Path("missing"), Path("missing"), "0" * 64)
    assert sentinel.read_text() == "keep"

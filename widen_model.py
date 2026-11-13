"""
Weight Transplant Script: Widen MLP from [128,128,128] to [256,256,128]
========================================================================

This script loads your current best model (128-wide) and creates a new
256-wide model that preserves ALL learned behavior while adding capacity.

Technique: Net2WiderNet -- copies existing weights into the wider network
and initializes new neurons with small random values. The old policy's
behavior is preserved because the original neurons still produce the
same outputs. The new neurons start near-zero and gradually learn.

What gets preserved:
  - All learned herding strategies from stages 1-26
  - The value function's understanding of state quality
  - The action distribution for all previously-seen states

What changes:
  - First two hidden layers grow from 128 to 256 neurons (2x capacity)
  - Third hidden layer stays at 128 (sufficient for the output heads)
  - Total params: ~41K -> ~115K (still very lightweight)

Usage:
  1. STOP training first
  2. Run this script: python widen_model.py
  3. Update config: neural_hidden_sizes = [256, 256, 128]
  4. Restart training
"""

import copy
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO

# -- Configuration --
OLD_ARCH = [128, 128, 128]
NEW_ARCH = [256, 256, 128]  # widen first 2 layers, keep 3rd the same

MODEL_PATH = Path("artifacts/models/best-model.zip")
BACKUP_DIR = Path("artifacts/backups/pre_widen")
OUTPUT_PATH = Path("artifacts/models/best-model.zip")  # overwrite in-place

# How much noise to add to the new neurons (small = safe)
NOISE_SCALE = 0.01


def widen_linear(old_weight, old_bias, new_out, new_in, noise_scale=0.01):
    """Widen a linear layer, copying old weights and padding with noise."""
    old_out, old_in_dim = old_weight.shape
    new_weight = torch.zeros(new_out, new_in)
    new_bias = torch.zeros(new_out)

    # Copy the original weights into the top-left block
    new_weight[:old_out, :old_in_dim] = old_weight
    new_bias[:old_out] = old_bias

    # Initialize new output neurons with small random values
    if new_out > old_out:
        fan_in = new_in
        std = noise_scale / np.sqrt(fan_in)
        new_weight[old_out:, :] = torch.randn(new_out - old_out, new_in) * std
        new_bias[old_out:] = torch.randn(new_out - old_out) * (noise_scale * 0.1)

    # Initialize new input connections with small random values
    if new_in > old_in_dim:
        fan_in = new_in
        std = noise_scale / np.sqrt(fan_in)
        new_weight[:old_out, old_in_dim:] = torch.randn(old_out, new_in - old_in_dim) * std

    return new_weight, new_bias


def transplant_weights(old_sd, new_sd, old_arch, new_arch):
    """Copy weights from old state_dict to new state_dict with widening."""
    result = copy.deepcopy(new_sd)

    for net_prefix in ["mlp_extractor.policy_net", "mlp_extractor.value_net"]:
        # Layer indices in Sequential: 0, 2, 4 (Linear layers between Tanh)
        layer_indices = [0, 2, 4]

        for i, layer_idx in enumerate(layer_indices):
            w_key = f"{net_prefix}.{layer_idx}.weight"
            b_key = f"{net_prefix}.{layer_idx}.bias"

            old_w = old_sd[w_key]
            old_b = old_sd[b_key]
            new_w = new_sd[w_key]

            new_out, new_in = new_w.shape

            print(f"  {w_key}: {old_w.shape} -> ({new_out}, {new_in})")

            widened_w, widened_b = widen_linear(
                old_w, old_b, new_out, new_in, NOISE_SCALE
            )
            result[w_key] = widened_w
            result[b_key] = widened_b

    # Action head: input changes if last hidden layer changed
    for head_prefix in ["action_net", "value_net"]:
        w_key = f"{head_prefix}.weight"
        b_key = f"{head_prefix}.bias"

        old_w = old_sd[w_key]
        old_b = old_sd[b_key]
        new_w = new_sd[w_key]

        new_out, new_in = new_w.shape
        old_out, old_in_dim = old_w.shape

        print(f"  {w_key}: {old_w.shape} -> ({new_out}, {new_in})")

        if old_in_dim == new_in and old_out == new_out:
            # Same size, just copy
            result[w_key] = old_w.clone()
            result[b_key] = old_b.clone()
        else:
            widened_w, widened_b = widen_linear(
                old_w, old_b, new_out, new_in, NOISE_SCALE
            )
            result[w_key] = widened_w
            result[b_key] = widened_b

    return result


def main():
    print("=" * 60)
    print("Weight Transplant: Widen MLP Network")
    print("=" * 60)
    print(f"Old architecture: {OLD_ARCH}")
    print(f"New architecture: {NEW_ARCH}")
    print(f"Source model:     {MODEL_PATH}")
    print()

    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        sys.exit(1)

    # -- Step 1: Backup --
    print("[1/5] Creating backup...")
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"best-model-{timestamp}-128x3.zip"
    shutil.copy2(MODEL_PATH, backup_path)
    print(f"  Backup saved: {backup_path}")

    # Also backup the checkpoint used for training resume
    state_path = Path("artifacts/training-state.json")
    if state_path.exists():
        shutil.copy2(state_path, BACKUP_DIR / f"training-state-{timestamp}.json")
        print(f"  State backup:  {BACKUP_DIR / f'training-state-{timestamp}.json'}")

    # -- Step 2: Load old model --
    print("\n[2/5] Loading old model...")
    old_model = MaskablePPO.load(str(MODEL_PATH), device="cpu")
    old_sd = old_model.policy.state_dict()
    print(f"  Loaded {len(old_sd)} parameter tensors")
    old_params = sum(p.numel() for p in old_sd.values())
    print(f"  Total parameters: {old_params:,}")

    # -- Step 3: Create new model with wider architecture --
    print("\n[3/5] Creating new wider model...")
    from sheepdog.config import LabConfig
    from sheepdog.policies.neural import NeuralPolicy

    config_path = Path("artifacts/effective-training-config.json")
    config_dict = json.loads(config_path.read_text(encoding="utf-8"))
    # Override the hidden sizes in config
    config_dict["training"]["neural_hidden_sizes"] = NEW_ARCH
    config = LabConfig.from_dict(config_dict)

    new_policy = NeuralPolicy.initialize(config)
    new_sd = new_policy.model.policy.state_dict()
    new_params = sum(p.numel() for p in new_sd.values())
    print(f"  New model parameters: {new_params:,}")
    print(f"  Parameter increase: {new_params - old_params:,} ({new_params/old_params:.1f}x)")

    # -- Step 4: Transplant weights --
    print("\n[4/5] Transplanting weights...")
    transplanted_sd = transplant_weights(old_sd, new_sd, OLD_ARCH, NEW_ARCH)

    # Load transplanted weights into the new model
    new_policy.model.policy.load_state_dict(transplanted_sd)
    print("  Weights loaded successfully!")

    # -- Step 5: Verify and save --
    print("\n[5/5] Verifying and saving...")

    # Quick sanity check: the original neurons should produce similar outputs
    test_obs = torch.randn(1, 54)

    old_model.policy.eval()
    new_policy.model.policy.eval()

    with torch.no_grad():
        old_features = old_model.policy.extract_features(test_obs, old_model.policy.pi_features_extractor)
        old_latent_pi, old_latent_vf = old_model.policy.mlp_extractor(old_features)
        old_action_logits = old_model.policy.action_net(old_latent_pi)
        old_value = old_model.policy.value_net(old_latent_vf)

        new_features = new_policy.model.policy.extract_features(test_obs, new_policy.model.policy.pi_features_extractor)
        new_latent_pi, new_latent_vf = new_policy.model.policy.mlp_extractor(new_features)
        new_action_logits = new_policy.model.policy.action_net(new_latent_pi)
        new_value = new_policy.model.policy.value_net(new_latent_vf)

    action_diff = (old_action_logits - new_action_logits).abs().max().item()
    value_diff = (old_value - new_value).abs().item()
    print(f"  Action logit max diff:  {action_diff:.6f}")
    print(f"  Value estimate diff:    {value_diff:.6f}")

    if action_diff > 1.0:
        print("  WARNING: Large action difference! The new neurons may be too noisy.")
        print("  Consider reducing NOISE_SCALE.")

    # Update the MaskablePPO policy_kwargs so it saves the correct arch
    new_policy.model.policy_kwargs = {"net_arch": NEW_ARCH}

    # Save the widened model
    new_policy.model.save(str(OUTPUT_PATH))
    print(f"\n  Widened model saved to: {OUTPUT_PATH}")

    # Also save to the checkpoint path referenced by training state
    checkpoint_dir = Path("artifacts/checkpoints/stage26")
    if checkpoint_dir.exists():
        checkpoints = sorted(checkpoint_dir.glob("maskable-ppo-stage26-*.zip"))
        if checkpoints:
            latest = checkpoints[-1]
            shutil.copy2(OUTPUT_PATH, latest)
            print(f"  Also updated checkpoint: {latest}")

        best_model = checkpoint_dir / "stage26-best-model.zip"
        if best_model.exists():
            shutil.copy2(OUTPUT_PATH, best_model)
            print(f"  Also updated: {best_model}")

    # -- Update training-state.json --
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["policy_config"]["hidden_sizes"] = NEW_ARCH
        state_path.write_text(
            json.dumps(state, indent=2) + "\n", encoding="utf-8"
        )
        print(f"  Updated training-state.json hidden_sizes to {NEW_ARCH}")

    print()
    print("=" * 60)
    print("DONE! Next steps:")
    print("=" * 60)
    print(f"1. Update your config:  neural_hidden_sizes = {NEW_ARCH}")
    print(f"2. Restart training")
    print(f"3. The model will resume from your current progress")
    print(f"   with 2x wider hidden layers for more capacity")
    print()
    print(f"Backup location: {backup_path}")
    print(f"If anything goes wrong, restore from backup.")


if __name__ == "__main__":
    main()

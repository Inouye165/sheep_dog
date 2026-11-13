# Stray approach reward adjustment — October 31, 2026

Reward version 1.1 separates `stray_approach_progress` from
`stray_ignore_penalty`. The ignore component retains its existing distance
costs and lower cap. The approach component replaces the isolation-amplified
dog movement bonus with a bounded state-potential difference.

For each unpenned sheep, let d be its distance to the nearest dog. The state
potential is the mean of d / (30 + d). The approach reward is
`20 * stray_ignore_penalty_scale * (previous_potential - current_potential)`.
It uses both previous and current sheep positions, avoids selecting a changing
farthest target, and retains a smooth gradient at long distances. At the saved
stage-34 scale of 0.013, its absolute per-step magnitude is at most 0.26.

The component is signed: retreat pays back approach. Its undiscounted sum is
zero on a closed state loop with a fixed unpenned set and no terminal transitions.
This is not discount-correct potential-based shaping and does not guarantee
policy invariance under PPO discounting or remove exploits in other rewards.
Penning and terminal transitions receive no approach shaping; task rewards
handle those transitions. Missing previous state also produces zero shaping.
The existing total-reward lower cap remains unchanged.

Version-1.0 checkpoint weights remain structurally compatible and compatibility
checks return a reward-change warning. Unknown reward-version mismatches still
fail. New trainer and runtime metadata report version 1.1. Historical replay
rewards are unchanged.

## Validate learning separately

No saved model, live process, curriculum setting, or PPO hyperparameter was
changed by this adjustment. Restart the backend to load the code. Preserve the
existing best checkpoint and fine-tune a copy in a separate experiment directory.
Re-evaluate its starting policy under the new reward function, then compare a
bounded continuation using task success, completion steps, and held-out seeds.
Reward totals across versions are not directly comparable. Passing regression
tests establishes the intended arithmetic, not an improvement in learned herding.

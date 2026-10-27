#!/usr/bin/env python3
"""Static checks for the proposed protocol. Does not run a robot policy or training."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

def validate(path: Path) -> str:
    raw = path.read_bytes()
    p = json.loads(raw)
    errors: list[str] = []
    lines: list[str] = []
    def check(condition: bool, description: str) -> None:
        if not condition:
            errors.append(description)

    check(p["schema_version"] == 1, "schema_version must be 1")
    check(p["status"] == "PROPOSED_NOT_EXECUTED", "Do not relabel this static plan as executed")
    for name, value in p["source_revisions"].items():
        if name.endswith("_commit"):
            check(bool(re.fullmatch(r"[0-9a-f]{40}", value)), f"Malformed commit: {name}")

    s = p["seed_splits"]
    reset_groups = {k: v for k, v in s.items()
                    if isinstance(v, list) and k not in {
                        "evaluation_sampling_development",
                        "evaluation_sampling_confirmation",
                        "ppo_training_seeds"}}
    for name, values in reset_groups.items():
        check(all(isinstance(x, int) and x >= 0 for x in values), f"Invalid seed: {name}")
        check(len(values) == len(set(values)), f"Duplicate seed inside {name}")
    check(set(s["e1_auxiliary"]).issubset(s["e1_development"]), "Aux must be a dev subset")
    keys = [k for k in reset_groups if k != "e1_auxiliary"]
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            check(not (set(reset_groups[a]) & set(reset_groups[b])),
                  f"Reset namespace overlap: {a}, {b}")
    for seed, bounds in s["ppo_training_ranges"].items():
        lo, hi = bounds
        check(lo <= hi, f"Invalid train range {seed}")
        for name, values in reset_groups.items():
            check(not any(lo <= value <= hi for value in values),
                  f"Train range {seed} overlaps {name}")
    train_ranges = list(s["ppo_training_ranges"].items())
    for i, (a, (lo_a, hi_a)) in enumerate(train_ranges):
        for b, (lo_b, hi_b) in train_ranges[i + 1:]:
            check(hi_a < lo_b or hi_b < lo_a, f"Train ranges overlap: {a}, {b}")

    e1, e2, shared = p["e1"], p["e2"], p["shared_eval"]
    check([g["id"] for g in e1["goals"]] == ["A", "B"], "Lock A/B goals")
    check([g["object"] for g in e1["goals"]] == ["cream_cheese_1", "butter_1"],
          "Lock cream cheese / butter objects")
    check(len(e1["policies"]) == 5, "Expected 5 E1 policies")
    check(e1["horizon"] == 400 and e1["stop_on_goal"] is False, "E1 fixed 400")
    check(e1["no_prompt_return_to_original"], "B must persist for the whole episode")
    check(shared["warmup_steps"] == 0, "Warmup must remain zero")
    check(shared["execute_action_chunk"] == 20, "Chunk must remain 20")
    check(shared["model_action_horizon"] == 50, "Prediction horizon must remain 50")
    check(shared["denoise_steps"] == 10, "Denoise steps must remain 10")
    check(shared["evaluation_updates"] == 0 and shared["evaluation_replay_insertions"] == 0,
          "No learning or replay writes in evaluation")
    check(shared["e1_ignore_native_goal_done"] and shared["e1_ignore_internal_horizon_done"],
          "Both E1 done-handling requirements are necessary")
    check(e2["horizon"] == 600 and len(e2["tasks"]) == 2, "E2 two tasks / 600 steps")
    check(e2["conditions"] == ["global", "fixed_stage", "oracle_stage"], "E2 conditions")
    check(set(e1["first_grasp_classes"]) == {"A", "B", "OTHER", "MULTIPLE", "NONE"},
          "All grasp categories must be retained")
    for item in e1["auxiliary_prompts"]:
        if item["id"] in {"empty", "unrelated"}:
            check(item["requested_goal_id"] is None, "No task success for unrelated text")
        if item["id"] == "empty":
            check(item["text"] == "", "Empty prompt must be an empty string")

    lines.append("STATIC PROTOCOL VALIDATION - not simulation or training")
    lines.append(f"Protocol: {p['protocol_id']}")
    lines.append(f"SHA256: {hashlib.sha256(raw).hexdigest()}")
    lines.append("")
    lines.append("Matrix upper bounds:")
    total_cases, total_steps = 0, 0
    default_ids: set[str] = set()
    for m in p["matrices"]:
        if "max_cases" in m:
            cases = m["max_cases"]
            steps = cases * m["horizon"]
            check(steps == m["cap"], f"Matrix cap mismatch: {m['id']}")
        else:
            cases = len(s[m["reset_split"]]) * math.prod(m["dimensions"].values())
            check(cases == m["expected_cases"], f"Case count mismatch: {m['id']}")
            steps = cases * m["horizon"]
        marker = "DEFAULT" if m["default"] else "CONDITIONAL"
        lines.append(f"  {m['id']:12s} {marker:11s} {cases:5,d} cases  {steps:9,d} steps")
        if m["default"]:
            default_ids.add(m["id"])
            total_cases += cases
            total_steps += steps
    check(default_ids == {"e0", "e1_dev", "e1_aux", "e1_confirm", "e2_dev"},
          "Unexpected default stage")
    check(total_cases == 1010, "First batch must have <=1010 cases")
    check(total_steps == 408800, "First batch policy-step bound must be 408800")
    cap = p["execution"]["first_batch_total_physical_step_cap"]
    check(cap >= total_steps, "Physical-step cap cannot be below the policy-step bound")
    check(p["execution"]["new_learning_runs_in_first_batch"] == 0, "No default training")
    order = p["decisions"]["first_batch_execution_order"]
    check(order.index("calibrate_fixed_schedule") < order.index("freeze_full_protocol")
          < order.index("e1_confirm"), "Calibrate before lock, confirm after lock")
    lines.extend([
        "",
        f"Default cases upper bound: {total_cases:,}",
        f"Default policy-step upper bound: {total_steps:,}",
        f"All physical env.step cap: {cap:,}",
        f"Additional construction/replay/retry allowance: {cap - total_steps:,}",
        "This allowance is shared; it is not a separate unlimited budget.",
    ])

    q = p["ppo"]
    nominal_steps = q["env"]["total_num_envs"] * q["env"]["max_steps_per_rollout_epoch"]
    chunk = q["actor"]["num_action_chunks"]
    chunk_samples = nominal_steps // chunk
    check(nominal_steps % chunk == 0, "Rollout must contain integral chunks")
    check(chunk_samples == q["actor"]["global_batch_size"], "PPO batch mismatch")
    check(q["actor"]["global_batch_size"] % q["actor"]["micro_batch_size"] == 0,
          "Micro batch must divide global batch")
    check(q["env"]["auto_reset"] is True and q["env"]["ignore_terminations"] is False,
          "Fragment continuation / terminal semantics changed")
    check(q["algorithm"]["bootstrap_type"] == "standard", "Use standard bootstrap")
    check("raw_seeded_reset" in q["initial_state_mode"], "Audit raw reset vs init-state library")
    formal = q["formal"]
    check(formal["runs"] == len(formal["training_seeds"]) * formal["groups"] == 6,
          "Formal run count mismatch")
    check(formal["runs"] * formal["valid_env_steps_per_run"] ==
          formal["total_target_valid_env_steps"] == 3000000, "Formal train budget mismatch")
    check(q["pilot"]["training_seed"] not in formal["training_seeds"],
          "Pilot seed must be distinct from formal seeds")
    for part in ("pilot", "formal"):
        run = q[part]
        check(run["max_iterations_safety"] * nominal_steps >= run["valid_env_steps_per_run"],
              f"Nominal iteration cap below target: {part}")
    check(q["allowed_budget_overshoot_valid_steps"] == nominal_steps == 1600,
          "Overshoot must be one rollout at most")
    evals = q["evaluation_budgets"]
    expected_eval = {
        "pilot_validation": (2 * 5 * 2 * 10 * 1, 600),
        "formal_validation": (6 * 6 * 2 * 10 * 1, 600),
        "formal_final_test": (6 * 2 * 40 * 2, 600),
        "shared_zero_training_final_test": (2 * 2 * 40 * 2, 600),
        "formal_retention_e1": (6 * 40 * 2 * 2, 400),
        "shared_zero_training_retention_e1": (1 * 40 * 2 * 2, 400),
    }
    for key, (count, horizon) in expected_eval.items():
        check(evals[key]["episodes"] == count and
              evals[key]["max_physical_steps"] == count * horizon,
              f"Evaluation budget mismatch: {key}")
    lines.extend([
        "",
        "Conditional PPO nominal collection: "
        f"{nominal_steps:,} low-level steps / {chunk_samples} chunks per iteration",
        "Formal PPO valid training target: 3,000,000 steps, excluding evaluation",
        "PPO effective steps require actual terminal masks; nominal counts are not results.",
    ])

    sf = p["sft_fallback"]
    check(sf["max_total_updates_across_profiles"] == 3000, "SFT total cap changed")
    check(sf["stage_interface_warmstart"]["mask_actions_crossing_stage_boundary"] is False,
          "Do not depend on an unimplemented SFT action mask")
    check(sf["stage_interface_warmstart"]["require_full_horizon_real_actions"] is True,
          "SFT windows need real actions")
    check(sf["stage_interface_warmstart"]["global_and_stage_share_identical_windows"] is True,
          "SFT variants must use identical windows")
    check(sf["if_both_profiles"]["max_updates_each"] * 2 <= sf["max_total_updates_across_profiles"],
          "Two SFT profiles exceed shared cap")
    later = p["later_modules"]
    check(later["deployable_stage"]["final_branches"] == 3 * 2 * 40 * 2 * 3,
          "Deployment matrix count")
    check(later["deployable_stage"]["max_policy_steps"] ==
          later["deployable_stage"]["final_branches"] * 600, "Deployment step count")
    check({"oracle_stage_tokens", "ground_truth_stage"}.issubset(
        later["deployable_stage"]["forbidden_inputs"]), "Avoid stage-label leakage")
    check(later["recovery"]["max_formal_branches"] == 2 * 40 * 2 * 2 * 3 * 2,
          "Recovery matrix count")
    check(all(module["default"] is False for module in later.values()),
          "Later modules must not silently enter the first batch")
    check("INCONCLUSIVE" not in p["execution"]["execution_statuses"] and
          "INCONCLUSIVE" in p["execution"]["scientific_decisions"],
          "Separate execution status from scientific judgment")
    lines.append("")
    if errors:
        lines.append("FAILED:")
        lines.extend("  - " + error for error in errors)
        raise ValueError("\n".join(lines))
    lines.append("PASS: structural constraints, split separation, matrix arithmetic and budget consistency.")
    lines.append("NOT CHECKED: hardware access, checkpoint contents, simulator, model parity or empirical validity.")
    return "\n".join(lines) + "\n"

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("protocol", nargs="?", type=Path, default=Path(__file__).with_name("protocol.v1.json"))
    ap.add_argument("--output", type=Path, help="Optional UTF-8 validation report")
    args = ap.parse_args()
    try:
        result = validate(args.protocol)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        result = f"{exc}\n"
        if args.output:
            args.output.write_text(result, encoding="utf-8")
        print(result, end="")
        return 1
    if args.output:
        args.output.write_text(result, encoding="utf-8")
    print(result, end="")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())


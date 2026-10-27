"""Exploratory selection/execution cross-tabs from completed, audited E1 cases.

No environment or model is loaded. These descriptive counts complement the
registered metrics; conditioning on first grasp is not a causal experiment.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from tools.research_core import atomic, file_sha, read


def summarize(output):
    output = Path(output)
    closure = read(output / 'closure_check.json')
    assert closure['passed'] and closure['all_main_cases_present']
    assert read(output / 'POSTPROCESS_COMPLETE.json')['passed']
    source = output / 'episodes.jsonl'
    all_records = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
    records = [r for r in all_records if r['case']['phase'] == 'e1_confirm']
    assert len(records) == 800 and all(r['status'] in ['COMPLETE', 'POLICY_FAILURE'] for r in records)
    policies = ['base', 'bounded_random', 'dsrl_s0', 'dsrl_s1', 'dsrl_s2']
    rows = []
    for policy in policies:
        for goal in ['A', 'B']:
            selected = [r for r in records if r['case']['policy'] == policy and r['case']['requested_goal_id'] == goal]
            assert len(selected) == 80
            row = dict(policy=policy, goal=goal, n=len(selected))
            for grasp, success in [(True, True), (True, False), (False, True), (False, False)]:
                key = ('correct' if grasp else 'other') + '_first__goal_' + ('ever' if success else 'never')
                row[key] = sum(bool(r['correct_first_grasp']) == grasp and bool(r['active_goal_ever_success']) == success for r in selected)
            for name in ['correct_first_grasp', 'active_goal_ever_success', 'active_goal_final',
                         'active_goal_stable_released_success', 'active_goal_first_completion_without_prior_wrong_goal',
                         'both_goals_ever']:
                row[name] = sum(bool(r[name]) for r in selected)
            assert sum(row[k] for k in row if '__goal_' in k) == 80
            assert row['correct_first__goal_ever'] + row['correct_first__goal_never'] == row['correct_first_grasp']
            assert row['correct_first__goal_ever'] + row['other_first__goal_ever'] == row['active_goal_ever_success']
            rows.append(row)
    atomic(output / 'diagnostic_breakdown.json', dict(
        exploratory=True, independent_units=40, sampling_repeats=2, rows=rows,
        caution='First-grasp conditioning is descriptive, not causal. OTHER includes wrong, other, multiple and no stable grasp.',
        source_episodes_sha256=file_sha(source), analysis_code_sha256=file_sha(__file__),
        simulation_steps_added=0, vla_calls_added=0))
    with (output / 'diagnostic_breakdown.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Post-hoc E2 description: "wrong_butter_grasp" only covers butter_2, so it
    # cannot rule out grasping the bowl or another unrelated object instead.
    stage_rows = []
    for task in ['T1', 'T2']:
        for condition in ['global', 'fixed_stage', 'oracle_stage']:
            selected = [r for r in all_records if r['case']['phase'] == 'e2_dev'
                        and r['case']['task'] == task and r['case']['condition'] == condition]
            assert len(selected) == 10
            counts = Counter('+'.join(sorted(r['first_grasp_objects'])) if r['first_grasp_objects'] else 'NONE' for r in selected)
            diagnostics = selected[0]['failure_diagnostics']
            stage_rows.append(dict(task=task, condition=condition, n=10, first_stable_objects=dict(sorted(counts.items())),
                episodes_with_diagnostic={k: sum(bool(r['failure_diagnostics'][k]) for r in selected) for k in diagnostics},
                recorded_event_totals={k: sum(r['failure_diagnostics'][k] for r in selected)
                                       for k in ['drop_events', 'repeated_stage_opening']}))
    atomic(output / 'e2_object_diagnostics.json', dict(exploratory_post_hoc=True, rows=stage_rows,
        cautions=['wrong_butter_grasp tracks butter_2 only, not all wrong objects',
                  'Heuristic drop/opening tags are descriptive; absence does not prove absence of every physical failure',
                  'No first-stage completion leaves the stage-transition mechanism untested'],
        source_episodes_sha256=file_sha(source), analysis_code_sha256=file_sha(__file__),
        simulation_steps_added=0, vla_calls_added=0))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 5, figsize=(14, 6), layout='constrained')
    for row in rows:
        ax = axes[['A', 'B'].index(row['goal']), policies.index(row['policy'])]
        counts = np.array([[row['correct_first__goal_ever'], row['correct_first__goal_never']],
                           [row['other_first__goal_ever'], row['other_first__goal_never']]])
        ax.imshow(counts, vmin=0, vmax=80, cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(counts[i, j]), ha='center', va='center', color='white' if counts[i, j] > 40 else '#132f40')
        ax.set_xticks([0, 1], ['Goal ever', 'Goal never'], fontsize=9)
        ax.set_yticks([0, 1], ['Correct first', 'Other first'], fontsize=9)
        ax.set_title(row['policy'] + ' / prompt ' + row['goal'], fontsize=10)
    fig.suptitle('Descriptive cross-tab: first stable grasp vs official active goal (80 branches/cell; 40 reset clusters)')
    folder = output / 'figures'
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / 'e1_selection_execution.png', dpi=180)
    plt.close(fig)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.input), indent=2))

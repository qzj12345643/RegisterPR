"""Plot saved Stage 1 proposals, ranked top five, selected trajectory, and GT."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=root / 'outputs/single_scene')
    parser.add_argument('--anchors', type=Path)
    args = parser.parse_args()
    report = json.loads((args.output / 'result.json').read_text())
    anchors = np.load(args.anchors or root / 'ckpts/trajectory_anchors_256.npy')
    saved = np.load(args.output / 'planner_outputs.npz')
    proposals = anchors + saved['offsets'].reshape(256, 8, 3)
    scores = saved['rewards'].reshape(256)
    top5 = np.argsort(-scores, kind='stable')[:5]
    prediction, gt = saved['trajectory'], saved['ground_truth']
    np.testing.assert_allclose(proposals[top5[0]], prediction, rtol=1e-5, atol=1e-5)
    assert int(top5[0]) == report['selected_anchor']
    assert np.isfinite(proposals).all() and np.isfinite(scores).all()

    def xy(traj):
        return np.vstack([np.zeros((1, 2)), traj[:, :2]])

    colors = ['#e63946', '#0077b6', '#8e44ad', '#f4a261', '#16a085']
    fig, ax = plt.subplots(figsize=(13, 10))
    fig.subplots_adjust(left=.09, right=.97, top=.88, bottom=.1)

    def draw(axis, zoom=False):
        for proposal in proposals:
            points = xy(proposal)
            axis.plot(points[:, 0], points[:, 1], color='#9aa5b1', alpha=.25 if zoom else .36, lw=.7, zorder=1)
        for rank in range(4, -1, -1):
            points = xy(proposals[top5[rank]])
            axis.plot(points[:, 0], points[:, 1], color=colors[rank], lw=2.1, alpha=.95, zorder=3)
        points = xy(prediction)
        axis.plot(points[:, 0], points[:, 1], color=colors[0], lw=3.1, marker='o', ms=4, zorder=5)
        points = xy(gt)
        axis.plot(points[:, 0], points[:, 1], color='#171717', ls='--', lw=2.3, marker='x', ms=5, zorder=6)
        axis.scatter([0], [0], marker='*', s=100, color='black', zorder=7)
        axis.grid(alpha=.18)
        axis.set_xlabel('Forward x (m)')
        axis.set_ylabel('Left y (m)')

    draw(ax)
    ax.set_aspect('equal', adjustable='datalim')
    handles = [Line2D([], [], color='#9aa5b1', lw=1, label='All 256 proposals (anchor + predicted offset)')]
    for rank, idx in enumerate(top5):
        label = f'Top {rank+1}: #{idx}, score {scores[idx]:.4f}'
        if rank == 0:
            label += '  [FINAL]'
        handles.append(Line2D([], [], color=colors[rank], lw=3 if rank == 0 else 2, marker='o' if rank == 0 else None, label=label))
    handles.append(Line2D([], [], color='#171717', lw=2.3, ls='--', marker='x', label='Ground truth'))
    ax.legend(handles=handles, loc='upper left', fontsize=9, framealpha=.96)
    fig.suptitle('WorldDrive Stage 1: 256 proposals, Top 5, final trajectory and GT', fontsize=16, y=.965)
    fig.text(.5, .923, f'Scene {report["token"]}  |  4-second horizon, 0.5-second steps  |  Final = highest-score proposal', ha='center', fontsize=11)
    # A magnified inset makes the closely spaced top-five trajectories readable.
    inset = ax.inset_axes([.53, .075, .44, .34])
    draw(inset, zoom=True)
    selected_points = np.concatenate([proposals[top5, :, :2].reshape(-1, 2), gt[:, :2], np.zeros((1, 2))])
    lower, upper = selected_points.min(axis=0), selected_points.max(axis=0)
    margin = np.maximum((upper-lower)*.12, [.6, .12])
    inset.set_xlim(lower[0]-margin[0], upper[0]+margin[0])
    inset.set_ylim(lower[1]-margin[1], upper[1]+margin[1])
    inset.set_title('Top 5 detail (independent axis scales)', fontsize=9)
    inset.tick_params(labelsize=8)
    inset.set_xlabel('Forward x (m)', fontsize=8)
    inset.set_ylabel('Left y (m)', fontsize=8)
    fig.text(.09, .035, 'Main plot: equal metric scale. Gray curves show every complete proposal; colored curves show ranking by final reward.', fontsize=10)
    for extension in ['png', 'pdf']:
        fig.savefig(args.output / f'proposals_top5_gt.{extension}', dpi=200)
    plt.close(fig)
    summary = {
        'token': report['token'], 'ranking': 'descending final weighted reward (not probability)',
        'top5': [{'rank': rank+1, 'proposal_index': int(idx), 'score': float(scores[idx]), 'is_final': rank == 0} for rank, idx in enumerate(top5)],
        'final_matches_top1': True,
    }
    (args.output / 'top5_proposals.json').write_text(json.dumps(summary, indent=2)+'\n')
    np.savez(args.output / 'all_proposals.npz', proposals=proposals, scores=scores, top5_indices=top5, final=prediction, ground_truth=gt)
    print(json.dumps(summary, indent=2))
    print(args.output / 'proposals_top5_gt.png')


if __name__ == '__main__':
    main()

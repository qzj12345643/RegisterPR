"""Replace cached pre-projection FAR futures while keeping trajectories fixed."""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navsim.agents.worlddrive.worlddrive_planner import TrajEncoder
from navsim.agents.worlddrive.worlddrive_refiner import TrajWorldRefiner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, default=ROOT / 'outputs/single_scene/far_future')
    args = parser.parse_args()
    out = args.case / 'ablation'
    out.mkdir(exist_ok=True)
    torch.set_num_threads(8)
    torch.manual_seed(0)
    device = 'cuda:0'
    saved = np.load(args.case / 'features_and_scores.npz')
    previous = json.loads((args.case / 'results.json').read_text())
    ids = saved['anchor_ids'][:5].astype(int)
    trajectories = torch.from_numpy(saved['trajectories'][:5]).float().to(device)
    future = torch.from_numpy(saved['decoder_hidden']).float().to(device)
    assert future.shape == (8, 2048, 512)
    anchors = torch.from_numpy(np.load(ROOT / 'ckpts/trajectory_anchors_256.npy')).float()
    encoder = TrajEncoder(traj_vocab=anchors)
    refiner = TrajWorldRefiner()
    weights = torch.load(ROOT / 'ckpts/worlddrive_stage2_train_89p2_pdms.ckpt', map_location='cpu', weights_only=False)['state_dict']
    weights = {k.removeprefix('agent.'): v for k, v in weights.items()}
    encoder.load_state_dict({k.removeprefix('trajencoder_wm.'): v for k, v in weights.items() if k.startswith('trajencoder_wm.')}, strict=True)
    refiner.load_state_dict({k.removeprefix('trajrefiner.'): v for k, v in weights.items() if k.startswith('trajrefiner.')}, strict=True)
    del weights
    encoder.to(device).eval(); refiner.to(device).eval()
    with torch.inference_mode():
        selected_anchors = encoder.traj_vocab[ids]
        embedding = encoder.traj_vocab_encoder(encoder.norm_odo(selected_anchors).reshape(5, -1)) + encoder.traj_offset_encoder((trajectories-selected_anchors).reshape(5, -1))

        def score(replacement):
            # Both consumers of future_scene_embed in original forward are
            # replaced: cross-attention memory AND the spatial mean residual.
            query = embedding[:, None]
            state = refiner.traj_refine_decoder(query, replacement)[:, 0] + replacement.mean(dim=1)
            return refiner.refine_reward_heads[0](state)[:, 0].cpu().numpy()

        baseline = score(future[:5])
        expected = saved['far_scores'][:5]
        np.testing.assert_allclose(baseline, expected, rtol=1e-5, atol=1e-4)
        print('Baseline agrees with previous full FAR forward:', float(np.max(np.abs(baseline-expected))), flush=True)
        # 25 combinations cover all 120 bijective assignments of five futures.
        matrix = np.stack([score(future[j:j+1].expand(5, -1, -1)) for j in range(5)], axis=1)
        np.testing.assert_allclose(np.diag(matrix), baseline, rtol=1e-5, atol=1e-4)
        mean_scores = score(future[:5].mean(dim=0, keepdim=True).expand(5, -1, -1))
        far_scores = score(future[6:7].expand(5, -1, -1))
        zero_scores = score(torch.zeros_like(future[:5]))

    def describe(name, scores):
        order = np.argsort(-scores, kind='stable')
        base_order = np.argsort(-baseline, kind='stable')
        base_ranks = np.argsort(base_order)
        ranks = np.argsort(order)
        spearman = 1-6*float(np.square(ranks-base_ranks).sum())/(5*(25-1))
        return {'name': name, 'scores': list(map(float, scores)), 'order_ids': ids[order].tolist(), 'selected_id': int(ids[order[0]]), 'winner_unchanged': bool(order[0] == base_order[0]), 'full_order_unchanged': bool(np.array_equal(order, base_order)), 'spearman': spearman, 'score_rmse': float(np.sqrt(np.mean((scores-baseline)**2))), 'max_abs_score_change': float(np.abs(scores-baseline).max())}

    conditions = [describe('matched baseline', baseline)]
    conditions.extend(describe(f'all share future #{idx}', matrix[:, j]) for j, idx in enumerate(ids))
    conditions += [describe('all share mean future', mean_scores), describe('all share far-control future #174', far_scores), describe('all zero future (out-of-distribution)', zero_scores)]
    permutations = []
    for permutation in itertools.permutations(range(5)):
        scores = matrix[np.arange(5), permutation]
        permutations.append(dict(describe('permutation', scores), future_source_ids=ids[list(permutation)].tolist(), fixed_points=sum(i == j for i, j in enumerate(permutation))))
    nonidentity = [p for p in permutations if p['fixed_points'] != 5]
    derangements = [p for p in permutations if p['fixed_points'] == 0]
    def aggregate(items):
        return {'count': len(items), 'same_winner': sum(p['winner_unchanged'] for p in items), 'same_full_order': sum(p['full_order_unchanged'] for p in items), 'mean_spearman': float(np.mean([p['spearman'] for p in items])), 'mean_score_rmse': float(np.mean([p['score_rmse'] for p in items])), 'winner_counts': {str(idx): int(sum(p['selected_id'] == int(idx) for p in items)) for idx in ids}}
    row_effect = matrix.mean(axis=1, keepdims=True)-matrix.mean()
    col_effect = matrix.mean(axis=0, keepdims=True)-matrix.mean()
    interaction = matrix-matrix.mean()-row_effect-col_effect
    total = np.square(matrix-matrix.mean()).sum()
    variance = {'trajectory_row': float(5*np.square(row_effect).sum()/total), 'future_column': float(5*np.square(col_effect).sum()/total), 'interaction': float(np.square(interaction).sum()/total)}
    result = {'token': previous['token'], 'proposal_ids_in_row_order': ids.tolist(), 'intervention': 'Cached 2048x512 refine_decoder output, before visual_proj; replace both attention memory and mean residual; trajectory and trajectory embedding fixed.', 'baseline_max_abs_error': float(np.max(np.abs(baseline-expected))), 'score_matrix_trajectory_by_future': matrix.tolist(), 'conditions': conditions, 'nonidentity_permutations': aggregate(nonidentity), 'fully_mismatched_permutations': aggregate(derangements), 'all_permutations': permutations, 'grid_variance_decomposition': variance, 'scope': 'One scene, fixed five similar proposals, frozen trained weights. No PDM or safety evaluation. Zero future is out-of-distribution.'}
    (out / 'results.json').write_text(json.dumps(result, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im = axes[0].imshow(matrix, cmap='viridis')
    for i in range(5):
        for j in range(5):
            axes[0].text(j, i, f'{matrix[i,j]:.3f}', ha='center', va='center', color='white' if matrix[i,j] < (matrix.min()+matrix.max())/2 else 'black')
    axes[0].set_xticks(range(5), ids); axes[0].set_yticks(range(5), ids)
    axes[0].set(xlabel='Source of future feature', ylabel='Fixed trajectory ID', title='FAR scores: all 25 trajectory/future combinations')
    fig.colorbar(im, ax=axes[0])
    for scores, label, style in [(baseline, 'Matched baseline', 'o-'), (matrix[:, 0], 'Shared Top1 future', 's--'), (mean_scores, 'Shared mean future', '^--'), (far_scores, 'Shared far-control future', 'x:')]:
        axes[1].plot(range(5), scores, style, label=label)
    axes[1].set_xticks(range(5), ids)
    axes[1].set(xlabel='Fixed trajectory ID', ylabel='FAR score (not probability)', title='Does sharing the future change the ranking?')
    axes[1].legend(); axes[1].grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out / 'ablation.png', dpi=180); plt.close(fig)
    lines = ['# FAR future 特征替换测试', '', f'场景 {previous["token"]}；固定原五条轨迹和轨迹编码，复用上次保存的精确512维特征。', '',
        '替换位置为 refine_decoder 输出，位于16维 visual_proj之前。同时替换 traj_refine_decoder 的注意力 memory 和加到评分特征上的 future spatial mean；不是只改诊断图。', '',
        f'原始评分复现最大误差：{result["baseline_max_abs_error"]:.8f}。', '',
        '| 条件 | 排序 | 最终选择 | score RMSE |', '|---|---|---:|---:|']
    for c in conditions:
        lines.append(f'| {c["name"]} | {" → ".join(map(str,c["order_ids"]))} | {c["selected_id"]} | {c["score_rmse"]:.4f} |')
    lines += ['', '## 打乱对应关系', '',
        f'穷举119种非原始排列：{result["nonidentity_permutations"]}', '',
        f'其中44种每条轨迹都错配future的排列：{result["fully_mismatched_permutations"]}', '',
        '这些结果只评价该case的分数与排序敏感度，不证明所选轨迹更安全或更准确，也不证明跨场景结论。共享Top5之一或均值属于局部替换；远距离future与零特征的干预幅度更大，其中零特征是分布外对照。', '',
        'grid_variance_decomposition 是25格分数的均衡双因素平方和分解，仅描述这张评分表的变化来源，不是全模型因果贡献。', '',
        '复跑：`CUDA_VISIBLE_DEVICES=4 MPLCONFIGDIR=outputs/.matplotlib .venv/bin/python scripts/ablate_far_future.py`', '']
    (out / 'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({k: v for k, v in result.items() if k not in ['all_permutations','score_matrix_trajectory_by_future']}, indent=2), flush=True)


if __name__ == '__main__':
    main()

"""Aggregate a complete WorldDrive navtest candidate evaluation."""
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[1]

def corr(x,y):
    a,b=rankdata(x),rankdata(y)
    if np.std(a)==0 or np.std(b)==0: return float('nan')
    return float(np.corrcoef(a,b)[0,1])

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'outputs/navtest_worlddrive_stage2_full');args=p.parse_args();out=args.output
    scenes=json.loads((out/'manifest.json').read_text());assert len(scenes)==12146
    rows=[];allm=[];selected=[];rhos=[];far_rhos=[];coverage={k:[] for k in [5,10,16,32,64]};errors=[];diversity=[];confidence=[]
    for s in scenes:
        token=s['token']
        with np.load(out/'candidates'/f'{token}.npz') as c, np.load(out/'pdm'/f'{token}.npz') as d:
            m=d['metrics'];assert m.shape==(256,7) and np.isfinite(m).all();score=m[:,-1]
            a,b=int(c['stage1_id']),int(c['stage2_id']);ids=c['top5_ids'];assert b in ids
            oracle=int(score.argmax());toporacle=int(ids[score[ids].argmax()]);indices=[a,b,toporacle,oracle]
            allm.append(m);selected.append(indices);rhos.append(corr(c['stage1_scores'],score));far_rhos.append(corr(c['far_scores'],score[ids]));errors.append(float(d['single_trajectory_max_error']))
            order=np.argsort(-c['stage1_scores'],kind='stable')
            for k in coverage:coverage[k].append(float(score[order[:k]].max()>=score[oracle]-1e-6))
            xy=c['proposals'][ids,:,:2];pairs=np.linalg.norm(xy[:,None]-xy[None,:],axis=-1);upper=np.triu_indices(5,1)
            ade=float(pairs.mean(-1)[upper].mean());fde=float(pairs[:,:,-1][upper].mean());diversity.append([ade,fde]);confidence.append(c['predicted_metrics'][b])
            rows.append(dict(token=token,stage1_id=a,stage2_id=b,top5_oracle_id=toporacle,oracle_id=oracle,stage1_pdms=score[a],stage2_pdms=score[b],top5_oracle_pdms=score[toporacle],oracle_pdms=score[oracle],far_delta=score[b]-score[a],far_regret=score[oracle]-score[b],top5_mean_pair_ade_m=ade,top5_mean_pair_fde_m=fde))
    m=np.stack(allm);ix=np.array(selected);r=np.arange(len(scenes));v=np.stack([m[r,ix[:,i]] for i in range(4)],axis=1);names=['Stage 1','Stage 2 / FAR','Top-5 oracle','256 oracle'];metricnames=['NC','DAC','EP','TTC','Comfort','DDC','PDMS'];means=v.mean(0)
    delta=v[:,1,-1]-v[:,0,-1];regrets=v[:,3,-1,None]-v[:,:2,-1];confidence=np.array(confidence)
    ranking={}
    for j,name in enumerate(names[:2]):
        ranking[name]={'mean_regret':float(regrets[:,j].mean()),'median_regret':float(np.median(regrets[:,j])),'p90_regret':float(np.quantile(regrets[:,j],.9)),'zero_regret_rate':float((regrets[:,j]<=1e-6).mean()),'first_argmax_match':float((ix[:,j]==ix[:,3]).mean())}
    safety={}
    for j,name in enumerate(names[:2]):
        z=v[:,j];oracle=v[:,3];fails=z[:,[0,1,3]]<1-1e-6;inferior=z[:,[0,1,3]]<oracle[:,[0,1,3]]-1e-6
        safety[name]={'NC_lt1':int(fails[:,0].sum()),'NC_zero':int((z[:,0]==0).sum()),'DAC_lt1':int(fails[:,1].sum()),'TTC_lt1':int(fails[:,2].sum()),'any_failure':int(fails.any(1).sum()),'safety_inferior_to_oracle':int(inferior.any(1).sum())}
    ep_cf=v[:,1,0]*v[:,1,1]*(5*v[:,3,2]+5*v[:,1,3]+2*v[:,1,4])/12
    summary={'n':len(scenes),'means':{n:dict(zip(metricnames,map(float,means[j]))) for j,n in enumerate(names)},'ranking':ranking,'stage1_spearman_mean':float(np.nanmean(rhos)),'stage1_spearman_valid':int(np.isfinite(rhos).sum()),'far_top5_spearman_mean':float(np.nanmean(far_rhos)),'far_spearman_valid':int(np.isfinite(far_rhos).sum()),'topk_oracle_coverage':{k:float(np.mean(x)) for k,x in coverage.items()},'far_improved':int((delta>1e-6).sum()),'far_worse':int((delta< -1e-6).sum()),'far_equal':int((abs(delta)<=1e-6).sum()),'far_changed_id':int((ix[:,0]!=ix[:,1]).sum()),'far_top5_optimal_rate':float((v[:,2,-1]-v[:,1,-1]<=1e-6).mean()),'safety':safety,'ep_only_counterfactual_pdms':float(ep_cf.mean()),'max_standard_pdm_error':max(errors),'top5_pair_ade_m_quantiles':np.quantile(np.array(diversity)[:,0],[0,.25,.5,.75,.9,1]).tolist(),'top5_pair_fde_m_quantiles':np.quantile(np.array(diversity)[:,1],[0,.25,.5,.75,.9,1]).tolist()}
    for idx,label in [(0,'NC'),(1,'DAC'),(3,'TTC')]:
        mask=v[:,1,idx]<1-1e-6
        summary[f'far_selected_{label}_failed_predicted_probability_mean']=float(confidence[mask,idx].mean()) if mask.any() else None
    (out/'aggregate_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    with (out/'per_scene.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(15,4))
    axs[0].bar(['Stage 1','FAR','Top5 oracle','256 oracle'],means[:,-1]*100);axs[0].set_ylabel('PDMS (0-100)');axs[0].set_ylim(0,100)
    for j,name in enumerate(names[:2]):
        x=np.sort(regrets[:,j]);axs[1].plot(x,np.arange(1,len(x)+1)/len(x),label=name)
    axs[1].set_xlabel('Oracle regret (0-1)');axs[1].set_ylabel('CDF');axs[1].legend()
    axs[2].bar(['Improved','Equal','Worse'],[summary['far_improved'],summary['far_equal'],summary['far_worse']]);axs[2].set_ylabel('Scenes: FAR vs Stage 1')
    fig.tight_layout();fig.savefig(out/'evaluation_summary.png',dpi=180);plt.close(fig)
    lines=['# WorldDrive Navtest 全量轨迹评测报告','','## 1. 结论摘要','',f'完成 NAVSIM v1 navtest 全部 **{len(scenes):,}** 个场景，每场景评估 256 条候选，共 **{len(scenes)*256:,}** 条轨迹。',f'Stage 1 PDMS 为 **{means[0,-1]*100:.4f}**，FAR 为 **{means[1,-1]*100:.4f}**，FAR 相对变化 **{delta.mean()*100:+.4f}** 分（0–100）。',f'Top-5 oracle 为 **{means[2,-1]*100:.4f}**，256 候选 oracle 为 **{means[3,-1]*100:.4f}**。Oracle 使用真实标签挑选，属于候选集合上界，不能部署。','','## 2. 协议与完整性','','- 模型：`worlddrive_stage2_train_89p2_pdms.ckpt`；同时读取其 Stage 1 选择与 FAR 最终选择。','- 每条候选 8 个未来位姿，间隔 0.5 s；官方 PDM 仿真 40 步 × 0.1 s。','- 图像与 metric cache 来自 DriveWorld-VLA 数据目录；模型、PDMSimulator、PDMScorer 来自本 WorldDrive 仓库。','- FP32 推理；VAE posterior 采样种子为 `int(token,16) % (2**31-1)`，可断点重现。','- 批量评估 256 候选，但 EP 按每个候选与参考轨迹组成的二元组独立归一化；不以全体候选最大进度作为分母。',f'- 每个场景的 Stage 1、FAR 选择均与官方单轨迹 `pdm_score` 复核，最大子指标绝对误差 **{max(errors):.3g}**。','- PDMS = NC × DAC × (5 EP + 5 TTC + 2 Comfort) / 12；DDC 权重为 0。','- 下表统一使用 0–1 标度；零 regret、oracle 覆盖与 FAR 增减的容差为 1e-6。','','## 3. 总体结果','','| 指标 | Stage 1 | FAR | Top-5 oracle | 256 oracle |','|---|---:|---:|---:|---:|']
    for k,n in enumerate(metricnames):lines.append('| '+n+' | '+' | '.join(f'{x:.6f}' for x in means[:,k])+' |')
    lines+=['','## 4. 排序质量','','| 指标 | Stage 1 | FAR |','|---|---:|---:|']
    for key in ranking[names[0]]:lines.append(f'| {key} | {ranking[names[0]][key]:.6f} | {ranking[names[1]][key]:.6f} |')
    lines += ['',f'Stage 1 全 256 候选逐场景 Spearman 均值：{summary["stage1_spearman_mean"]:.6f}（有效 {summary["stage1_spearman_valid"]} 场景）；FAR 仅 Top 5：{summary["far_top5_spearman_mean"]:.6f}（有效 {summary["far_spearman_valid"]} 场景）。常数排名场景不计入相关性均值。','', '| Stage 1 top-k | 包含并列 PDM oracle 的比例 |','|---|---:|']
    for k,val in summary['topk_oracle_coverage'].items():lines.append(f'| {k} | {val:.2%} |')
    lines+=['', '零 regret 接受并列最优；首个 argmax 一致率仅按候选索引比较，两者不等价。','','## 5. FAR 的实际收益','',f'FAR 改选 {summary["far_changed_id"]} 个场景；真实 PDMS 提升 {summary["far_improved"]} 个、下降 {summary["far_worse"]} 个、持平 {summary["far_equal"]} 个。',f'FAR 在 Top 5 内达到并列最优的比例为 {summary["far_top5_optimal_rate"]:.2%}。',f'候选召回损失（256 oracle − Top-5 oracle）为 {(means[3,-1]-means[2,-1])*100:.4f} 分；Top-5 排序损失（Top-5 oracle − FAR）为 {(means[2,-1]-means[1,-1])*100:.4f} 分。','','这些结果检验最终重排收益，不能单独证明未来 latent 的语义预测是收益来源；要判断其是否优于等容量 Transformer，需要控制参数量、训练数据与训练预算的重训练消融。','','## 6. 安全与进度损失','','| 事件 | Stage 1 场景数 | FAR 场景数 |','|---|---:|---:|']
    for key in safety[names[0]]:lines.append(f'| {key} | {safety[names[0]][key]} | {safety[names[1]][key]} |')
    lines+=['','各事件可能重叠；NC<1 包括部分碰撞扣分，TTC 失败不是实际碰撞。',f'只将 FAR 所选轨迹的 EP 替换成同场景 256 oracle EP，其余指标保持不变，反事实平均 PDMS 为 {ep_cf.mean():.6f}，相对 FAR 变化 {ep_cf.mean()-means[1,-1]:+.6f}。此项仅作误差分解，不是可实现收益。','','## 7. Top-5 轨迹差异','',f'Top-5 内 10 对轨迹的平均 ADE（xy 欧氏距离）的场景中位数为 {summary["top5_pair_ade_m_quantiles"][2]:.4f} m，P90 为 {summary["top5_pair_ade_m_quantiles"][4]:.4f} m；平均 FDE 的中位数为 {summary["top5_pair_fde_m_quantiles"][2]:.4f} m。','此指标按相同未来时间点比较，既包含路线也包含速度差异；本评测没有对全量未来 latent 做语义或图像相似度检验。','','## 8. 产物与复现','','- [汇总 JSON](../outputs/navtest_worlddrive_stage2_full/aggregate_summary.json)','- [逐场景 CSV](../outputs/navtest_worlddrive_stage2_full/per_scene.csv)','- [完整性检查](../outputs/navtest_worlddrive_stage2_full/input_audit.json)','- 每场景候选与预测分数：`outputs/navtest_worlddrive_stage2_full/candidates/`','- 每条候选真实 PDM 子指标：`outputs/navtest_worlddrive_stage2_full/pdm/`','- 实现：`scripts/prepare_worlddrive_navtest.py`、`scripts/infer_worlddrive_navtest.py`、`scripts/score_worlddrive_navtest.py`、`scripts/report_worlddrive_navtest.py`。','','![评测汇总](../outputs/navtest_worlddrive_stage2_full/evaluation_summary.png)','']
    report=ROOT/'docs/worlddrive_navsim_test_trajectory_report.md';report.write_text('\n'.join(lines));print(report);print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

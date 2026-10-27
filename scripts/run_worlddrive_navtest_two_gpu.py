"""Run two resumable inference shards, streaming PDM, and final report."""
import json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/navtest_worlddrive_stage2_full'
CPU='/mnt/cfs/qiuzijun/miniconda3/envs/driveworld-vla/bin/python'
os.chdir(ROOT)
processes=[];handles=[]
def launch(command,log,env=None):
    handle=open(ROOT/'logs'/log,'a');handles.append(handle)
    process=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,env=env)
    processes.append(process);return process

def state(status,**kw):
    p=OUT/'pipeline_status.tmp';p.write_text(json.dumps(dict(status=status,pid=os.getpid(),gpus=[5,6],updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw),indent=2));os.replace(p,OUT/'pipeline_status.json')
try:
    env=os.environ.copy();env.update(OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    gpu=[]
    for rank,device in enumerate([5,6]):
        local=env.copy();local['CUDA_VISIBLE_DEVICES']=str(device)
        gpu.append(launch([str(ROOT/'.venv/bin/python'),'scripts/infer_worlddrive_navtest.py','--rank',str(rank),'--world-size','2','--batch-size','1','--save-visual'],f'navtest_gpu{device}.log',local))
    cpu=launch([CPU,'scripts/score_worlddrive_navtest.py','--workers','24','--watch'],'navtest_pdm_full.log')
    state('running',gpu_pids=[p.pid for p in gpu],pdm_pid=cpu.pid)
    while any(p.poll() is None for p in gpu):
        if any(p.poll() not in [None,0] for p in gpu):raise RuntimeError('GPU worker failed; inspect logs/navtest_gpu*.log')
        if cpu.poll() not in [None,0]:raise RuntimeError('PDM worker failed; inspect logs/navtest_pdm_full.log')
        time.sleep(10)
    if any(p.returncode for p in gpu):raise RuntimeError('GPU inference failed')
    expected=len(json.loads((OUT/'manifest.json').read_text()))
    assert len(list((OUT/'candidates').glob('*.npz')))==expected
    state('scoring',pdm_pid=cpu.pid)
    if cpu.wait()!=0:raise RuntimeError('PDM scoring failed')
    state('reporting')
    report=launch([CPU,'scripts/report_worlddrive_navtest.py'],'navtest_report.log')
    if report.wait()!=0:raise RuntimeError('Report generation failed')
    state('complete',report=str(ROOT/'docs/worlddrive_navsim_test_trajectory_report.md'))
except BaseException as exc:
    state('failed',error=repr(exc))
    for p in processes:
        if p.poll() is None:p.terminate()
    raise
finally:
    for h in handles:h.close()

import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from exp_3.simulator import DockerSimulator
from exp_5.safety import fit_bounds


def calibrate(config):
    output=ROOT/'exp_5/calibration'
    output.mkdir(exist_ok=False)
    backend=DockerSimulator(config['docker_image'])
    rows=[]
    try:
        for seed in config['calibration_seeds']:
            backend.reset(seed=seed,warmup_steps=config['warmup_steps'],window=60,
                          frequency_seconds=config['frequency_seconds'])
            episode=[]
            for step in range(config['calibration_steps']):
                row,_=backend.step(None)
                if not np.isfinite(row).all():
                    raise ValueError('Invalid calibration observation')
                episode.append(row)
                if len(episode)>=4 and np.all(np.abs(np.diff(np.asarray(episode[-4:])[:,:41],axis=0))<=config['freeze_atol']):
                    raise ValueError('Frozen calibration trajectory')
                rows.append([seed,step+1,*row])
            print(f'Calibration seed {seed} complete',flush=True)
    finally:
        backend.close()
    values=np.asarray(rows)
    np.savetxt(output/'observations.csv',values,delimiter=',',header=','.join(['seed','step']+
               [f'CV[{i}]' for i in range(41)]+[f'MV[{i}]' for i in range(12)]),comments='')
    bounds=fit_bounds(values[:,2:])
    bounds.update(seeds=config['calibration_seeds'],steps_per_seed=config['calibration_steps'])
    (output/'bounds.json').write_text(json.dumps(bounds,indent=2)+'\n')
    return bounds


if __name__=='__main__':
    calibrate(json.loads((ROOT/'exp_5/config.json').read_text()))

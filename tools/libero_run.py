"""Run one immutable, recorded LIBERO baseline, evaluation or integration smoke."""
import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from tools.run_logger import RunLogger, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['smoke', 'base', 'train', 'eval'], required=True)
    parser.add_argument('--config', default='configs/libero_baseline.json')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', required=True)
    parser.add_argument('--restore-path', default='')
    args = parser.parse_args()
    directory = Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text())
    config.update(mode=args.mode, seed=args.seed, outputdir=str(directory),
                  restore_path=args.restore_path, suffix='', prefix='dsrl_pi0_libero',
                  launch_group_id='baseline', wandb_project='DSRL_pi0_Libero',
                  local_logging=True, config_path=str(Path(args.config).resolve()))
    config['discount'] = config['train_kwargs']['discount']
    if args.mode == 'smoke':
        config.update(max_steps=16, batch_size=16, start_online_updates=1,
                      max_timesteps=40, eval_episodes=1, eval_interval=10000,
                      checkpoint_interval=16, log_interval=1)
    if args.mode in ['base', 'eval']:
        config['eval_episodes'] = 100
    write_json(directory / 'config.json', config)
    versions = {d.metadata['Name']: d.version for d in metadata.distributions()}
    write_json(directory / 'dependencies.json', versions)
    write_json(directory / 'run.json', {
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'submodules': subprocess.check_output(['git', 'submodule', 'status'], text=True),
        'mode': args.mode, 'seed': args.seed, 'job_id': os.environ.get('SLURM_JOB_ID'),
        'hostname': os.uname().nodename, 'python': sys.executable,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'model_source': 's3://openpi-assets/checkpoints/pi0_libero',
        'model_path': os.environ['DSRL_MODEL_PATH'], 'controller_resume': 'weights and optimizer only',
    })
    logger = RunLogger(directory)
    config['run_logger'] = logger
    try:
        logger.log({'phase': 'initializing'}, 0)
        from jaxrl2.utils.general_utils import AttrDict
        from examples.train_sim import main as run
        run(AttrDict(config))
        logger.log({'phase': 'completed'}, logger.latest.get('gradient_updates', 0))
        write_json(directory / 'SUCCESS.json', {'mode': args.mode, 'passed': True})
    except BaseException as error:
        write_json(directory / 'FAILED.json', {'type': type(error).__name__,
                                              'error': str(error), 'traceback': traceback.format_exc()})
        raise
    finally:
        logger.close()


if __name__ == '__main__':
    main()

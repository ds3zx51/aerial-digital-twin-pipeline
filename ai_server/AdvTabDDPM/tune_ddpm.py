import sys
import os
# print(sys.path)
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import tddpm_lib
# print(tddpm_lib.__file__)

import subprocess
import optuna
# from optuna.importance import MeanDecreaseImpurityImportanceEvaluator # add for optuna
# from optuna.importance import get_param_importances # add for optuna
from copy import deepcopy
import shutil
import argparse
from pathlib import Path

import torch
# if torch.cuda.is_available():
#     print(f"tune_ddpm.py Number of GPUs available: {torch.cuda.device_count()}")
#     print(f"Current GPU: {torch.cuda.current_device()}")
#     print(f"GPU name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
#     os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use '0' if you have only one GPU
#     # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# else:
#     print("CUDA is not available.")
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print('device: ', device)

parser = argparse.ArgumentParser()
parser.add_argument('ds_name', type=str)
parser.add_argument('train_size', type=int)
parser.add_argument('eval_type', type=str)
parser.add_argument('eval_model', type=str)
parser.add_argument('prefix', type=str)
parser.add_argument('--eval_seeds', action='store_true',  default=False)

args = parser.parse_args()
train_size = args.train_size
ds_name = args.ds_name
eval_type = args.eval_type 
assert eval_type in ('merged', 'synthetic')
prefix = str(args.prefix)

pipeline = f'pipeline.py'
base_config_path = f'exp/{ds_name}/config.toml'
parent_path = Path(f'exp/{ds_name}/')
exps_path = Path(f'exp/{ds_name}/many-exps/') # temporary dir. maybe will be replaced with tempdiвdr
eval_seeds = f'eval_seeds.py'

os.makedirs(exps_path, exist_ok=True)

def _suggest_mlp_layers(trial):
    def suggest_dim(name):
        t = trial.suggest_int(name, d_min, d_max)
        return 2 ** t
    min_n_layers, max_n_layers, d_min, d_max = 1, 4, 7, 10
    n_layers = 2 * trial.suggest_int('n_layers', min_n_layers, max_n_layers)
    d_first = [suggest_dim('d_first')] if n_layers else []
    d_middle = (
        [suggest_dim('d_middle')] * (n_layers - 2)
        if n_layers > 2
        else []
    )
    d_last = [suggest_dim('d_last')] if n_layers > 1 else []
    d_layers = d_first + d_middle + d_last
    return d_layers

def objective(trial):
    
    # lr = trial.suggest_loguniform('lr', 2e-5, 1e-4) # modify for tune better
    lr = trial.suggest_loguniform('lr', 0.00001, 0.003)
    d_layers = _suggest_mlp_layers(trial)
    # weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3) # modify for tune better 
    weight_decay = 0.0
    # batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024, 2048]) # modify for tune better 
    batch_size = trial.suggest_categorical('batch_size', [256, 4096])
    # steps = trial.suggest_categorical('steps', [10000, 15000, 20000]) # modify for tune better 
    steps = trial.suggest_categorical('steps', [5000, 20000, 30000])
    # steps = trial.suggest_categorical('steps', [500]) # for debug
    gaussian_loss_type = 'mse'
    scheduler = trial.suggest_categorical('scheduler', ['cosine', 'linear'])
    # num_timesteps = trial.suggest_categorical('num_timesteps', [50, 200, 500]) # modify for tune better 
    num_timesteps = trial.suggest_categorical('num_timesteps', [100, 1000])
    # num_samples = int(train_size * (2 ** trial.suggest_int('num_samples', -1, 1))) # modify for tune better 
    num_samples = int(train_size * (2 ** trial.suggest_int('num_samples', -2, 1)))
    # dropout = trial.suggest_uniform('dropout', 0.1, 0.2) # add for tune better

    base_config = tddpm_lib.load_config(base_config_path)

    base_config['train']['main']['lr'] = lr
    base_config['train']['main']['steps'] = steps
    base_config['train']['main']['batch_size'] = batch_size
    base_config['train']['main']['weight_decay'] = weight_decay
    base_config['model_params']['rtdl_params']['d_layers'] = d_layers
    # base_config['model_params']['rtdl_params']['dropout'] = dropout # add for tune better
    base_config['eval']['type']['eval_type'] = eval_type
    base_config['sample']['num_samples'] = num_samples
    base_config['diffusion_params']['gaussian_loss_type'] = gaussian_loss_type
    base_config['diffusion_params']['num_timesteps'] = num_timesteps
    base_config['diffusion_params']['scheduler'] = scheduler

    base_config['parent_dir'] = str(exps_path / f"{trial.number}") # modify for synthetic path
    base_config['eval']['type']['eval_model'] = args.eval_model
    if args.eval_model == "mlp":
        base_config['eval']['T']['normalization'] = "quantile"
        base_config['eval']['T']['cat_encoding'] = "one-hot"

    trial.set_user_attr("config", base_config)

    tddpm_lib.dump_config(base_config, exps_path / 'config.toml')

    subprocess.run(['python', f'{pipeline}', '--config', f'{exps_path / "config.toml"}', '--train', '--change_val'], check=True)

    n_datasets = 5
    score = 0.0

    for sample_seed in range(n_datasets):
        base_config['sample']['seed'] = sample_seed
        tddpm_lib.dump_config(base_config, exps_path / 'config.toml')

        subprocess.run(['python', f'{pipeline}', '--config', f'{exps_path / "config.toml"}', '--sample', '--eval', '--change_val'], check=True)

        report_path = str(Path(base_config['parent_dir']) / f'results_eval.json')
        report = tddpm_lib.load_json(report_path)
        print('report', report)

        score += report['results']['test'].get('acc') or report['results']['test'].get('f1', 0.0)

    shutil.rmtree(exps_path / f"{trial.number}") # modify for synthetic path

    return score / n_datasets

study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=0),
)

study.optimize(objective, n_trials=50, show_progress_bar=True)

best_config_path = parent_path / f'{prefix}_best/config.toml'
best_config = study.best_trial.user_attrs['config']
best_config["parent_dir"] = str(parent_path / f'{prefix}_best/')

os.makedirs(parent_path / f'{prefix}_best', exist_ok=True)
tddpm_lib.dump_config(best_config, best_config_path)
tddpm_lib.dump_json(optuna.importance.get_param_importances(study), parent_path / f'{prefix}_best/importance.json') # modify for optuna

subprocess.run(['python', f'{pipeline}', '--config', f'{best_config_path}', '--train', '--sample'], check=True) # modify

if args.eval_seeds:
    best_exp = str(parent_path / f'{prefix}_best/config.toml')
    subprocess.run(['python', f'{eval_seeds}', '--config', f'{best_exp}', '10', "ddpm", eval_type, args.eval_model, '5'], check=True) # modify
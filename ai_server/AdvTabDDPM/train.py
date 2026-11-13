import sys
import os
# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import tddpm_lib

import torch
import numpy as np
import zero
from tab_ddpm import GaussianMultinomialDiffusion
from utils_train import get_model, make_dataset
from trainer import Trainer
from tab_ddpm import load_pretrained_classifier # add for bdpm

# print("train.py Number of GPUs available:", torch.cuda.device_count())
# print("Current GPU:", torch.cuda.current_device())
# print("GPU name:", torch.cuda.get_device_name(torch.cuda.current_device()))
# print("device: ", torch.device("cuda:0"))

def train(
    parent_dir,
    real_data_path = 'data/higgs-small',
    steps = 1000,
    lr = 0.002,
    weight_decay = 1e-4,
    batch_size = 1024,
    model_type = 'mlp',
    model_params = None,
    num_timesteps = 1000,
    gaussian_loss_type = 'mse',
    scheduler = 'cosine',
    T_dict = None,
    num_numerical_features = 0,
    device = torch.device('cuda:0'),
    seed = 0,
    change_val = False,
    save_path = 'exp/colosseum/ddpm_tune_best/result'
):

    real_data_path = os.path.normpath(real_data_path)
    parent_dir = os.path.normpath(parent_dir)
    
    print("train real_data_path: ", real_data_path)
    print("train parent_dir: ", parent_dir)

    zero.improve_reproducibility(seed)

    T = tddpm_lib.Transformations(**T_dict)

    dataset = make_dataset(
        real_data_path,
        T,
        num_classes=model_params['num_classes'],
        is_y_cond=model_params['is_y_cond'],
        change_val=change_val
    )

    K = np.array(dataset.get_category_sizes('train'))
    if len(K) == 0 or T_dict['cat_encoding'] == 'one-hot':
        K = np.array([0])
    print(K)

    num_numerical_features = dataset.X_num['train'].shape[1] if dataset.X_num is not None else 0
    d_in = np.sum(K) + num_numerical_features
    model_params['d_in'] = d_in
    print(d_in)
    
    print(model_params)
    model = get_model(
        model_type,
        model_params,
        num_numerical_features,
        category_sizes=dataset.get_category_sizes('train')
    )
    model.to(device)

    train_loader = tddpm_lib.prepare_fast_dataloader(dataset, split='train', batch_size=batch_size)
    val_loader = tddpm_lib.prepare_fast_dataloader(dataset, split='val', batch_size=batch_size)

    classifier = load_pretrained_classifier(device=device)
    diffusion = GaussianMultinomialDiffusion(
        num_classes=K,
        num_numerical_features=num_numerical_features,
        denoise_fn=model,
        gaussian_loss_type=gaussian_loss_type,
        num_timesteps=num_timesteps,
        scheduler=scheduler,
        device=device,
        classifier=classifier
    )
    diffusion.to(device)
    # diffusion.train()

    trainer = Trainer(
        diffusion,
        train_iter = train_loader,
        val_iter = val_loader,
        test_iter = None,
        lr=lr,
        weight_decay=weight_decay,
        steps=steps,
        device=device
    )

    # Run the training loop
    trainer.run_loop()

    # Save the train, and validation losses
    trainer.loss_history.to_csv(os.path.join(save_path, 'train_loss.csv'), index=False)
    trainer.val_loss_history.to_csv(os.path.join(save_path, 'validation_loss.csv'), index=False)
    print(f'train and validation loss history save at {save_path}')

    # Save model
    torch.save(diffusion._denoise_fn.state_dict(), os.path.join(parent_dir, "model.pt"))
    torch.save(trainer.ema_model.state_dict(), os.path.join(parent_dir, 'model_ema.pt'))
    torch.save(diffusion.bdpm.state_dict(), os.path.join(parent_dir, "bdpm.pt"))
    print(f'model save at {parent_dir}')

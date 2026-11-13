import torch
import numpy as np
import zero
import os
from tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
from utils_train import get_model, make_dataset
from trainer import Trainer
import tddpm_lib
from scipy.spatial.distance import cdist

def to_good_ohe(ohe, X): # Convert one-hot encoded data back to a good form. ohe is the one-hot encoder, X is the input data.
    indices = np.cumsum([0] + ohe._n_features_outs) # Get cumulative sum of the number of features for each category
    Xres = [] # To store the reconstructed data from one-hot encoded form
    for i in range(1, len(indices)):
        x_ = np.max(X[:, indices[i - 1]:indices[i]], axis=1) # Get max value per row (one-hot encoded)
        t = X[:, indices[i - 1]:indices[i]] - x_.reshape(-1, 1) # Subtract the max value from each one-hot encoded value
        Xres.append(np.where(t >= 0, 1, 0)) # Recreate one-hot encoding (1 if t >= 0)
    return np.hstack(Xres) # Return concatenated result

def round_columns(X_real, X_synth, columns):
    for col in columns:
        uniq = np.unique(X_real[:,col])
        dist = cdist(X_synth[:, col][:, np.newaxis].astype(float), uniq[:, np.newaxis].astype(float))
        X_synth[:, col] = uniq[dist.argmin(axis=1)]
    return X_synth

from train import load_pretrained_classifier # add for bdpm
def sample(
    parent_dir,
    real_data_path='data/higgs-small',
    batch_size=2000,
    num_samples=0,
    model_type='mlp',
    model_params=None,
    model_path=None,
    num_timesteps=1000,
    steps=1000,
    gaussian_loss_type='mse',
    scheduler='cosine',
    T_dict=None,
    num_numerical_features=0,
    disbalance=None,
    device=torch.device('cuda:0'),
    seed=0,
    change_val=False,
    save_path='exp/colosseum/ddpm_tune_best/result',
    do_tune=False
):
    print("sample real_data_path: ", real_data_path)
    print("sample parent_dir: ", parent_dir)
    print("model_path: ", model_path)

    zero.improve_reproducibility(seed)

    T = tddpm_lib.Transformations(**T_dict)
    D = make_dataset(
        real_data_path,
        T,
        num_classes=model_params['num_classes'],
        is_y_cond=model_params['is_y_cond'],
        change_val=change_val
    )

    K = np.array(D.get_category_sizes('train'))
    if len(K) == 0 or T_dict['cat_encoding'] == 'one-hot':
        K = np.array([0])

    num_numerical_features_ = D.X_num['train'].shape[1] if D.X_num is not None else 0
    d_in = np.sum(K) + num_numerical_features_
    model_params['d_in'] = int(d_in)
    model = get_model(
        model_type,
        model_params,
        num_numerical_features_,
        category_sizes=D.get_category_sizes('train')
    )

    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    classifier = load_pretrained_classifier() # add for bdpm
    diffusion = GaussianMultinomialDiffusion(
        K,
        num_numerical_features=num_numerical_features_,
        denoise_fn=model,
        num_timesteps=num_timesteps,
        gaussian_loss_type=gaussian_loss_type,
        scheduler=scheduler,
        device=device,
        classifier=classifier
    )

    diffusion.to(device)

    diffusion.eval()

    if not do_tune:
        test_loader = tddpm_lib.prepare_fast_dataloader(D, split='test', batch_size=batch_size)
        trainer = Trainer(
            diffusion,
            train_iter = None,
            val_iter = None,
            test_iter = test_loader,
            lr = 0.0, 
            weight_decay = 0.0,
            steps = steps,
            device=device
        )
        trainer.run_loop(train_mode = False)

        trainer.test_loss_history.to_csv(os.path.join(save_path, 'test_loss.csv'), index=False)

    # ===== Sample the synthetic data =====
    _, empirical_class_dist = torch.unique(torch.from_numpy(D.y['train']), return_counts=True) # Compute class distribution # computer the traffic type ratio
    # empirical_class_dist = empirical_class_dist.float() + torch.tensor([-5000., 10000.]).float()
    
    print(f'disbalance is {disbalance}')
    if disbalance == 'fix': # If disbalance is specified, handle class imbalance
        empirical_class_dist[0], empirical_class_dist[1] = empirical_class_dist[1], empirical_class_dist[0] # Swap the two most frequent classes
        x_gen, y_gen = diffusion.sample_all(num_samples, batch_size, empirical_class_dist.float(), ddim=False) # Generate samples

    elif disbalance == 'fill':
        ix_major = empirical_class_dist.argmax().item() # Get the index of the majority class
        val_major = empirical_class_dist[ix_major].item() # Get the value of the majority class
        x_gen, y_gen = [], [] # Initialize lists for generated data
        for i in range(empirical_class_dist.shape[0]):
            if i == ix_major:
                continue # Skip the majority class
            distrib = torch.zeros_like(empirical_class_dist) # Create an empty distribution
            distrib[i] = 1 # Set the current class to 1 in the distribution
            num_samples = val_major - empirical_class_dist[i].item() # Calculate how many samples are needed to balance the class
            x_temp, y_temp = diffusion.sample_all(num_samples, batch_size, distrib.float(), ddim=False) # Sample for the current class
            x_gen.append(x_temp)
            y_gen.append(y_temp)
        
        x_gen = torch.cat(x_gen, dim=0) # Concatenate generated samples
        y_gen = torch.cat(y_gen, dim=0) # Concatenate generated labels

    else:
        x_gen, y_gen = diffusion.sample_all(num_samples, 1024, empirical_class_dist.float(), ddim=False) # modify for bdpm
        # x_gen, y_gen = diffusion.sample_all(num_samples, batch_size, empirical_class_dist.float(), ddim=False) # Default sampling behavior

    X_gen, y_gen = x_gen.cpu().numpy(), y_gen.cpu().numpy()
    # X_gen, y_gen = x_gen.numpy(), y_gen.numpy() # modify for bdpm # Convert generated data to NumPy arrays

    # Handle one-hot encoded categorical features 
    num_numerical_features = num_numerical_features + int(D.is_regression and not model_params["is_y_cond"]) # Adjust number of numerical features
    X_num_ = X_gen # Get the generated numerical features

    if num_numerical_features < X_gen.shape[1]: # If there are categorical features, handle one-hot encoding and inverse transformations
        np.save(os.path.join(parent_dir, 'X_cat_unnorm'), X_gen[:, num_numerical_features:]) # Save unnormalized categorical features
        # _, _, cat_encoder = tddpm_lib.cat_encode({'train': X_cat_real}, T_dict['cat_encoding'], y_real, T_dict['seed'], True)
        if T_dict['cat_encoding'] == 'one-hot': # If categorical features were one-hot encoded
            X_gen[:, num_numerical_features:] = to_good_ohe(D.cat_transform.steps[0][1], X_num_[:, num_numerical_features:]) # Convert back to good one-hot form
        X_cat = D.cat_transform.inverse_transform(X_gen[:, num_numerical_features:]) # Apply inverse transformation to categorical features

    if num_numerical_features_ != 0: # Normalize numerical features and save them
        # _, normalize = tddpm_lib.normalize({'train' : X_num_real}, T_dict['normalization'], T_dict['seed'], True)
        np.save(os.path.join(parent_dir, 'X_num_unnorm'), X_gen[:, :num_numerical_features]) # Save unnormalized numerical features
        X_num_ = D.num_transform.inverse_transform(X_gen[:, :num_numerical_features]) # Apply inverse transformation to numerical features
        X_num = X_num_[:, :num_numerical_features] # Extract the numerical features

        X_num_real = np.load(os.path.join(real_data_path, "X_num_train.npy"), allow_pickle=True) # Load real numerical features for comparison
        disc_cols = [] # List to store discrete columns
        for col in range(X_num_real.shape[1]):
            uniq_vals = np.unique(X_num_real[:, col]) # Get unique values of each column
            if len(uniq_vals) <= 32 and ((uniq_vals - np.round(uniq_vals)) == 0).all(): # If values are discrete
                disc_cols.append(col) # Add to discrete columns list
        print("Discrete cols:", disc_cols)
        if model_params['num_classes'] == 0: # If regression
            y_gen = X_num[:, 0] # First column is the target variable
            X_num = X_num[:, 1:] # Remove the target column from numerical features
        if len(disc_cols):
            X_num = round_columns(X_num_real, X_num, disc_cols) # Round discrete columns

    # Save the processed data
    if num_numerical_features != 0: 
        print("Num shape: ", X_num.shape)
        np.save(os.path.join(parent_dir, 'X_num_train'), X_num)
        print(f'X_num_train {X_num} save to {parent_dir}')
    if num_numerical_features < X_gen.shape[1]:
        np.save(os.path.join(parent_dir, 'X_cat_train'), X_cat)
        print(f'X_cat_train {X_cat} save to {parent_dir}')
    np.save(os.path.join(parent_dir, 'y_train'), y_gen)
    print(f'y_train {y_gen} save to {parent_dir}')
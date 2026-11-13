import numpy as np
import os
import tddpm_lib
from tab_ddpm.modules import MLPDiffusion, ResNetDiffusion

def get_model(
    model_name, # Name of the model to use ('mlp', 'resnet', etc.)
    model_params, # Dictionary of model hyperparameters
    n_num_features, # Number of numerical features in the dataset
    category_sizes # Sizes of categorical feature categories
): 
    print(model_name) # Print the model name (for debugging purposes)
    if model_name == 'mlp': # Check if model_name is 'mlp' and create an instance of MLPDiffusion
        model = MLPDiffusion(**model_params) # Instantiate the MLPDiffusion model
    elif model_name == 'resnet': # Check if model_name is 'resnet' and create an instance of ResNetDiffusion
        model = ResNetDiffusion(**model_params) # Instantiate the ResNetDiffusion model
    else: # Raise an error if the model name is unknown
        raise "Unknown model!" # Handle unknown model case
    return model # Return the instantiated model

def update_ema(target_params, source_params, rate=0.999):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.
    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params): # Loop through pairs of target and source parameters
        targ.detach().mul_(rate).add_(src.detach(), alpha=1 - rate) # Detach gradients and update target parameters with EMA rule

"""
Concatenate the y labels to X if needed (for conditional models).

:param X: Input feature matrix.
:param y: Label array to concatenate.
:return: Concatenated matrix of X and y.
"""
def concat_y_to_X(X, y):
    if X is None: # If X is None, return y as a reshaped column matrix
        return y.reshape(-1, 1)
    return np.concatenate([y.reshape(-1, 1), X], axis=1) # Concatenate y as a new column to X along axis 1

def make_dataset(
    data_path: str, # Path to the dataset
    T: tddpm_lib.Transformations, # Data transformation settings
    num_classes: int, # Number of output classes for classification
    is_y_cond: bool, # Whether y is used as a conditional input
    change_val: bool # Whether the validation set should be altered
):
    # classification
    if num_classes > 0: # If there are classes (classification task)
        X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy')) or not is_y_cond else None # Prepare storage for categorical features and labels
        X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) else None # Prepare storage for numerical features and labels
        y = {} # Storage for labels

        for split in ['train', 'val', 'test']: # Iterate through data splits (train, val, test)
            X_num_t, X_cat_t, y_t = tddpm_lib.read_pure_data(data_path, split) # Read raw data for the current split
            if X_num is not None: # If numerical features exist, store them for the current split
                X_num[split] = X_num_t
            if not is_y_cond: # If not using conditional y, concatenate y to X_cat
                X_cat_t = concat_y_to_X(X_cat_t, y_t)
            if X_cat is not None: # If categorical features exist, store them for the current split
                X_cat[split] = X_cat_t
            y[split] = y_t # Store labels for the current split
    else: # If regression task (num_classes = 0)
        # regression
        X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy')) else None # Prepare storage for categorical features and labels
        X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) or not is_y_cond else None # Prepare storage for numerical features and labels
        y = {} # Storage for labels

        for split in ['train', 'val', 'test']: # Iterate through data splits (train, val, test)
            X_num_t, X_cat_t, y_t = tddpm_lib.read_pure_data(data_path, split) # Read raw data for the current split
            if not is_y_cond: # If not using conditional y, concatenate y to X_num
                X_num_t = concat_y_to_X(X_num_t, y_t)
            if X_num is not None: # If numerical features exist, store them for the current split
                X_num[split] = X_num_t
            if X_cat is not None: # If categorical features exist, store them for the current split
                X_cat[split] = X_cat_t
            y[split] = y_t # Store labels for the current split

    info = tddpm_lib.load_json(os.path.join(data_path, 'info.json')) # Load additional information from the dataset's info.json

    # # add
    # # Calculate n_classes
    # # Concatenate all label arrays
    # all_labels = np.concatenate([y['train'], y['val'], y['test']])
    # # Find unique classes
    # unique_classes = np.unique(all_labels)
    # # Get the number of unique classes
    # n_classes = len(unique_classes)
    # info['n_classes'] = n_classes

    D = tddpm_lib.Dataset( # Create a Dataset object using loaded data and transformations
        X_num, # Numerical features
        X_cat, # Categorical features
        y, # Labels
        y_info={}, # Additional label information (optional)
        task_type=tddpm_lib.TaskType(info['task_type']), # Task type (classification/regression)
        n_classes=info.get('n_classes') # Number of classes (optional)
    )

    if change_val: # Optionally alter the validation set
        D = tddpm_lib.change_val(D)
    
    return tddpm_lib.transform_dataset(D, T, None) # Apply transformations to the dataset
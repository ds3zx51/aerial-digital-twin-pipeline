import sys
import os
# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from copy import deepcopy
import torch
import numpy as np
from utils_train import update_ema
import pandas as pd
import torch.nn.functional as F # add for new loss


def _loss_to_float(value):
    """Convert a scalar tensor or Python numeric value to float."""
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)

from classifier_train import TRACTOR_model, restore_train_config # add for new loss
import json # add for new loss

class Trainer:
    def __init__(self, diffusion, train_iter, val_iter, test_iter, lr, weight_decay, steps, device=torch.device('cuda:0')): 
        self.diffusion = diffusion
        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()
        
        self.train_iter = train_iter
        self.val_iter = val_iter  # Add validation iterator # add for val loss
        self.test_iter = test_iter
        self.steps = steps
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.loss_history = pd.DataFrame(columns=['step', 'train_mloss', 'train_gloss', 'train_loss'])
        self.val_loss_history = pd.DataFrame(columns=['step', 'val_mloss', 'val_gloss', 'val_loss']) # add for val loss
        self.test_loss_history = pd.DataFrame(columns=['step', 'test_mloss', 'test_gloss', 'test_loss']) # add for test loss
        self.log_every = 100
        self.print_every = 500
        self.ema_every = 1000

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    # add for val loss
    # add for test loss
    def _eval_step(self, x, out_dict, step, total_steps):
        """
        Performs a forward pass for evaluation and returns multinomial and Gaussian losses.
        """
        x = x.to(self.device)
        for k in out_dict:
            out_dict[k] = out_dict[k].long().to(self.device)
        
        # Forward pass without gradient computation
        with torch.no_grad():
            loss_multi, loss_gauss = self.diffusion.mixed_loss(x, out_dict, step, total_steps)
        
        return loss_multi, loss_gauss
    
    # def _run_step(self, x, out_dict, step):
    def _run_step(self, x, out_dict, step, total_steps, save_num_list=None, save_cat_list=None, save_y_list=None):
        x = x.to(self.device)
        for k in out_dict:
            out_dict[k] = out_dict[k].long().to(self.device)
        self.optimizer.zero_grad()

        # 判斷是否處於最後 1/3 訓練步驟
        saving_boundary_data = step > total_steps * (1999 / 2000)

        # 新版 mixed_loss()：支援 saving_boundary_data
        result = self.diffusion.mixed_loss(x, out_dict, step=step, total_steps=total_steps, do_train=True, saving_boundary_data=saving_boundary_data)

        if saving_boundary_data:
            print(f'saving_boundary_data is {saving_boundary_data}')
            loss_multi, loss_gauss, loss_cd, x_adv_num, x_adv_cat, y = result
            save_num_list.append(x_adv_num)
            save_cat_list.append(x_adv_cat)
            save_y_list.append(y)
        else:
            loss_multi, loss_gauss, loss_cd = result
        '''
        # modify for new loss
        # ========== Step 1: DDPM Loss ==========
        # loss_multi, loss_gauss, x0_hat = self.diffusion.mixed_loss(x, out_dict, return_x0=True)
        loss_multi, loss_gauss, loss_cd = self.diffusion.mixed_loss(x, out_dict, do_train=True)
        loss_ddpm = loss_multi + loss_gauss

        # # ========== Step 2: Classifier Prediction ==========
        # with open(os.path.join('train_log1/slice__model_train_result', "train_config.json"), "r") as f:
        #     train_config = restore_train_config(json.load(f))
        # # Rebuild the model and load the weight
        # global_model = train_config['global_model']
        # Nclass       = train_config['Nclass']
        # num_feats    = train_config['num_feats']
        # slice_len    = train_config['slice_len']
        # pos_enc      = train_config.get('pos_enc', False)

        # # Load the model
        # model, loss_fn = TRACTOR_model(Nclass, global_model, num_feats, slice_len, pos_enc)
        # # Load the trained model weights
        # model.load_state_dict(torch.load(os.path.join('train_log1/slice__model_train_result', "model.1.trans_v1.pt"), map_location='cuda:0')['model_state_dict'])
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model.to(device)
        # model.eval()
        # with torch.no_grad():
        #     x0_hat_classifier = x0_hat
        #     x0_hat_classifier = x0_hat_classifier[:, torch.arange(x0_hat.shape[1]) != x0_hat_classifier.shape[1] - 2]  # 移除倒數第二欄 'slice_id'
        #     if x0_hat_classifier.ndim == 2:
        #         x0_hat_classifier = x0_hat_classifier.unsqueeze(0) # [seq_len, batch_size, embed_dim] [1, B, F]
        #     # Transpose to batch-first: [B, L, D]
        #     x0_hat_classifier = x0_hat_classifier.permute(1, 0, 2)  # Now shape is [B, L, D]
        #     logits = model(x0_hat_classifier)  # x0_hat: generated sample
        #     probs = F.softmax(logits, dim=1)
        #     y_true = out_dict['y']
        #     y_pred = torch.argmax(probs, dim=1)
        # correct_class_probs = probs.gather(1, y_true.unsqueeze(1)).squeeze(1)
        # sum_other_probs = probs.sum(dim=1) - correct_class_probs
        # cd = correct_class_probs - sum_other_probs  # Confidence Difference
        # # print(f'y_true {y_true}, y_predict {y_pred}, correct_class_probs {correct_class_probs}, sum_other_probs {sum_other_probs}, cd {cd}')

        # # ========== Step 3: CD Loss ==========
        # delta = 0.2
        # beta = 0.8

        # is_boundary = (cd.abs() <= delta).float()
        # is_general_correct = (cd > beta).float()
        # is_wrong = (y_pred != y_true).float()

        # cd_loss_boundary = is_boundary * cd**2
        # cd_loss_general = is_general_correct * F.relu(beta - cd)**2 #(cd - beta)**2
        # cd_loss_wrong = is_wrong * 1
        # # loss_cd = (cd_loss_boundary + cd_loss_general + cd_loss_wrong).mean()
        # # loss_cd = (cd_loss_boundary + cd_loss_general + cd_loss_wrong).mean()
        # # print(f'cd_loss_boundary {cd_loss_boundary}, cd_loss_general {cd_loss_general}, cd_loss_wrong {cd_loss_wrong}, loss_cd {loss_cd}')
        # loss_cd = cd_loss_boundary.mean()

        # # ========== Step 4: Perturbation Bound Loss ==========
        # eps = 0.2
        # x_norm = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)
        # x0_hat_norm = (x0_hat - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)
        # diff = (x0_hat_norm - x_norm).abs().mean(dim=1)  # L1 norm over features and slice dim

        # loss_bound_boundary = is_boundary * (torch.clamp(diff, max=eps) / eps)
        # loss_bound_general = (1 - is_boundary) * F.mse_loss(diff, torch.zeros_like(diff), reduction='none')#(diff ** 2)
        # # loss_bound = (loss_bound_boundary + loss_bound_general).mean()
        # loss_bound = loss_bound_boundary.mean()
        '''
        # ========== Step 5: Combine ==========
        lambda_ddpm = 0
        lambda_cd = 1 #0.001
        lambda_bound = 0
        loss_bound = 0
        loss_ddpm = loss_multi + loss_gauss

        loss_total = lambda_ddpm * loss_ddpm + lambda_cd * loss_cd + lambda_bound * loss_bound
        # if step < 5000:
        #     loss_total = loss_multi + loss_gauss  # Warm-up: 僅生成
        # else:
        #     loss_total = loss_multi + loss_gauss + lambda_cd * loss_cd + lambda_bound * loss_bound
        
        loss_total.backward()
        self.optimizer.step()

        return loss_multi, loss_gauss, lambda_cd * loss_cd, lambda_bound * loss_bound # modify for new loss

    # add for test loss
    def eval_loop(self):  # Evaluation loop
        step = 0
        total_mloss = 0.0
        total_gloss = 0.0
        total_samples = 0

        while step < self.steps:
            with torch.no_grad():  # Disable gradients for evaluation
                x, out_dict = next(self.test_iter)
                out_dict = {'y': out_dict}
                loss_multi, loss_gauss = self._eval_step(x, out_dict)
                
                total_mloss += loss_multi.item() * len(x)
                total_gloss += loss_gauss.item() * len(x)
                total_samples += len(x)

            if (step + 1) % self.log_every == 0:
                test_mloss = total_mloss / total_samples if total_samples > 0 else 0.0
                test_gloss = total_gloss / total_samples if total_samples > 0 else 0.0
                test_loss = np.around(test_mloss + test_gloss, 4)
                self.test_loss_history.loc[len(self.test_loss_history)] = [step + 1, test_mloss, test_gloss, test_loss]
                if (step + 1) % self.print_every == 0:
                    print(f'[Eval] Step {(step + 1)}/{self.steps} MLoss: {test_mloss} GLoss: {test_gloss} Sum: {test_loss}')
                total_mloss, total_gloss, total_samples = 0.0, 0.0, 0

            step += 1

    def run_loop(self, train_mode = True):
        step = 0
        if train_mode:
            curr_loss_multi = 0.0
            curr_loss_gauss = 0.0
            curr_loss_cd = 0.0 # add for new loss
            curr_loss_bound = 0.0 # add for new loss
            curr_count = 0

            # 儲存 perturbed 資料用的 list
            save_num_list = []
            save_cat_list = []
            save_y_list = []

        # add for val loss
        total_mloss = 0.0
        total_gloss = 0.0
        total_samples = 0

        total_steps = self.steps # add for cdnear0

        while step < self.steps:
            # === Training step ===
            if train_mode:
                self.diffusion.train()
                x, out_dict = next(self.train_iter)
                out_dict = {'y': out_dict}
                # batch_loss_multi, batch_loss_gauss, batch_loss_cd, batch_loss_bound = self._run_step(x, out_dict, step) # modify for cdnear0 # modify for new loss
                batch_loss_multi, batch_loss_gauss, batch_loss_cd, batch_loss_bound = \
                        self._run_step(x, out_dict, step, total_steps, save_num_list, save_cat_list, save_y_list)

                self._anneal_lr(step)
                # Accumulate scalar losses for logging
                curr_count += 1
                curr_loss_multi += _loss_to_float(batch_loss_multi)
                curr_loss_gauss += _loss_to_float(batch_loss_gauss)
                curr_loss_cd += _loss_to_float(batch_loss_cd)
                curr_loss_bound += _loss_to_float(batch_loss_bound)

                # Write one record every log_every steps
                if (step + 1) % self.log_every == 0 or (step + 1) == self.steps:
                    mloss = curr_loss_multi / max(curr_count, 1)
                    gloss = curr_loss_gauss / max(curr_count, 1)
                    cdloss = curr_loss_cd / max(curr_count, 1)
                    bloss = curr_loss_bound / max(curr_count, 1)

                    self.loss_history.loc[len(self.loss_history)] = [
                        step + 1,
                        mloss,
                        gloss,
                        cdloss,
                    ]

                    print(
                        f'[TrainLog] Step {step + 1}/{self.steps} '
                        f'MLoss={mloss:.6f} '
                        f'GLoss={gloss:.6f} '
                        f'BDPMLoss={cdloss:.6f} '
                        f'BoundLoss={bloss:.6f}'
                    )

                    curr_loss_multi = 0.0
                    curr_loss_gauss = 0.0
                    curr_loss_cd = 0.0
                    curr_loss_bound = 0.0
                    curr_count = 0
            else: # modify for new new loss
                # === Validation and Test step ===
                # add for val and test loss
                self.diffusion.eval()  # Set model to evaluation mode
                # with torch.no_grad():  # Disable gradients for evaluation
                if train_mode:
                    x, out_dict = next(self.val_iter)
                else:
                    x, out_dict = next(self.test_iter)
                out_dict = {'y': out_dict}
                loss_multi, loss_gauss = self._eval_step(x, out_dict, step, total_steps)
                
                total_mloss += loss_multi#.item() * len(x)
                total_gloss += loss_gauss#.item() * len(x)
                total_samples += len(x)

            # # === Logging ===
            # # Calculate and log validation and test loss
            # if (step + 1) % self.log_every == 0:
            #     if train_mode:
            #         mloss = np.around(curr_loss_multi / curr_count, 4)
            #         gloss = np.around(curr_loss_gauss / curr_count, 4)
            #         cdloss = np.around(curr_loss_cd / curr_count, 4)
            #         bloss = np.around(curr_loss_bound / curr_count, 4)
            #         print(f'[Train] Step {(step + 1)}/{self.steps}')
            #         if (step + 1) % self.print_every == 0:
            #             print(f'[Train] Step {(step + 1)}/{self.steps} MLoss: {mloss} GLoss: {gloss} cdLoss: {cdloss} boundLoss: {bloss} Sum: {mloss + gloss + cdloss + bloss}')
            #         self.loss_history.loc[len(self.loss_history)] = [step + 1, mloss, gloss, mloss + gloss]
            #         curr_loss_multi, curr_loss_gauss, curr_count = 0.0, 0.0, 0
            #         curr_loss_cd = 0.0 # add for bdpm loss

            #     # add for val loss
            #     eval_mloss = total_mloss / total_samples if total_samples > 0 else 0.0
            #     eval_gloss = total_gloss / total_samples if total_samples > 0 else 0.0
            #     eval_loss = np.around(eval_mloss + eval_gloss, 4)
            #     if train_mode:
            #         self.val_loss_history.loc[len(self.val_loss_history)] = [step + 1, eval_mloss, eval_gloss, eval_loss]
            #     else:
            #         self.test_loss_history.loc[len(self.test_loss_history)] = [step + 1, eval_mloss, eval_gloss, eval_loss]
            #     if (step + 1) % self.print_every == 0:
            #         print(f'[Eval] Step {(step + 1)}/{self.steps} MLoss: {eval_mloss} GLoss: {eval_gloss} Sum: {eval_loss}')
            #     total_mloss, total_gloss, total_samples = 0.0, 0.0, 0
            
            if train_mode:
                update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())
            
            step += 1
        
        if train_mode:
            # === 訓練結束後，儲存 perturbed 資料 ===
            print('save_num_list: ', save_num_list)
            if train_mode and save_num_list:
                import numpy as np, os, torch
                from collections import Counter

                save_dir = 'exp/colosseum/ddpm_tune_best'
                os.makedirs(save_dir, exist_ok=True)

                X_num = torch.cat(save_num_list, dim=0)
                X_cat = torch.cat(save_cat_list, dim=0)
                y = torch.cat(save_y_list, dim=0)

                total_samples = X_num.shape[0]
                print(f"[i] Total perturbed samples collected: {total_samples}")

                max_samples = 50000

                if total_samples > max_samples:
                    print(f"[i] Exceeds max limit ({max_samples}), performing balanced sampling...")

                    # Balanced sampling
                    y_np = y.numpy()
                    num_classes = len(np.unique(y_np))
                    samples_per_class = max_samples // num_classes

                    selected_indices = []
                    for cls in range(num_classes):
                        cls_indices = np.where(y_np == cls)[0]
                        if len(cls_indices) >= samples_per_class:
                            chosen = np.random.choice(cls_indices, samples_per_class, replace=False)
                        else:
                            # 若某類別樣本太少，就全部取出
                            chosen = cls_indices
                        selected_indices.extend(chosen)

                    # 如果還沒滿 max_samples，隨機補齊
                    remaining = max_samples - len(selected_indices)
                    if remaining > 0:
                        other_indices = list(set(range(len(y_np))) - set(selected_indices))
                        fill_indices = np.random.choice(other_indices, remaining, replace=False)
                        selected_indices.extend(fill_indices)

                    # 最後轉成排序後的 tensor index（為了一致性）
                    selected_indices = np.array(selected_indices)
                    np.random.shuffle(selected_indices)  # 不一定要排序，這裡隨機打亂即可

                    X_num = X_num[selected_indices]
                    X_cat = X_cat[selected_indices]
                    y = y[selected_indices]

                # 顯示類別分布統計
                y_np_final = y.numpy()
                print(f"[✔] Final y distribution: {dict(Counter(y_np_final))}")

                # 儲存
                np.save(os.path.join(save_dir, 'X_num_train.npy'), X_num.numpy())
                np.save(os.path.join(save_dir, 'X_cat_train.npy'), X_cat.numpy())
                np.save(os.path.join(save_dir, 'y_train.npy'), y_np_final)

                print(f"[✔] Saved {X_num.shape[0]} balanced samples to {save_dir}")

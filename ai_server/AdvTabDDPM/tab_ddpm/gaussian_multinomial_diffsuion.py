"""
Based on https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
and https://github.com/ehoogeboom/multinomial_diffusion
"""

import torch.nn.functional as F
import torch
import math

import numpy as np
from .utils import *

from classifier_train import TRACTOR_model, restore_train_config # add for new new loss
import os # add for new new loss
import json # add for new new loss

# add for new new loss
def soft_argmax(logits, beta=10):
    """soft differentiable approximation to argmax"""
    softmax = F.softmax(logits * beta, dim=-1)
    return (softmax * torch.arange(logits.size(-1), device=logits.device)).sum(dim=-1)


# add for bdpm loss
def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

from torch import nn
class BDPM_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dim_t=128):
        super().__init__()
        self.dim_t = dim_t
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )
        self.proj = nn.Linear(input_dim, dim_t)
        self.mlp = nn.Sequential(
            nn.Linear(dim_t, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, t):
        emb = self.time_embed(timestep_embedding(t, self.dim_t))
        x = self.proj(x) + emb
        return self.mlp(x)
    

"""
Based in part on: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/5989f4c77eafcdc6be0fb4739f0f277a6dd7f7d8/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L281
"""
eps = 1e-8

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

# add for bdpm
def load_pretrained_classifier(device='cuda:0'):
    with open(os.path.join('classifier_train_result/slice_model_train_result', "train_config.json"), "r") as f:
        train_config = restore_train_config(json.load(f))

    global_model = train_config['global_model']
    Nclass = train_config['Nclass']
    num_feats = train_config['num_feats']
    slice_len = train_config['slice_len']
    pos_enc = train_config.get('pos_enc', False)

    clf_model, _ = TRACTOR_model(Nclass, global_model, num_feats, slice_len, pos_enc)
    clf_model.load_state_dict(torch.load(
        os.path.join('classifier_train_result/slice_model_train_result', "model.1.trans_v1.pt"),
        map_location=device)['model_state_dict'])

    clf_model = clf_model.to(device)
    clf_model.eval()
    for p in clf_model.parameters():
        p.requires_grad = False
    return clf_model
def preprocess_for_classifier(x_hat, num_feats, num_classes, device):
    x_hat_num = x_hat[:, :num_feats]
    x_hat_cat_logits = x_hat[:, num_feats:]

    if x_hat_cat_logits.shape[1] > 0:
        split_logits = torch.split(x_hat_cat_logits, list(num_classes), dim=1)
        x_cat_hat = torch.stack([soft_argmax(logit) for logit in split_logits], dim=1)
    else:
        x_cat_hat = torch.empty((x_hat.shape[0], 0), device=device)

    x0_clf = torch.cat([x_hat_num, x_cat_hat], dim=1)
    x0_clf = x0_clf[:, torch.arange(x0_clf.shape[1]) != x0_clf.shape[1] - 2]  # remove slice_id

    if x0_clf.ndim == 2:
        x0_clf = x0_clf.unsqueeze(0)  # [1, B, F]
    return x0_clf.permute(1, 0, 2)  # [B, L, D]

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler # add for npy classify
class GaussianMultinomialDiffusion(torch.nn.Module):
    def __init__(
            self,
            num_classes: np.array,
            num_numerical_features: int,
            denoise_fn,
            num_timesteps=1000,
            gaussian_loss_type='mse',
            gaussian_parametrization='eps',
            multinomial_loss_type='vb_stochastic',
            parametrization='x0',
            scheduler='cosine',
            device=torch.device('cpu'),
            classifier=None # add for bdpm
        ):

        super(GaussianMultinomialDiffusion, self).__init__()
        assert multinomial_loss_type in ('vb_stochastic', 'vb_all')
        assert parametrization in ('x0', 'direct')

        if multinomial_loss_type == 'vb_all':
            print('Computing the loss using the bound on _all_ timesteps.'
                  ' This is expensive both in terms of memory and computation.')

        self.num_numerical_features = num_numerical_features
        self.num_classes = num_classes # it as a vector [K1, K2, ..., Km]
        self.num_classes_expanded = torch.from_numpy(
            np.concatenate([num_classes[i].repeat(num_classes[i]) for i in range(len(num_classes))])
        ).to(device)

        self.slices_for_classes = [np.arange(self.num_classes[0])]
        offsets = np.cumsum(self.num_classes)
        for i in range(1, len(offsets)):
            self.slices_for_classes.append(np.arange(offsets[i - 1], offsets[i]))
        self.offsets = torch.from_numpy(np.append([0], offsets)).to(device)

        self._denoise_fn = denoise_fn
        self.gaussian_loss_type = gaussian_loss_type
        self.gaussian_parametrization = gaussian_parametrization
        self.multinomial_loss_type = multinomial_loss_type
        self.num_timesteps = num_timesteps
        self.parametrization = parametrization
        self.scheduler = scheduler
        self.classifier = classifier # add for bdpm
        self.bdpm = BDPM_MLP(num_numerical_features).to(device) # add for bdpm loss
        self.cd_guidance_scale = 1.0 # add for bdpm # 或你要的數值，例如 0.5, 2.0 等
        self.scaler_fitted = False # add for npy classify
        self.scaler = StandardScaler() # add for npy classify
        self.delta_accumulate = 0.0 # add for record delta
        self.delta_count = 0 # add for record delta

        alphas = 1. - get_named_beta_schedule(scheduler, num_timesteps)
        alphas = torch.tensor(alphas.astype('float64'))
        betas = 1. - alphas

        log_alpha = np.log(alphas)
        log_cumprod_alpha = np.cumsum(log_alpha)

        log_1_min_alpha = log_1_min_a(log_alpha)
        log_1_min_cumprod_alpha = log_1_min_a(log_cumprod_alpha)

        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.tensor(np.append(1.0, alphas_cumprod[:-1]))
        alphas_cumprod_next = torch.tensor(np.append(alphas_cumprod[1:], 0.0))
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
        sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)

        # Gaussian diffusion

        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.from_numpy(
            np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
        ).float().to(device)
        self.posterior_mean_coef1 = (
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).float().to(device)
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev)
            * np.sqrt(alphas.numpy())
            / (1.0 - alphas_cumprod)
        ).float().to(device)

        assert log_add_exp(log_alpha, log_1_min_alpha).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_alpha, log_1_min_cumprod_alpha).abs().sum().item() < 1e-5
        assert (np.cumsum(log_alpha) - log_cumprod_alpha).abs().sum().item() < 1.e-5

        # Convert to float32 and register buffers.
        self.register_buffer('alphas', alphas.float().to(device))
        self.register_buffer('log_alpha', log_alpha.float().to(device))
        self.register_buffer('log_1_min_alpha', log_1_min_alpha.float().to(device))
        self.register_buffer('log_1_min_cumprod_alpha', log_1_min_cumprod_alpha.float().to(device))
        self.register_buffer('log_cumprod_alpha', log_cumprod_alpha.float().to(device))
        self.register_buffer('alphas_cumprod', alphas_cumprod.float().to(device))
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.float().to(device))
        self.register_buffer('alphas_cumprod_next', alphas_cumprod_next.float().to(device))
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod.float().to(device))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod.float().to(device))
        self.register_buffer('sqrt_recip_alphas_cumprod', sqrt_recip_alphas_cumprod.float().to(device))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', sqrt_recipm1_alphas_cumprod.float().to(device))

        self.register_buffer('Lt_history', torch.zeros(num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(num_timesteps))
    
    # Gaussian part
    def gaussian_q_mean_variance(self, x_start, t):
        mean = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_1_min_cumprod_alpha, t, x_start.shape
        )
        return mean, variance, log_variance
    
    def gaussian_q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )
    
    def gaussian_q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def gaussian_p_mean_variance(
        self, model_output, x, t, clip_denoised=False, denoised_fn=None, model_kwargs=None
    ):
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)

        model_variance = torch.cat([self.posterior_variance[1].unsqueeze(0).to(x.device), (1. - self.alphas)[1:]], dim=0)
        # model_variance = self.posterior_variance.to(x.device)
        model_log_variance = torch.log(model_variance)

        model_variance = extract(model_variance, t, x.shape)
        model_log_variance = extract(model_log_variance, t, x.shape)


        if self.gaussian_parametrization == 'eps':
            pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
        elif self.gaussian_parametrization == 'x0':
            pred_xstart = model_output
        else:
            raise NotImplementedError
            
        model_mean, _, _ = self.gaussian_q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        ), f'{model_mean.shape}, {model_log_variance.shape}, {pred_xstart.shape}, {x.shape}'

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }
    
    def _vb_terms_bpd(
        self, model_output, x_start, x_t, t, clip_denoised=False, model_kwargs=None
    ):
        true_mean, _, true_log_variance_clipped = self.gaussian_q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.gaussian_p_mean_variance(
            model_output, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"], "out_mean": out["mean"], "true_mean": true_mean}
    
    def _prior_gaussian(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.gaussian_q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)
    
    def _gaussian_loss(self, model_out, x_start, x_t, t, noise, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}

        terms = {}
        # print('gaussian_loss_type: ', self.gaussian_loss_type) # mse
        if self.gaussian_loss_type == 'mse':
            terms["loss"] = mean_flat((noise - model_out) ** 2)
        elif self.gaussian_loss_type == 'kl':
            terms["loss"] = self._vb_terms_bpd(
                model_output=model_out,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]


        return terms['loss']
    
    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
    
    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def gaussian_p_sample(
        self,
        model_out,
        x,
        t,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None,
    ):
        out = self.gaussian_p_mean_variance(
            model_out,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = torch.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0

        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    # Multinomial part

    def multinomial_kl(self, log_prob1, log_prob2):
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    def q_pred_one_timestep(self, log_x_t, t):
        log_alpha_t = extract(self.log_alpha, t, log_x_t.shape)
        log_1_min_alpha_t = extract(self.log_1_min_alpha, t, log_x_t.shape)

        # alpha_t * E[xt] + (1 - alpha_t) 1 / K
        log_probs = log_add_exp(
            log_x_t + log_alpha_t,
            log_1_min_alpha_t - torch.log(self.num_classes_expanded)
        )

        return log_probs

    def q_pred(self, log_x_start, t):
        log_cumprod_alpha_t = extract(self.log_cumprod_alpha, t, log_x_start.shape)
        log_1_min_cumprod_alpha = extract(self.log_1_min_cumprod_alpha, t, log_x_start.shape)

        log_probs = log_add_exp(
            log_x_start + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - torch.log(self.num_classes_expanded)
        )

        return log_probs

    def predict_start(self, model_out, log_x_t, t, out_dict):

        # model_out = self._denoise_fn(x_t, t.to(x_t.device), **out_dict)

        assert model_out.size(0) == log_x_t.size(0)
        assert model_out.size(1) == self.num_classes.sum(), f'{model_out.size()}'

        log_pred = torch.empty_like(model_out)
        for ix in self.slices_for_classes:
            log_pred[:, ix] = F.log_softmax(model_out[:, ix], dim=1)
        return log_pred

    def q_posterior(self, log_x_start, log_x_t, t):
        # q(xt-1 | xt, x0) = q(xt | xt-1, x0) * q(xt-1 | x0) / q(xt | x0)
        # where q(xt | xt-1, x0) = q(xt | xt-1).

        # EV_log_qxt_x0 = self.q_pred(log_x_start, t)

        # print('sum exp', EV_log_qxt_x0.exp().sum(1).mean())
        # assert False

        # log_qxt_x0 = (log_x_t.exp() * EV_log_qxt_x0).sum(dim=1)
        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_EV_qxtmin_x0 = self.q_pred(log_x_start, t_minus_1)

        num_axes = (1,) * (len(log_x_start.size()) - 1)
        t_broadcast = t.to(log_x_start.device).view(-1, *num_axes) * torch.ones_like(log_x_start)
        log_EV_qxtmin_x0 = torch.where(t_broadcast == 0, log_x_start, log_EV_qxtmin_x0.to(torch.float32))

        # unnormed_logprobs = log_EV_qxtmin_x0 +
        #                     log q_pred_one_timestep(x_t, t)
        # Note: _NOT_ x_tmin1, which is how the formula is typically used!!!
        # Not very easy to see why this is true. But it is :)
        unnormed_logprobs = log_EV_qxtmin_x0 + self.q_pred_one_timestep(log_x_t, t)

        log_EV_xtmin_given_xt_given_xstart = \
            unnormed_logprobs \
            - sliced_logsumexp(unnormed_logprobs, self.offsets)

        return log_EV_xtmin_given_xt_given_xstart

    def p_pred(self, model_out, log_x, t, out_dict):
        if self.parametrization == 'x0':
            log_x_recon = self.predict_start(model_out, log_x, t=t, out_dict=out_dict)
            log_model_pred = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_x, t=t)
        elif self.parametrization == 'direct':
            log_model_pred = self.predict_start(model_out, log_x, t=t, out_dict=out_dict)
        else:
            raise ValueError
        return log_model_pred

    @torch.no_grad()
    def p_sample(self, model_out, log_x, t, out_dict):
        model_log_prob = self.p_pred(model_out, log_x=log_x, t=t, out_dict=out_dict)
        out = self.log_sample_categorical(model_log_prob)
        return out

    @torch.no_grad()
    def p_sample_loop(self, shape, out_dict):
        device = self.log_alpha.device

        b = shape[0]
        # start with random normal image.
        img = torch.randn(shape, device=device)

        for i in reversed(range(1, self.num_timesteps)):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), out_dict)
        return img

    @torch.no_grad()
    def _sample(self, image_size, out_dict, batch_size = 16):
        return self.p_sample_loop((batch_size, 3, image_size, image_size), out_dict)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in reversed(range(0, t)):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def log_sample_categorical(self, logits):
        full_sample = []
        for i in range(len(self.num_classes)):
            one_class_logits = logits[:, self.slices_for_classes[i]]
            uniform = torch.rand_like(one_class_logits)
            gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
            sample = (gumbel_noise + one_class_logits).argmax(dim=1)
            full_sample.append(sample.unsqueeze(1))
        full_sample = torch.cat(full_sample, dim=1)
        log_sample = index_to_log_onehot(full_sample, self.num_classes)
        return log_sample

    def q_sample(self, log_x_start, t):
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)

        log_sample = self.log_sample_categorical(log_EV_qxt_x0)

        return log_sample

    def nll(self, log_x_start, out_dict):
        b = log_x_start.size(0)
        device = log_x_start.device
        loss = 0
        for t in range(0, self.num_timesteps):
            t_array = (torch.ones(b, device=device) * t).long()

            kl = self.compute_Lt(
                log_x_start=log_x_start,
                log_x_t=self.q_sample(log_x_start=log_x_start, t=t_array),
                t=t_array,
                out_dict=out_dict)

            loss += kl

        loss += self.kl_prior(log_x_start)

        return loss

    def kl_prior(self, log_x_start):
        b = log_x_start.size(0)
        device = log_x_start.device
        ones = torch.ones(b, device=device).long()

        log_qxT_prob = self.q_pred(log_x_start, t=(self.num_timesteps - 1) * ones)
        log_half_prob = -torch.log(self.num_classes_expanded * torch.ones_like(log_qxT_prob))

        kl_prior = self.multinomial_kl(log_qxT_prob, log_half_prob)
        return sum_except_batch(kl_prior)

    def compute_Lt(self, model_out, log_x_start, log_x_t, t, out_dict, detach_mean=False):
        log_true_prob = self.q_posterior(
            log_x_start=log_x_start, log_x_t=log_x_t, t=t)
        log_model_prob = self.p_pred(model_out, log_x=log_x_t, t=t, out_dict=out_dict)

        if detach_mean:
            log_model_prob = log_model_prob.detach()

        kl = self.multinomial_kl(log_true_prob, log_model_prob)
        kl = sum_except_batch(kl)

        decoder_nll = -log_categorical(log_x_start, log_model_prob)
        decoder_nll = sum_except_batch(decoder_nll)

        mask = (t == torch.zeros_like(t)).float()
        loss = mask * decoder_nll + (1. - mask) * kl

        return loss

    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = (Lt_sqrt / Lt_sqrt.sum()).to(device)

            t = torch.multinomial(pt_all, num_samples=b, replacement=True).to(device)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError

    def _multinomial_loss(self, model_out, log_x_start, log_x_t, t, pt, out_dict):

        # print('multinomial_loss_type: ', self.multinomial_loss_type) # vb_stochastic
        if self.multinomial_loss_type == 'vb_stochastic':
            kl = self.compute_Lt(
                model_out, log_x_start, log_x_t, t, out_dict
            )
            kl_prior = self.kl_prior(log_x_start)
            # Upweigh loss term of the kl
            vb_loss = kl / pt + kl_prior

            return vb_loss

        elif self.multinomial_loss_type == 'vb_all':
            # Expensive, dont do it ;).
            # DEPRECATED
            return -self.nll(log_x_start)
        else:
            raise ValueError()

    def log_prob(self, x, out_dict):
        b, device = x.size(0), x.device
        if self.training:
            return self._multinomial_loss(x, out_dict)

        else:
            log_x_start = index_to_log_onehot(x, self.num_classes)

            t, pt = self.sample_time(b, device, 'importance')

            kl = self.compute_Lt(
                log_x_start, self.q_sample(log_x_start=log_x_start, t=t), t, out_dict)

            kl_prior = self.kl_prior(log_x_start)

            # Upweigh loss term of the kl
            loss = kl / pt + kl_prior

            return -loss
    '''
    def mixed_loss(self, x, out_dict, do_train=False):
        b = x.shape[0]
        device = x.device
        t, pt = self.sample_time(b, device, 'uniform')

        x_num = x[:, :self.num_numerical_features]
        x_cat = x[:, self.num_numerical_features:]
        
        x_num_t = x_num
        log_x_cat_t = x_cat
        if x_num.shape[1] > 0:
            noise = torch.randn_like(x_num)
            x_num_t = self.gaussian_q_sample(x_num, t, noise=noise)
        if x_cat.shape[1] > 0:
            log_x_cat = index_to_log_onehot(x_cat.long(), self.num_classes)
            log_x_cat_t = self.q_sample(log_x_start=log_x_cat, t=t)
        
        x_in = torch.cat([x_num_t, log_x_cat_t], dim=1)
        if do_train:
            x_in.requires_grad_(True) # add for new new loss

        model_out = self._denoise_fn(
            x_in,
            t,
            **out_dict
        )

        if do_train:
            # add for new new loss
            # ====== CD Guidance Step ======
            model_out.requires_grad_(True)
            x0_hat = model_out  # 不要 detach！
            x0_hat_num = x0_hat[:, :self.num_numerical_features]
            x0_hat_cat = x0_hat[:, self.num_numerical_features:]

            # === Load classifier ===
            with open(os.path.join('train_log1/slice__model_train_result', "train_config.json"), "r") as f:
                train_config = restore_train_config(json.load(f))
            global_model = train_config['global_model']
            Nclass       = train_config['Nclass']
            num_feats    = train_config['num_feats']
            slice_len    = train_config['slice_len']
            pos_enc      = train_config.get('pos_enc', False)

            clf_model, _ = TRACTOR_model(Nclass, global_model, num_feats, slice_len, pos_enc)
            clf_model.load_state_dict(torch.load(os.path.join('train_log1/slice__model_train_result', "model.1.trans_v1.pt"), map_location='cuda:0')['model_state_dict'])
            clf_model = clf_model.to(x0_hat.device)
            clf_model.eval()
            for p in clf_model.parameters():
                p.requires_grad = False  # 確保 classifier 不會更新

            # === Preprocess input for classifier ===
            if x_cat.shape[1] > 0:
                split_logits = torch.split(x0_hat_cat, list(self.num_classes), dim=1)
                x_cat_hat_list = [soft_argmax(logit) for logit in split_logits]
                x_cat_hat = torch.stack(x_cat_hat_list, dim=1)  # shape [B, num_cat_features]
            else:
                x_cat_hat = torch.empty((b, 0), device=device)
            x0_clf = torch.cat([x0_hat_num, x_cat_hat], dim=1) if x_cat.shape[1] > 0 else x0_hat_num

            x0_clf = x0_clf[:, torch.arange(x0_clf.shape[1]) != x0_clf.shape[1] - 2]  # 移除 'slice_id'
            if x0_clf.ndim == 2:
                x0_clf = x0_clf.unsqueeze(0)  # [1, B, F]
            x0_clf = x0_clf.permute(1, 0, 2)  # [B, L, D]

            # === Compute CD (no torch.no_grad!) ===
            probs = F.softmax(clf_model(x0_clf), dim=1)
            # print('gradient: ', x0_clf.requires_grad, x0_clf.grad_fn)
            y_true = out_dict['y'].to(probs.device)
            correct_class_probs = probs.gather(1, y_true.unsqueeze(1)).squeeze(1)
            sum_other_probs = probs.sum(dim=1) - correct_class_probs
            cd = correct_class_probs - sum_other_probs
            y_pred = torch.argmax(probs, dim=1)
            # print(f'y true: {y_true}, y predict: {y_pred}, correct_class_probs: {correct_class_probs}, sum_other_probs: {sum_other_probs}, cd: {cd}')

            # === CD Loss ===
            # target_cd_value = 0.0  # boundary zone target
            # cd_loss = ((cd - target_cd_value)**2).mean()

            # delta = 0.2
            # is_boundary = (cd.abs() <= delta).float()
            # cd_loss_boundary = is_boundary * cd**2
            # cd_loss = cd_loss_boundary.mean()

            cd_abs = cd.abs()
            # 初始化權重為 0
            weight = torch.zeros_like(cd)

            # (1) 最大懲罰區 −1 ~ −0.2
            mask1 = (cd < -0.2) & (cd >= -1.0)
            weight[mask1] = 3.0

            # (2) 次大懲罰區 +0.2 ~ +0.8
            mask2 = (cd > 0.2) & (cd <= 0.8)
            weight[mask2] = 2.0

            # (3) 最小懲罰區 −0.2 ~ +0.2
            mask3 = (cd_abs <= 0.2)
            weight[mask3] = 0.5

            # 使用平方懲罰形式（可改成 abs() 等其他形式）
            cd_loss = torch.clamp((weight * cd**2), max=10).mean()
            # print(f'weight: {weight}, cd_loss: {cd_loss}')
            # print("cd_loss:", cd_loss.item())
            # print("cd_loss.requires_grad:", cd_loss.requires_grad)
            # print("x0_hat.requires_grad:", x0_hat.requires_grad)

            # === 計算梯度並修正 model_out ===
            grad_cd = torch.autograd.grad(cd_loss, x0_hat, retain_graph=True, create_graph=True)[0]
            # rho = 0.5  # guidance strength (你可以用 Optuna 調這參數)
            # model_out = model_out - rho * grad_cd
            # print("grad_cd:", grad_cd)
            # print("grad_cd is None?", grad_cd is None)
            
            # 確認 grad_cd 有非零元素
            # print("grad_cd.norm():", grad_cd.norm().item())


        model_out_num = model_out[:, :self.num_numerical_features]
        model_out_cat = model_out[:, self.num_numerical_features:]

        loss_multi = torch.zeros((1,)).float()
        loss_gauss = torch.zeros((1,)).float()
        if x_cat.shape[1] > 0:
            loss_multi = self._multinomial_loss(model_out_cat, log_x_cat, log_x_cat_t, t, pt, out_dict) / len(self.num_classes)
        
        if x_num.shape[1] > 0:
            loss_gauss = self._gaussian_loss(model_out_num, x_num, x_num_t, t, noise)

        # loss_multi = torch.where(out_dict['y'] == 1, loss_multi, 2 * loss_multi)
        # loss_gauss = torch.where(out_dict['y'] == 1, loss_gauss, 2 * loss_gauss)

        # if return_x0:
        #     # 將 model_out 組回類似原始輸入格式的 x0_hat
        #     # 類別從 log-prob → one-hot → index
        #     if x_cat.shape[1] > 0:
        #         split_logits = torch.split(model_out_cat, list(self.num_classes), dim=1)
        #         x_cat_hat_list = [log_onehot_to_index(logit) for logit in split_logits]
        #         x_cat_hat = torch.stack(x_cat_hat_list, dim=1)  # shape [B, num_cat_features]
        #     else:
        #         x_cat_hat = torch.empty((b, 0), device=device)
        #     x0_hat = torch.cat([model_out_num, x_cat_hat], dim=1) if x_cat.shape[1] > 0 else model_out_num
        #     return loss_multi.mean(), loss_gauss.mean(), x0_hat
        if do_train:
            return loss_multi.mean(), loss_gauss.mean(), cd_loss
        else:
            return loss_multi.mean(), loss_gauss.mean()
    '''
    '''
    def mixed_loss(self, x, out_dict, return_x0=False):
        b = x.shape[0]
        device = x.device
        t, pt = self.sample_time(b, device, 'uniform')

        x_num = x[:, :self.num_numerical_features]
        x_cat = x[:, self.num_numerical_features:]
        
        x_num_t = x_num
        log_x_cat_t = x_cat
        if x_num.shape[1] > 0:
            noise = torch.randn_like(x_num)
            x_num_t = self.gaussian_q_sample(x_num, t, noise=noise)
        if x_cat.shape[1] > 0:
            log_x_cat = index_to_log_onehot(x_cat.long(), self.num_classes)
            log_x_cat_t = self.q_sample(log_x_start=log_x_cat, t=t)
        
        x_in = torch.cat([x_num_t, log_x_cat_t], dim=1)

        model_out = self._denoise_fn(
            x_in,
            t,
            **out_dict
        )

        model_out_num = model_out[:, :self.num_numerical_features]
        model_out_cat = model_out[:, self.num_numerical_features:]

        loss_multi = torch.zeros((1,)).float()
        loss_gauss = torch.zeros((1,)).float()
        if x_cat.shape[1] > 0:
            loss_multi = self._multinomial_loss(model_out_cat, log_x_cat, log_x_cat_t, t, pt, out_dict) / len(self.num_classes)
        
        if x_num.shape[1] > 0:
            loss_gauss = self._gaussian_loss(model_out_num, x_num, x_num_t, t, noise)

        # loss_multi = torch.where(out_dict['y'] == 1, loss_multi, 2 * loss_multi)
        # loss_gauss = torch.where(out_dict['y'] == 1, loss_gauss, 2 * loss_gauss)

        if return_x0:
            # 將 model_out 組回類似原始輸入格式的 x0_hat
            # 類別從 log-prob → one-hot → index
            if x_cat.shape[1] > 0:
                split_logits = torch.split(model_out_cat, list(self.num_classes), dim=1)
                x_cat_hat_list = [log_onehot_to_index(logit) for logit in split_logits]
                x_cat_hat = torch.stack(x_cat_hat_list, dim=1)  # shape [B, num_cat_features]
            else:
                x_cat_hat = torch.empty((b, 0), device=device)
            x0_hat = torch.cat([model_out_num, x_cat_hat], dim=1) if x_cat.shape[1] > 0 else model_out_num
            return loss_multi.mean(), loss_gauss.mean(), x0_hat
        return loss_multi.mean(), loss_gauss.mean()
    '''
    '''
    def compute_confidence_gradient(self, x, classifier, y_true):
        x = x.detach().clone()  # 保證 leaf tensor
        x.requires_grad = True
        x_clf_input = preprocess_for_classifier(x, self.num_numerical_features, self.num_classes, x.device)
        out = classifier(x_clf_input)
        loss = F.cross_entropy(out, y_true)
        loss.backward()
        return x.grad
    
    def mixed_loss(self, x, out_dict, do_train=False):
        b = x.shape[0]
        device = x.device
        t, pt = self.sample_time(b, device, 'uniform')

        x_num = x[:, :self.num_numerical_features]
        x_cat = x[:, self.num_numerical_features:]

        if x_num.shape[1] > 0:
            noise = torch.randn_like(x_num)
            x_num_t = self.gaussian_q_sample(x_num, t, noise=noise)
        else:
            x_num_t = x_num

        if x_cat.shape[1] > 0:
            log_x_cat = index_to_log_onehot(x_cat.long(), self.num_classes)
            log_x_cat_t = self.q_sample(log_x_start=log_x_cat, t=t)
        else:
            log_x_cat_t = x_cat

        x_in = torch.cat([x_num_t, log_x_cat_t], dim=1)

        # if do_train:
        #     grad = self.compute_confidence_gradient(x_in, self.classifier, out_dict['y'])
        #     assert grad is not None, "Gradient is None!"
        #     grad = F.normalize(grad, p=2, dim=-1)
        #     x_concat = torch.cat([x_num, log_x_cat], dim=1)
        #     perturb = x_in - x_concat.detach()
        #     proj = torch.sum(perturb * grad, dim=1, keepdim=True) * grad
        #     orthogonal_perturb = perturb - proj
        #     x_in = x_concat + orthogonal_perturb

        model_out = self._denoise_fn(x_in, t, **out_dict)
        model_out_num = model_out[:, :self.num_numerical_features]
        model_out_cat = model_out[:, self.num_numerical_features:]

        loss_multi = self._multinomial_loss(model_out_cat, log_x_cat, log_x_cat_t, t, pt, out_dict) / len(self.num_classes) if x_cat.shape[1] > 0 else 0
        loss_gauss = self._gaussian_loss(model_out_num, x_num, x_num_t, t, noise) if x_num.shape[1] > 0 else 0

        if do_train:
            model_out_num = model_out[:, :self.num_numerical_features]
            x_adv = x.clone()
            x_adv[:, :self.num_numerical_features] += model_out_num
            # ========== Perturbation output ==========
            x_clean_clf = x[:, torch.arange(x.shape[1]) != x.shape[1] - 2]  # remove slice_id
            if x_clean_clf.ndim == 2:
                x_clean_clf = x_clean_clf.unsqueeze(0)  # [1, B, F]
            x_clean_input = x_clean_clf.permute(1, 0, 2)

            x_adv_clf = x_adv[:, torch.arange(x.shape[1]) != x.shape[1] - 2]  # remove slice_id
            if x_adv_clf.ndim == 2:
                x_adv_clf = x_adv_clf.unsqueeze(0)  # [1, B, F]
            x_adv_input = x_adv_clf.permute(1, 0, 2)

            # perturbation = model_out  # 要求 model_out 學習對抗樣本的 direction
            # perturbation = preprocess_for_classifier(perturbation, self.num_numerical_features, self.num_classes, x.device)
            # x_adv = x_clean_input + perturbation  # 或是 x_num + model_out_num，針對 numeric

            # ========== L_min: 特徵改變不大 ==========
            feat_clean = self.classifier(x_clean_input)         # target model 預測原始樣本
            feat_adv = self.classifier(x_adv_input)       # 預測對抗樣本
            loss_min = F.mse_loss(feat_adv, feat_clean)

            # ========== L_CE: 錯誤預測目標 ==========
            y = out_dict['y']
            logits_adv = self.classifier(x_adv_input)
            # loss_ce = F.cross_entropy(logits_adv, y)

            # correct_class_probs = probs.gather(1, y_true.unsqueeze(1)).squeeze(1)
            # sum_other_probs = probs.sum(dim=1) - correct_class_probs
            # cd = correct_class_probs - sum_other_probs  # Confidence Difference

            probs = torch.softmax(logits_adv, dim=1)
            correct_probs = probs[torch.arange(probs.size(0)), y]
            probs_clone = probs.clone()
            probs_clone[torch.arange(probs.size(0)), y] = -1
            max_other_probs, _ = probs_clone.max(dim=1)
            confidence_diff = correct_probs - max_other_probs
            delta = 0.2
            loss_cd = torch.mean(
                F.relu(confidence_diff - delta) +  # 超出上界 +0.2
                F.relu(-delta - confidence_diff)   # 低於下界 -0.2
            )

            # ========== L_bound: 控制擾動大小 ==========
            # loss_bound = torch.norm(perturbation.view(x_clean_input.shape[0], -1), p=1, dim=1).mean()
            loss_bound = torch.norm(model_out_num, p=1)

            gamma1, gamma2, gamma3 = 0.5, 1.0, 1e-2
            loss_bdpm = gamma1 * loss_min + gamma2 * loss_cd + gamma3 * loss_bound

            # # === BDPM guidance ===
            # with torch.no_grad():    
            #     x_clf_input = preprocess_for_classifier(x_in, self.num_numerical_features, self.num_classes, x.device)
            #     pred_adv = self.classifier(x_clf_input) # F(x + perturbation)

            #     x_clean_clf = x[:, torch.arange(x.shape[1]) != x.shape[1] - 2]  # remove slice_id
            #     if x_clean_clf.ndim == 2:
            #         x_clean_clf = x_clean_clf.unsqueeze(0)  # [1, B, F]
            #     x_clean_input = x_clean_clf.permute(1, 0, 2)
            #     pred_clean = self.classifier(x_clean_input) # F(x)

            # perturb = x_clf_input - x_clean_input

            # loss_min = F.mse_loss(pred_adv, pred_clean)
            # loss_ce = F.cross_entropy(pred_adv, out_dict['y'])
            # # epsilon = 0.5
            # # loss_bound = torch.norm(torch.clamp(perturb, -epsilon, epsilon), p=1)
            # # loss_bound = torch.norm(perturb, p=1) # L1
            # loss_bound = torch.norm(perturb.view(b, -1), p=2, dim=1).mean() # L2
            # # print(f'loss_min: {loss_min}, loss_ce:{loss_ce}, loss_bound: {loss_bound}')
            # # print("Perturbation mean norm:", perturb.abs().mean().item())

            # gamma1, gamma2, gamma3 = 1.0, 1.0, 1e-2
            # loss_bdpm = gamma1 * loss_min + gamma2 * loss_ce + gamma3 * loss_bound

        if do_train:
            return loss_multi.mean(), loss_gauss.mean(), loss_bdpm.mean()
        else:
            return loss_multi.mean(), loss_gauss.mean()
        '''

    # add for bdpm loss
    def mixed_loss(self, x, out_dict, step=None, total_steps=30000, do_train=False, saving_boundary_data=False):
        b = x.shape[0]
        device = x.device
        t, pt = self.sample_time(b, device)

        # ====== 原本的 loss 部分 ======
        loss_multi = 0#self._multinomial_loss(x, out_dict, t, pt)  # 原始 multi-modal loss
        loss_gauss = 0#self._gaussian_loss(x, out_dict, t, pt)      # 原始 Gaussian loss

        if do_train:
            # ====== BDPM Losses ======
            model_out = x  # 可以是 x0_hat or epsilon
            model_out.requires_grad_(True)
            x0_hat = model_out

            x0_hat_num = x0_hat[:, :self.num_numerical_features]  # 數值部分
            x0_hat_cat = x0_hat[:, self.num_numerical_features:]  # 分類部分

            # ====== 套用 normalization（x0_hat_num）=====
            # add for npy classify
            if not self.scaler_fitted:
                x0_hat_num_np = x0_hat_num.detach().cpu().numpy()
                x0_hat_num_np = self.scaler.fit_transform(x0_hat_num_np)
                self.scaler_fitted = True
            else:
                x0_hat_num_np = x0_hat_num.detach().cpu().numpy()
                x0_hat_num_np = self.scaler.transform(x0_hat_num_np)
            x0_hat_num = torch.tensor(x0_hat_num_np, dtype=torch.float32, device=x0_hat.device)

            # ====== BDPM 擾動產生與套用 ======
            self.bdpm.train()
            perturb = self.bdpm(x0_hat_num, t)
            x_adv_num = x0_hat_num + perturb

            x0_hat = torch.cat([x0_hat_num, x0_hat_cat], dim=1)
            x_adv = torch.cat([x_adv_num, x0_hat_cat], dim=1)

            # # add for npy classify
            # x_adv_num_np = x_adv_num.detach().cpu().numpy()
            # x_adv_num_original_np = self.scaler.inverse_transform(x_adv_num_np)
            # x_adv_num_original = torch.tensor(x_adv_num_original_np, dtype=torch.float32, device=x_adv.device)

            # x_adv_save = torch.cat([x_adv_num_original, x0_hat_cat], dim=1)
            x_adv_save = x_adv # modify for npy classify
            x0_hat = x0_hat[:, torch.arange(x.shape[1]) != x.shape[1] - 2]  # remove slice_id
            if x0_hat.ndim == 2:
                x0_hat = x0_hat.unsqueeze(0)  # [1, B, F]
            x0_hat = x0_hat.permute(1, 0, 2)
            x_adv = x_adv[:, torch.arange(x.shape[1]) != x.shape[1] - 2]  # remove slice_id
            if x_adv.ndim == 2:
                x_adv = x_adv.unsqueeze(0)  # [1, B, F]
            x_adv = x_adv.permute(1, 0, 2)

            # 對分類器做預測
            pred_orig = self.classifier(x0_hat)
            pred_adv = self.classifier(x_adv)

            y = out_dict['y']

            # loss_min = ((pred_adv - pred_orig) ** 2).mean()
            # loss_ce = F.cross_entropy(pred_adv, y)


            # logits from pred_adv (no softmax)
            logits = pred_adv
            true_logit = logits[torch.arange(logits.size(0)), y]
            logits_clone = logits.clone()
            logits_clone[torch.arange(logits.size(0)), y] = float('-inf')
            max_other_logit, _ = logits_clone.max(dim=1)

            confidence_diff_logit = true_logit - max_other_logit
            delta = 1.0
            # 以 batch 為例
            q1 = confidence_diff_logit.quantile(0.25)
            q3 = confidence_diff_logit.quantile(0.75)
            q2 = confidence_diff_logit.quantile(0.5)
            iqr = q3 - q1
            k = 1.0
            delta = q3 + k * iqr  # k 可設為 1.5 或 3，視情況而定
            # delta = q2
            print(f'delta: {delta}')

            if step == total_steps:
                cd_values = confidence_diff_logit.detach().cpu().numpy()
                # 畫 histogram
                plt.figure(figsize=(8, 5))
                plt.hist(cd_values, bins=100, color='skyblue', edgecolor='black')
                plt.axvline(0, color='red', linestyle='--', label='CD = 0')
                plt.axvline(+1.0, color='green', linestyle='--', label='delta = +1.0')
                plt.axvline(-1.0, color='green', linestyle='--', label='delta = -1.0')
                plt.title("Histogram of Confidence Difference (logit-based)")
                plt.xlabel("Confidence Difference (true_logit - max_other_logit)")
                plt.ylabel("Frequency")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f"Histogram of Confidence Difference.png")
            # # plt.show()

            # print(f'true_logit {true_logit}, max_other_logit {max_other_logit}, confidence_diff_logit {confidence_diff_logit}')
            cd_near0 = (confidence_diff_logit.abs() < delta).float().mean()
            cd_gt_delta = (confidence_diff_logit > delta).float().mean()
            cd_lt_minus_delta = (confidence_diff_logit < -delta).float().mean()
            # loss_cd = ((confidence_diff_logit / delta) ** 2).mean()
            loss_cd_pos = ((F.relu(confidence_diff_logit - delta)) ** 2).mean()  # CD > +delta
            loss_cd_neg = ((F.relu(-delta - confidence_diff_logit)) ** 2).mean()  # CD < -delta（你最關心的）
            # 新增鼓勵靠近 0 的項
            cd_center_loss = (confidence_diff_logit ** 2).mean()
            loss_cd = 1.0 * cd_center_loss + 0.0 * loss_cd_pos + 0.0 * loss_cd_neg  # 加大CD<−delta懲罰

            # Optional: mask out extreme bad examples from loss_min
            # cd_mask = (confidence_diff_logit > -delta)  # only those close to boundary
            core_delta = delta * 0.8  # 更窄的範圍，如 [-0.28, +0.28]
            cd_mask = (confidence_diff_logit > -core_delta) & (confidence_diff_logit < core_delta)

            if cd_mask.any():
                loss_min = ((pred_adv[cd_mask] - pred_orig[cd_mask]) ** 2).mean()
            else:
                loss_min = torch.tensor(0.0, device=x.device)


            # probs = torch.softmax(pred_adv, dim=1)
            # correct_probs = probs[torch.arange(probs.size(0)), y]
            # probs_clone = probs.clone()
            # probs_clone[torch.arange(probs.size(0)), y] = -1
            # max_other_probs, _ = probs_clone.max(dim=1)
            # confidence_diff = correct_probs - max_other_probs
            # loss_cd = ((confidence_diff / delta) ** 2).mean()

            # torch.mean(
            #     F.relu(confidence_diff - delta) +  # 超出上界 +0.2
            #     F.relu(-delta - confidence_diff)   # 低於下界 -0.2
            # )
            # print("CD Mean:", confidence_diff.mean().item(), 
            #     "CD Max:", confidence_diff.max().item(), 
            #     "CD Min:", confidence_diff.min().item())

            loss_bound = perturb.abs().mean()
            # alpha = (confidence_diff_logit.abs() / delta).clamp(min=1.0).unsqueeze(1)
            # loss_bound = (alpha * (perturb ** 2)).mean()


            # 加權係數
            gamma_1, gamma_2, gamma_3 = 0.5, 1.0, 0.3

            loss_bdpm = gamma_1 * loss_min + gamma_2 * loss_cd + gamma_3 * loss_bound
            print(f'loss_bdpm {loss_bdpm.item():.4f}, loss_min {loss_min.item():.4f}, loss_cd {loss_cd.item():.4f}, loss_bound {loss_bound.item():.4f}, CD mean {confidence_diff_logit.mean().item():.4f}, CD %near0 {cd_near0.item() * 100:.2f}%, CD %>+delta {cd_gt_delta.item() * 100:.2f}%, CD %<-delta {cd_lt_minus_delta.item() * 100:.2f}%')

            # ====== 整體 total loss ======
            loss_total = loss_multi + loss_gauss + loss_bdpm
            if saving_boundary_data:
                x_adv_num = x_adv_save[:, :self.num_numerical_features]
                x_adv_cat = x_adv_save[:, self.num_numerical_features:]

                # add for record delta
                self.delta_accumulate += delta.item()
                self.delta_count += 1
                delta_mean = self.delta_accumulate / self.delta_count
                print(f"delta_accumulate mean: {delta_mean:.4f}")

                # y_true = y
                # y_pred = pred_adv.argmax(dim=1)

                # # 建立 mask：在 boundary zone 且預測正確
                # boundary_mask = (confidence_diff_logit.abs() < delta)
                # correct_mask = (y_pred == y_true)
                # final_mask = boundary_mask | correct_mask

                # if final_mask.any():
                #     x_adv_num = x_adv_num[final_mask].detach().cpu()
                #     x_adv_cat = x_adv_cat[final_mask].detach().cpu()
                #     y = y[final_mask].detach().cpu()
                #     print(f"[✔] Saved {len(y)} samples in boundary zone and correctly classified.")
                # else:
                #     print("[!] No valid boundary+correct samples found.")
                #     x_adv_num = torch.empty(0)
                #     x_adv_cat = torch.empty(0)
                #     y = torch.empty(0)

                # return loss_multi, loss_gauss, loss_bdpm, x_adv_num, x_adv_cat, y
                return loss_multi, loss_gauss, loss_bdpm, x_adv_num.detach().cpu(), x_adv_cat.detach().cpu(), y.detach().cpu()
            else:
                return loss_multi, loss_gauss, loss_bdpm
        else:
            loss_bdpm = 0
            return loss_multi, loss_gauss


    @torch.no_grad()
    def mixed_elbo(self, x0, out_dict):
        b = x0.size(0)
        device = x0.device

        x_num = x0[:, :self.num_numerical_features]
        x_cat = x0[:, self.num_numerical_features:]
        has_cat = x_cat.shape[1] > 0
        if has_cat:
            log_x_cat = index_to_log_onehot(x_cat.long(), self.num_classes).to(device)

        gaussian_loss = []
        xstart_mse = []
        mse = []
        mu_mse = []
        out_mean = []
        true_mean = []
        multinomial_loss = []
        for t in range(self.num_timesteps):
            t_array = (torch.ones(b, device=device) * t).long()
            noise = torch.randn_like(x_num)

            x_num_t = self.gaussian_q_sample(x_start=x_num, t=t_array, noise=noise)
            if has_cat:
                log_x_cat_t = self.q_sample(log_x_start=log_x_cat, t=t_array)
            else:
                log_x_cat_t = x_cat

            model_out = self._denoise_fn(
                torch.cat([x_num_t, log_x_cat_t], dim=1),
                t_array,
                **out_dict
            )
            
            model_out_num = model_out[:, :self.num_numerical_features]
            model_out_cat = model_out[:, self.num_numerical_features:]

            kl = torch.tensor([0.0])
            if has_cat:
                kl = self.compute_Lt(
                    model_out=model_out_cat,
                    log_x_start=log_x_cat,
                    log_x_t=log_x_cat_t,
                    t=t_array,
                    out_dict=out_dict
                )

            out = self._vb_terms_bpd(
                model_out_num,
                x_start=x_num,
                x_t=x_num_t,
                t=t_array,
                clip_denoised=False
            )

            multinomial_loss.append(kl)
            gaussian_loss.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_num) ** 2))
            # mu_mse.append(mean_flat(out["mean_mse"]))
            out_mean.append(mean_flat(out["out_mean"]))
            true_mean.append(mean_flat(out["true_mean"]))

            eps = self._predict_eps_from_xstart(x_num_t, t_array, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        gaussian_loss = torch.stack(gaussian_loss, dim=1)
        multinomial_loss = torch.stack(multinomial_loss, dim=1)
        xstart_mse = torch.stack(xstart_mse, dim=1)
        mse = torch.stack(mse, dim=1)
        # mu_mse = torch.stack(mu_mse, dim=1)
        out_mean = torch.stack(out_mean, dim=1)
        true_mean = torch.stack(true_mean, dim=1)



        prior_gauss = self._prior_gaussian(x_num)

        prior_multin = torch.tensor([0.0])
        if has_cat:
            prior_multin = self.kl_prior(log_x_cat)

        total_gauss = gaussian_loss.sum(dim=1) + prior_gauss
        total_multin = multinomial_loss.sum(dim=1) + prior_multin
        return {
            "total_gaussian": total_gauss,
            "total_multinomial": total_multin,
            "losses_gaussian": gaussian_loss,
            "losses_multinimial": multinomial_loss,
            "xstart_mse": xstart_mse,
            "mse": mse,
            # "mu_mse": mu_mse
            "out_mean": out_mean,
            "true_mean": true_mean
        }

    @torch.no_grad()
    def gaussian_ddim_step(
        self,
        model_out_num,
        x,
        t,
        clip_denoised=False,
        denoised_fn=None,
        eta=0.0
    ):
        out = self.gaussian_p_mean_variance(
            model_out_num,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=None,
        )

        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = extract(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = extract(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        noise = torch.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise

        return sample
    
    @torch.no_grad()
    def gaussian_ddim_sample(
        self,
        noise,
        T,
        out_dict,
        eta=0.0
    ):
        x = noise
        b = x.shape[0]
        device = x.device
        for t in reversed(range(T)):
            print(f'Sample timestep {t:4d}', end='\r')
            t_array = (torch.ones(b, device=device) * t).long()
            out_num = self._denoise_fn(x, t_array, **out_dict)
            x = self.gaussian_ddim_step(
                out_num,
                x,
                t_array
            )
        print()
        return x


    @torch.no_grad()
    def gaussian_ddim_reverse_step(
        self,
        model_out_num,
        x,
        t,
        clip_denoised=False,
        eta=0.0
    ):
        assert eta == 0.0, "Eta must be zero."
        out = self.gaussian_p_mean_variance(
            model_out_num,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=None,
            model_kwargs=None,
        )

        eps = (
            extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = extract(self.alphas_cumprod_next, t, x.shape)

        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_next)
            + torch.sqrt(1 - alpha_bar_next) * eps
        )

        return mean_pred

    @torch.no_grad()
    def gaussian_ddim_reverse_sample(
        self,
        x,
        T,
        out_dict,
    ):
        b = x.shape[0]
        device = x.device
        for t in range(T):
            print(f'Reverse timestep {t:4d}', end='\r')
            t_array = (torch.ones(b, device=device) * t).long()
            out_num = self._denoise_fn(x, t_array, **out_dict)
            x = self.gaussian_ddim_reverse_step(
                out_num,
                x,
                t_array,
                eta=0.0
            )
        print()

        return x


    @torch.no_grad()
    def multinomial_ddim_step(
        self,
        model_out_cat,
        log_x_t,
        t,
        out_dict,
        eta=0.0
    ):
        # not ddim, essentially
        log_x0 = self.predict_start(model_out_cat, log_x_t=log_x_t, t=t, out_dict=out_dict)

        alpha_bar = extract(self.alphas_cumprod, t, log_x_t.shape)
        alpha_bar_prev = extract(self.alphas_cumprod_prev, t, log_x_t.shape)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        coef1 = sigma
        coef2 = alpha_bar_prev - sigma * alpha_bar
        coef3 = 1 - coef1 - coef2
        

        log_ps = torch.stack([
            torch.log(coef1) + log_x_t,
            torch.log(coef2) + log_x0,
            torch.log(coef3) - torch.log(self.num_classes_expanded)
        ], dim=2)

        log_prob = torch.logsumexp(log_ps, dim=2)

        out = self.log_sample_categorical(log_prob)

        return out

    @torch.no_grad()
    def sample_ddim(self, num_samples, y_dist):
        b = num_samples
        device = self.log_alpha.device
        z_norm = torch.randn((b, self.num_numerical_features), device=device)

        has_cat = self.num_classes[0] != 0
        log_z = torch.zeros((b, 0), device=device).float()
        if has_cat:
            uniform_logits = torch.zeros((b, len(self.num_classes_expanded)), device=device)
            log_z = self.log_sample_categorical(uniform_logits)

        y = torch.multinomial(
            y_dist,
            num_samples=b,
            replacement=True
        )
        out_dict = {'y': y.long().to(device)}
        for i in reversed(range(0, self.num_timesteps)):
            print(f'Sample timestep {i:4d}', end='\r')
            t = torch.full((b,), i, device=device, dtype=torch.long)
            model_out = self._denoise_fn(
                torch.cat([z_norm, log_z], dim=1).float(),
                t,
                **out_dict
            )
            model_out_num = model_out[:, :self.num_numerical_features]
            model_out_cat = model_out[:, self.num_numerical_features:]
            z_norm = self.gaussian_ddim_step(model_out_num, z_norm, t, clip_denoised=False)
            if has_cat:
                log_z = self.multinomial_ddim_step(model_out_cat, log_z, t, out_dict)

        z_ohe = torch.exp(log_z).round()
        z_cat = log_z
        if has_cat:
            z_cat = ohe_to_categories(z_ohe, self.num_classes)
        sample = torch.cat([z_norm, z_cat], dim=1).cpu()
        return sample, out_dict
    
    # modify for bdpm loss
    # @torch.no_grad()
    # def sample(self, num_samples, y_dist):
    #     b = num_samples
    #     device = self.log_alpha.device
    #     z_norm = torch.randn((b, self.num_numerical_features), device=device)

    #     has_cat = self.num_classes[0] != 0
    #     log_z = torch.zeros((b, 0), device=device).float()
    #     if has_cat:
    #         uniform_logits = torch.zeros((b, len(self.num_classes_expanded)), device=device)
    #         log_z = self.log_sample_categorical(uniform_logits)

    #     y = torch.multinomial(
    #         y_dist,
    #         num_samples=b,
    #         replacement=True
    #     )
    #     out_dict = {'y': y.long().to(device)}
    #     for i in reversed(range(0, self.num_timesteps)):
    #         print(f'Sample timestep {i:4d}', end='\r')
    #         t = torch.full((b,), i, device=device, dtype=torch.long)
    #         model_out = self._denoise_fn(
    #             torch.cat([z_norm, log_z], dim=1).float(),
    #             t,
    #             **out_dict
    #         )
    #         model_out_num = model_out[:, :self.num_numerical_features]
    #         model_out_cat = model_out[:, self.num_numerical_features:]
    #         z_norm = self.gaussian_p_sample(model_out_num, z_norm, t, clip_denoised=False)['sample']
    #         if has_cat:
    #             log_z = self.p_sample(model_out_cat, log_z, t, out_dict)

    #     print()
    #     z_ohe = torch.exp(log_z).round()
    #     z_cat = log_z
    #     if has_cat:
    #         z_cat = ohe_to_categories(z_ohe, self.num_classes)
    #     sample = torch.cat([z_norm, z_cat], dim=1).cpu()
    #     return sample, out_dict
    '''
    # @torch.no_grad()
    def sample(self, num_samples, y_dist):
        b = num_samples
        device = self.log_alpha.device

        with torch.no_grad():
            z_norm = torch.randn((b, self.num_numerical_features), device=device)

            has_cat = self.num_classes[0] != 0
            log_z = torch.zeros((b, 0), device=device).float()
            if has_cat:
                uniform_logits = torch.zeros((b, len(self.num_classes_expanded)), device=device)
                log_z = self.log_sample_categorical(uniform_logits)

            y = torch.multinomial(
                y_dist,
                num_samples=b,
                replacement=True
            )
            out_dict = {'y': y.long().to(device)}

        self.classifier.eval()
        for param in self.classifier.parameters():
            param.requires_grad = False

        for i in reversed(range(0, self.num_timesteps)):
            print(f'Sample timestep {i:4d}', end='\r')
            t = torch.full((b,), i, device=device, dtype=torch.long)
            model_out = self._denoise_fn(
                torch.cat([z_norm, log_z], dim=1).float(),
                t,
                **out_dict
            )
            model_out_num = model_out[:, :self.num_numerical_features]
            model_out_cat = model_out[:, self.num_numerical_features:]


            # for bdpm loss
            # 原始樣本 z_norm 使用模型輸出
            z_next = self.gaussian_p_sample(model_out_num, z_norm, t, clip_denoised=False)['sample']

            # 若啟用 BDPM 導引，則加上擾動（只對數值特徵）
            if hasattr(self, 'bdpm'):
                perturb = self.bdpm(model_out_num.detach(), t)
                z_next = z_next + perturb  # 也可以試試加權調整，或限制 perturb norm

            z_next.requires_grad_(True)
            # == 插入 CD guidance（timestep 內） ==
            # 將類別資訊組合上來，這裡直接取 log_z，無需 ohe
            tau = 1.0
            z_ohe_soft = torch.nn.functional.gumbel_softmax(log_z, tau=tau, hard=False)  # [B, C]

            # 接著把 soft one-hot -> 類別 index（只是為了構造 classifier input）
            # 假設 self.num_classes = [15, 3, 3]，總共 21 維 one-hot 編碼
            num_classes_list = self.num_classes  # e.g., [15, 3, 3]
            cat_feature_soft_list = []
            start = 0
            for n_cls in num_classes_list:
                end = start + n_cls
                one_hot_slice = z_ohe_soft[:, start:end]  # 取該欄位的 one-hot
                class_range = torch.arange(n_cls, device=z_ohe_soft.device).float()
                soft_label = (one_hot_slice * class_range).sum(dim=1, keepdim=True)
                cat_feature_soft_list.append(soft_label)
                start = end

            # 拼接所有欄位的類別 index（[B, 1] → [B, num_cat_features]）
            z_cat_soft = torch.cat(cat_feature_soft_list, dim=1)

            # 拼接數值與類別（用 soft index）
            z_temp = torch.cat([z_next, z_cat_soft], dim=1)
            # z_temp = torch.cat([z_next, z_ohe_soft], dim=1)  # 這個用於 backprop，仍保留 differentiability

            # classifier 輸入的 shape 為 [B, 1, F]，移除 slice_id（最後一個欄位）
            sample_clf = z_temp[:, torch.arange(z_temp.shape[1]) != z_temp.shape[1] - 2]
            sample_clf = sample_clf.unsqueeze(1).float()
            logits = self.classifier(sample_clf)
            # prob = torch.nn.functional.softmax(logits, dim=1)
            # top2 = prob.topk(2, dim=1).values
            # cd = top2[:, 0] - top2[:, 1]
            true_logit = logits[torch.arange(logits.size(0)), y]
            logits_clone = logits.clone()
            logits_clone[torch.arange(logits.size(0)), y] = float('-inf')
            max_other_logit, _ = logits_clone.max(dim=1)
            confidence_diff_logit = true_logit - max_other_logit

            # 可以選擇只導引 CD 小於閾值者
            delta = 1.0
            mask = torch.abs(confidence_diff_logit) < delta
            if mask.any():
                guidance_loss = torch.abs(confidence_diff_logit[mask]).mean()
                grad = torch.autograd.grad(guidance_loss, z_temp, retain_graph=False, create_graph=False)[0]
                z_next = z_next - self.cd_guidance_scale * grad[:, :self.num_numerical_features].detach()

            z_norm = z_next.detach()  # detach to避免梯度累積
            # z_norm = z_next # modify for bdpm

            if has_cat:
                log_z = self.p_sample(model_out_cat, log_z, t, out_dict)

        print()
        z_ohe = torch.exp(log_z).round()
        z_cat = log_z
        if has_cat:
            z_cat = ohe_to_categories(z_ohe, self.num_classes)
        sample = torch.cat([z_norm, z_cat], dim=1)#.cpu() # modify for bdpm

        sample = sample.detach()  # detach to avoid accumulating gradients
        torch.cuda.empty_cache()
        return sample, out_dict
    '''
    @torch.no_grad()
    def sample(self, num_samples, y_dist):
        b = num_samples
        device = self.log_alpha.device
        z_norm = torch.randn((b, self.num_numerical_features), device=device)

        has_cat = self.num_classes[0] != 0
        log_z = torch.zeros((b, 0), device=device).float()
        if has_cat:
            uniform_logits = torch.zeros((b, len(self.num_classes_expanded)), device=device)
            log_z = self.log_sample_categorical(uniform_logits)

        y = torch.multinomial(
            y_dist,
            num_samples=b,
            replacement=True
        )
        out_dict = {'y': y.long().to(device)}

        for i in reversed(range(0, self.num_timesteps)):
            print(f'Sample timestep {i:4d}', end='\r')
            t = torch.full((b,), i, device=device, dtype=torch.long)

            model_out = self._denoise_fn(
                torch.cat([z_norm, log_z], dim=1).float(),
                t,
                **out_dict
            )
            model_out_num = model_out[:, :self.num_numerical_features]
            model_out_cat = model_out[:, self.num_numerical_features:]

            # diffusion sampling
            z_next = self.gaussian_p_sample(model_out_num, z_norm, t, clip_denoised=False)['sample']

            # BDPM perturbation only for later steps
            if i < self.num_timesteps // 3:
                x0_hat_num = model_out_num.detach()
                perturb = self.bdpm(x0_hat_num, t)

                # 強化「CD 接近 0」的擾動方向
                if hasattr(self, 'classifier') and self.classifier is not None:
                    self.classifier.eval()
                    x_cls_orig = x0_hat_num
                    x_cls_perturbed = x0_hat_num + perturb

                    # Predict categorical class
                    # 將 logits 分割成對應類別欄位
                    cat_logits_split = torch.split(model_out_cat, list(self.num_classes), dim=1)  # list of 3 tensors: [b, 15], [b, 3], [b, 3]
                    # 對每欄取 argmax 得到預測類別 index
                    pred_cat = torch.stack(
                        [logits.argmax(dim=1) for logits in cat_logits_split],
                        dim=1  # shape: [b, 3]
                    )

                    x_cls_orig_clf = torch.cat([x_cls_orig, pred_cat], dim=1)
                    x_cls_perturbed_clf = torch.cat([x_cls_perturbed, pred_cat], dim=1)
                    x_cls_orig_clf = x_cls_orig_clf[:, torch.arange(x_cls_orig_clf.shape[1]) != x_cls_orig_clf.shape[1] - 2]  # remove slice_id
                    x_cls_orig_clf = x_cls_orig_clf.unsqueeze(1)
                    x_cls_perturbed_clf = x_cls_perturbed_clf[:, torch.arange(x_cls_perturbed_clf.shape[1]) != x_cls_perturbed_clf.shape[1] - 2]  # remove slice_id
                    x_cls_perturbed_clf = x_cls_perturbed_clf.unsqueeze(1)
                    logits_orig = self.classifier(x_cls_orig_clf)
                    logits_pert = self.classifier(x_cls_perturbed_clf)
                    if logits_orig.ndim == 3:
                        logits_orig = logits_orig.squeeze(1)
                        logits_pert = logits_pert.squeeze(1)

                    probs_orig = F.softmax(logits_orig, dim=1)
                    probs_pert = F.softmax(logits_pert, dim=1)
                    y = out_dict['y']
                    cd_orig = probs_orig[torch.arange(b), y] - probs_orig.max(dim=1)[0]
                    cd_pert = probs_pert[torch.arange(b), y] - probs_pert.max(dim=1)[0]

                    # 若 perturb 無幫助，則不使用
                    improve = (cd_pert.abs() < cd_orig.abs())  # 越靠近 0 越好
                    perturb = perturb * improve.float().unsqueeze(1)
                    # print('cd_orig.mean()', cd_orig.mean().item(), 'cd_pert.mean()', cd_pert.mean().item())
                    # print('improve.sum() / b =', improve.float().mean().item())
                                                            
                    # Adaptive scaling based on CD reduction ratio
                    gamma = 1.0
                    reduction_ratio = ((cd_orig.abs() - cd_pert.abs()) / cd_orig.abs()).clamp(min=0.0)
                    min_scale = 0.05  # 下限，避免 zero perturb
                    adaptive_scale = gamma * (reduction_ratio + min_scale).unsqueeze(1)
                    # adaptive_scale = gamma * reduction_ratio.unsqueeze(1)
                    perturb = perturb * adaptive_scale

                z_norm = z_next + perturb # z_next = z_next + perturb

            z_norm = z_next

            if has_cat:
                log_z = self.p_sample(model_out_cat, log_z, t, out_dict)

        print()

        z_ohe = torch.exp(log_z).round()
        z_cat = log_z
        if has_cat:
            z_cat = ohe_to_categories(z_ohe, self.num_classes)

        sample = torch.cat([z_norm, z_cat], dim=1)

        # 加入 classifier 分析 CD（信心差）是否靠近邊界
        if hasattr(self, 'classifier') and self.classifier is not None:
            with torch.no_grad():
                sample_clf = sample
                sample_clf = sample_clf[:, torch.arange(sample_clf.shape[1]) != sample_clf.shape[1] - 2]
                sample_clf = sample_clf.unsqueeze(1).float() if z_norm.ndim == 2 else z_norm
                logits = self.classifier(sample_clf)  # [B, 1, num_classes] or [B, num_classes]
                if logits.ndim == 3:
                    logits = logits.squeeze(1)

                probs = F.softmax(logits, dim=1)
                y = out_dict['y']
                true_logit = logits[torch.arange(b), y]
                probs_clone = logits.clone()
                probs_clone[torch.arange(b), y] = -float('inf')
                second_logit = probs_clone.max(dim=1)[0]
                cd = true_logit - second_logit

                print(f"\nClassifier CD Analysis:")
                print(f" - Mean CD:        {cd.mean().item():.4f}")
                print(f" - % near 0 (|CD|<1.0): {(cd.abs() < 1.0).float().mean().item() * 100:.2f}%")
                print(f" - % CD > +δ:      {(cd > 1.0).float().mean().item() * 100:.2f}%")
                print(f" - % CD < -δ:      {(cd < -1.0).float().mean().item() * 100:.2f}%")

        return sample.cpu(), out_dict


    def sample_all(self, num_samples, batch_size, y_dist, ddim=False):
        if ddim:
            print('Sample using DDIM.')
            sample_fn = self.sample_ddim
        else:
            sample_fn = self.sample
        
        b = batch_size

        all_y = []
        all_samples = []
        num_generated = 0
        while num_generated < num_samples:
            sample, out_dict = sample_fn(b, y_dist)
            mask_nan = torch.any(sample.isnan(), dim=1)
            sample = sample[~mask_nan]
            out_dict['y'] = out_dict['y'][~mask_nan]

            all_samples.append(sample)
            all_y.append(out_dict['y'].cpu())
            if sample.shape[0] != b:
                raise FoundNANsError
            num_generated += sample.shape[0]

        x_gen = torch.cat(all_samples, dim=0)[:num_samples]
        y_gen = torch.cat(all_y, dim=0)[:num_samples]

        return x_gen, y_gen
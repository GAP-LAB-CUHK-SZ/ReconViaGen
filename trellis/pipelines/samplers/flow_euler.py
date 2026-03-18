from typing import *
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from .base import Sampler
from .classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin
import math
from trellis.modules.spatial import patchify, unpatchify
from trellis.utils import render_utils, postprocessing_utils
from trellis.utils import loss_utils
import trellis.modules.sparse as sp
import torch.nn.functional as F


class FlowEulerSampler(Sampler):
    """
    Generate samples from a flow-matching model using Euler sampling.

    Args:
        sigma_min: The minimum scale of noise in flow.
    """
    def __init__(
        self,
        sigma_min: float,
    ):
        self.sigma_min = sigma_min

    def _eps_to_xstart(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * eps) / (1 - t)
  
    def _xstart_to_x_t(self, x_0, t, eps):
        assert x_0.shape == eps.shape
        return (1-t) * x_0 + (self.sigma_min + (1 - self.sigma_min) * t) * eps
        # return (1-t) * x_0  + t * eps + self.sigma_min * (1-t)  * eps 

    def _xstart_to_eps(self, x_t, t, x_0):
        assert x_t.shape == x_0.shape
        return (x_t - (1 - t) * x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _v_to_xstart_eps(self, x_t, t, v):
        assert x_t.shape == v.shape
        eps = (1 - t) * v + x_t
        x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v
        return x_0, eps
        
    def _xstart_to_v(self, x_0, x_t, t):
        assert x_0.shape == x_t.shape
        return (x_t - (1 - self.sigma_min) * x_0) / (self.sigma_min + (1 - self.sigma_min) * t)


    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        return model(x_t, t, cond, **kwargs)

    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        param = kwargs.pop("parameterization", "v")
        if param == "v":
            pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
            pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        elif param == "x0":
            pred_x_0 = self._inference_model(model, x_t, t, cond, **kwargs)
            pred_v = self._xstart_to_v(x_0=pred_x_0, x_t=x_t, t=t)
        return pred_x_0, None, pred_v

    def _get_model_gt(self, x_0, t, noise):
        gt_x_t = self._xstart_to_x_t(x_0, t, noise)
        gt_v = self._xstart_to_v(x_0, gt_x_t, t)
        return gt_x_t, gt_v

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method.
        
        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0, "pred_eps": pred_eps})

    def sample_once_opt(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method.
        
        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0, "pred_eps": pred_eps})

    def sample_ss_once_opt_delta_v(
        self,
        model,
        ss_decoder,
        learning_rate,
        ss,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method.
        
        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        torch.cuda.empty_cache()
        with torch.no_grad():
            pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        pred_v_opt = torch.nn.Parameter(pred_v.detach().clone())
        optimizer = torch.optim.Adam([pred_v_opt], betas=(0.5, 0.9), lr=learning_rate)
        total_steps = 5
        with tqdm(total=total_steps, disable=True, desc='Sparse Structure (opt): optimizing') as pbar:
            for step in range(total_steps):
                optimizer.zero_grad()
                pred_x_0, _ = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v_opt)
                logits = F.sigmoid(ss_decoder(pred_x_0))
                loss = 1 - (2 * (logits * ss.float()).sum() + 1) / (logits.sum() + ss.float().sum() + 1)
                loss.backward()
                optimizer.step()
                pbar.set_postfix({'loss': loss.item()})
                pbar.update()
        
        pred_x_prev = x_t - (t - t_prev) * pred_v_opt.detach()
        torch.cuda.empty_cache()
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0, "pred_eps": pred_eps})

    def sample_slat_once_opt_delta_v(
        self,
        model,
        slat_decoder_gs,
        slat_decoder_mesh,
        std, 
        mean,
        dreamsim_model,
        learning_rate,
        input_images,
        extrinsics, 
        intrinsics,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method.
        
        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        torch.cuda.empty_cache()
        with torch.no_grad():
            pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        pred_v_opt_feat = torch.nn.Parameter(pred_v.feats.detach().clone())
        optimizer = torch.optim.Adam([pred_v_opt_feat], betas=(0.5, 0.9), lr=learning_rate)
        pred_v_opt = sp.SparseTensor(feats=pred_v_opt_feat, coords=pred_v.coords)
        total_steps = 5
        image_resolution = 259
        input_images = F.interpolate(input_images, size=(image_resolution, image_resolution), mode='bilinear', align_corners=False)
        with tqdm(total=total_steps, disable=True, desc='Appearance (opt): optimizing') as pbar:
            for step in range(total_steps):
                optimizer.zero_grad()
                pred_x_0, _ = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v_opt)
                pred_gs = slat_decoder_gs(pred_x_0 * std + mean)
                # pred_mesh = slat_decoder_mesh(pred_x_0 * std + mean)
                rend_gs = render_utils.render_frames(pred_gs[0], extrinsics, intrinsics, {'resolution': image_resolution, 'bg_color': (0, 0, 0)}, need_depth=True, opt=True)['color']
                # rend_mesh = render_utils.render_frames_opt(pred_mesh[0], extrinsics, intrinsics, {'resolution': 518, 'bg_color': (0, 0, 0)}, need_depth=True, opt=True)['color']
                rend_gs = torch.stack(rend_gs, dim=0)
                loss_gs = loss_utils.l1_loss(rend_gs, input_images, size_average=False).mean(dim=(1,2,3)) + \
                    (1 - loss_utils.ssim(rend_gs, input_images, size_average=False)) + \
                        loss_utils.lpips(rend_gs, input_images, size_average=False).mean(dim=(1,2,3)) + \
                            dreamsim_model(rend_gs, input_images)
                loss_gs = loss_gs[loss_gs <= 0.8].mean()
                # loss_gs = (1 - loss_utils.ssim(rend_gs, input_images)) + loss_utils.lpips(rend_gs, input_images) + dreamsim_model(rend_gs, input_images).mean()
                # loss_mesh = loss_utils.l1_loss(rend_mesh, input_images) + 0.2 * (1 - loss_utils.ssim(rend_mesh, input_images)) + 0.2 * loss_utils.lpips(rend_mesh, input_images)
                loss = loss_gs + 0.2 * loss_utils.l1_loss(pred_v_opt_feat, pred_v.feats)
                loss.backward()
                optimizer.step()
                pbar.set_postfix({'loss': loss.item()})
                pbar.update()
        
        pred_x_prev = x_t - (t - t_prev) * pred_v_opt.detach()
        torch.cuda.empty_cache()
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0, "pred_eps": pred_eps})

    def sample_opt(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            out = self.sample_once_opt(model, sample, t, t_prev, cond, **kwargs)
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
        ret.samples = sample
        return ret

    def sample_ss_opt_delta_v(
        self,
        model,
        ss_decoder,
        ss_learning_rate,
        ss_start_t,
        ss,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        # def cosine_anealing(step, total_steps, start_lr, end_lr):
        #     return end_lr + 0.5 * (start_lr - end_lr) * (1 + np.cos(np.pi * step / total_steps))
        for i, (t, t_prev) in enumerate(tqdm(t_pairs, desc="Sampling", disable=not verbose)):
            if t > ss_start_t:
                out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
                sample = out.pred_x_prev
                ret.pred_x_t.append(out.pred_x_prev)
                ret.pred_x_0.append(out.pred_x_0)
            else:
                # learning_rate = cosine_anealing(i - int(np.where(t_seq <= start_t)[0].min()), int(steps - np.where(t_seq <= start_t)[0].min()), apperance_learning_rate, 1e-5)
                learning_rate = ss_learning_rate
                out = self.sample_ss_once_opt_delta_v(model, ss_decoder, ss_learning_rate, ss, sample, t, t_prev, cond, **kwargs)
                sample = out.pred_x_prev
                ret.pred_x_t.append(out.pred_x_prev)
                ret.pred_x_0.append(out.pred_x_0)
        ret.samples = sample
        return ret

    def sample_slat_opt_delta_v(
        self,
        model,
        slat_decoder_gs,
        slat_decoder_mesh,
        std,
        mean,
        dreamsim_model,
        apperance_learning_rate,
        start_t,
        input_images,
        extrinsics, 
        intrinsics,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        # def cosine_anealing(step, total_steps, start_lr, end_lr):
        #     return end_lr + 0.5 * (start_lr - end_lr) * (1 + np.cos(np.pi * step / total_steps))
        for i, (t, t_prev) in enumerate(tqdm(t_pairs, desc="Sampling", disable=not verbose)):
            if t > start_t:
                out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
                sample = out.pred_x_prev
                ret.pred_x_t.append(out.pred_x_prev)
                ret.pred_x_0.append(out.pred_x_0)
            else:
                # learning_rate = cosine_anealing(i - int(np.where(t_seq <= start_t)[0].min()), int(steps - np.where(t_seq <= start_t)[0].min()), apperance_learning_rate, 1e-5)
                learning_rate = apperance_learning_rate
                out = self.sample_slat_once_opt_delta_v(model, slat_decoder_gs, slat_decoder_mesh, std, mean, dreamsim_model, learning_rate, input_images, extrinsics, intrinsics, sample, t, t_prev, cond, **kwargs)
                sample = out.pred_x_prev
                ret.pred_x_t.append(out.pred_x_prev)
                ret.pred_x_0.append(out.pred_x_0)
        ret.samples = sample
        return ret

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
        ret.samples = sample
        return ret



class LatentMatchSampler(FlowEulerSampler):
    """
    Generate samples from a Bridge Matching model using Euler sampling.
    This sampler is designed for Latent Bridge Matching (LBM), where
    the target (x_1) for training is assumed to be sampled from a Gaussian distribution,
    and the source (x_0) for inference is also typically a Gaussian noise.

    Args:
        sigma_bridge: The sigma parameter for the Bridge Matching stochastic interpolant.
                      This controls the amount of stochasticity in the SDE (LBM paper Eq 1).

    """
    def __init__(
        self,
        sigma_bridge: float = 0.1,
        **kwargs
    ):
        # Call parent constructor with a dummy sigma_min.
        # sigma_min is specific to Flow Matching's interpolant, which we override.
        super().__init__(sigma_min=0.0, **kwargs)
        self.sigma_bridge = sigma_bridge

    # Override _xstart_to_x_t for Bridge Matching's stochastic interpolant
    # This method is used to generate gt_x_t for training.
    def _xstart_to_x_t(self, x_0: torch.Tensor, t: float, noise: torch.Tensor,  x_1: torch.Tensor) -> torch.Tensor:
        """
        Calculates x_t according to the Bridge Matching stochastic interpolant.
        This function is used during training to generate noisy samples x_t from
        paired x_0 and x_1 samples. The 'x_1' argument is crucial for Bridge Matching.

        Args:
            x_0: The source latent tensor (e.g., from a data distribution or Gaussian).
            t: The current timestep (float between 0 and 1).
            eps: A random noise tensor (epsilon).
            x_1: The target latent tensor. Required for Bridge Matching.

        Returns:
            The interpolated latent tensor x_t.
        """        
        # LBM interpolant formula: x_t = (1-t)x_0 + t*x_1 + sigma_bridge*sqrt(t*(1-t))*epsilon
        return (1 - t) * x_0 + t * x_1 + self.sigma_bridge * math.sqrt(t * (1 - t)) * noise

    # This method is used to calculate gt_v for training.
    def _xstart_to_v(self, x_0: torch.Tensor, x_t: torch.Tensor, t: float, x_1: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculates the ground truth drift (v) that the model should predict for Bridge Matching.
        This function is used in the training objective to define the target for the model.

        Args:
            x_0: The source latent tensor.
            x_t: The interpolated latent tensor at time t.
            t: The current timestep (float between 0 and 1).
            x_1: The target latent tensor. Required for Bridge Matching.

        Returns:
            The target drift tensor v.
        """
        if x_1 is None:
            # This branch should ideally not be hit during _get_model_gt for LBM.
            raise ValueError("For Bridge Matching's target drift calculation, x_1 (target latent) must be provided.")
        assert x_t.shape == x_1.shape, "x_t and x_1 shapes must match."
        # LBM drift formula: v = (x_1 - x_t) / (1 - t)
        # Add a small epsilon to (1-t) to prevent division by zero if t is exactly 1.
        epsilon_t = 1e-5 # Small epsilon for numerical stability
        return (x_t - x_0) / (t + epsilon_t)

    # Override _get_model_gt to provide ground truth for Bridge Matching training.
    # In this simplified case, x_1 is sampled from a Gaussian distribution.
    def _get_model_gt(self, x_0: torch.Tensor, t: float, x_1: torch.Tensor):
        """
        Calculates ground truth x_t and v_target for Bridge Matching training purposes.
        In this simplified case, x_1 is sampled from a Gaussian distribution.
        
        Args:
            x_0: The source latent tensor (e.g., from a data distribution, or another Gaussian).
            t: The current timestep.
            noise: A random noise tensor (epsilon).

        Returns:
            A tuple (gt_x_t, gt_v).
        """
        # Sample x_1 from a Gaussian distribution with the same shape as x_0
        # This simulates the target distribution being Gaussian.
        if isinstance(x_0, sp.SparseTensor):
            noise = sp.SparseTensor(
                feats=torch.randn_like(x_0.feats).to(x_0.feats.device),
                coords=x_0.coords,
            )
        else:
            noise = torch.randn_like(x_0).to(x_0.device)
        # For Bridge Matching, _xstart_to_x_t needs x_1
        gt_x_t = self._xstart_to_x_t(x_0, t, noise, x_1=x_1) 
        gt_v = self._xstart_to_v(x_0, gt_x_t, t, x_1=x_1) 
        return gt_x_t, gt_v

    # Override sample_once to include the stochastic term for SDE integration.
    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t: torch.Tensor,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ) -> edict:
        """
        Performs a single Euler step to sample x_{t_next} from x_t for Bridge Matching.
        The model is assumed to predict the drift 'v' as per LBM's formulation.
        
        Args:
            model: The model to sample from (should be trained for Bridge Matching).
            x_t: The [N x C x ...] tensor of current latent inputs at time t.
            t: The current timestep.
            t_next: The next timestep in the forward integration sequence (t+dt).
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            An edict containing:
            - 'pred_x_prev': The estimated latent tensor at t_next.
            - 'pred_x_0': A prediction of x_0 (may be None as direct derivation is complex in LBM).
            - 'pred_eps': A prediction of eps (may be None).
        """
        # Get model's prediction of the drift (v)
        # We use the parent's _get_model_prediction. Its _v_to_xstart_eps uses sigma_min,
        # which is a dummy value here. For LBM, pred_v is the main output.
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        
        # Calculate time step difference (dt)
        dt = t - t_prev # This is the forward step size
        
        # Sample noise for the stochastic part of the SDE
        # The SDE for LBM is dx_t = v(x_t, t) dt + sigma dB_t
        # For Euler, dB_t approx sqrt(dt) * Z, where Z ~ N(0,I)
        # noise_increment = sp.SparseTensor(
        #     feats=torch.randn_like(x_t.feats).to(x_t.feats.device),
        #     coords=x_t.coords,
        # )
        # if isinstance(x_t, sp.SparseTensor):
        #     noise_increment = sp.SparseTensor(
        #         feats=torch.randn_like(x_t.feats).to(x_t.feats.device),
        #         coords=x_t.coords,
        #     )
        # else:
        #     noise_increment = torch.randn_like(x_t).to(x_t.device)
        # noise_increment = noise_increment * self.sigma_bridge * torch.sqrt(torch.tensor(max(0.0, dt), device=x_t.device))
        # pred_x_prev = x_t - (t - t_prev) * pred_v - noise_increment
        pred_x_prev = x_t - (t - t_prev) * pred_v 
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0, "pred_eps": pred_eps})
    
class FlowMatchingSampler(FlowEulerSampler):
    """
    Implementation of Flow Matching using Euler sampling.
    Inherits from FlowEulerSampler and modifies key methods for flow matching.
    """
    def __init__(self, sigma_min: float = 0.0):
        super().__init__(sigma_min=sigma_min)

    def _compute_velocity(self, x_t: torch.Tensor, x_0: torch.Tensor, t: float) -> torch.Tensor:
        return ((1 - self.sigma_min) * x_t - x_0 ) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _get_model_gt(self, x_1: torch.Tensor, t: float, x_0: torch.Tensor = None):
        # TODO: Implement this method
        pass
        # """
        # Get ground truth for training.
        # Args:
        #     x_1: Target endpoint
        #     t: Time point
        #     noise: Initial noise to use as x_0
        # """
        # x_t = (1 - t) * x_0 + t * x_1
        # v = self._compute_velocity(x_t, x_0, t)
        # eps = x_t + (1 - t) * v  # Convert velocity to noise
        # return x_t, eps, v

    def _v_to_xstart_eps(self, x_t: torch.Tensor, t: float, v: torch.Tensor):
        """Convert velocity to x_0 and noise predictions"""
        eps = x_t + (1 - t) * v
        x_0 = self._eps_to_xstart(x_t, t, eps)
        return x_0, eps

    @torch.no_grad()
    def sample(
        self,
        model,
        x_1: torch.Tensor,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Generate samples by following the flow from noise to x_1.
        Args:
            model: The model to sample from
            x_1: Target endpoint
            cond: Conditional information
            steps: Number of sampling steps
            rescale_t: Time rescaling factor
            verbose: Whether to show progress bar
            **kwargs: Additional model arguments
        Returns:
            Dictionary containing sampling trajectory and predictions
        """
        # Initialize with noise as x_0
        noise = torch.randn_like(x_1)
        current_x = noise
        
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list(zip(t_seq[:-1], t_seq[1:]))
        
        ret = edict({
            "samples": None,
            "pred_x_t": [],
            "pred_x_0": []
        })
        
        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            out = self.sample_once(model, current_x, t, t_prev, cond, **kwargs)
            current_x = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
            
        ret.samples = current_x
        return ret

    def sample_once(
        self,
        model,
        x_t: torch.Tensor,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ) -> Dict:
        """
        Sample x_{t-1} from the model using Euler method.
        Args:
            model: The model to sample from
            x_t: Current state
            t: Current time
            t_prev: Next time step
            cond: Conditional information
            **kwargs: Additional model arguments
        Returns:
            Dictionary containing predictions
        """
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        pred_x_prev = x_t + (t_prev - t) * pred_v
        return edict({
            "pred_x_prev": pred_x_prev,
            "pred_x_0": pred_x_0,
            "pred_eps": pred_eps
        })

class FlowEulerCfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, **kwargs)


class FlowEulerGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            cfg_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, cfg_interval=cfg_interval, **kwargs)

    def sample_opt(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            cfg_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample_opt(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, cfg_interval=cfg_interval, **kwargs)

    def sample_ss_opt_delta_v(
        self,
        model,
        ss_decoder,
        ss_learning_rate,
        ss_start_t,
        ss,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            cfg_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample_ss_opt_delta_v(model, ss_decoder, ss_learning_rate, ss_start_t, ss, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, cfg_interval=cfg_interval, **kwargs)


    def sample_slat_opt_delta_v(
        self,
        model,
        slat_decoder_gs,
        slat_decoder_mesh,
        std,
        mean,
        dreamsim_model,
        apperance_learning_rate,
        start_t,
        input_images,
        extrinsics, 
        intrinsics,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            cfg_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample_slat_opt_delta_v(model, slat_decoder_gs, slat_decoder_mesh, std, mean, dreamsim_model, apperance_learning_rate, start_t, input_images, extrinsics, intrinsics,noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, cfg_interval=cfg_interval, **kwargs)


class LatentMatchGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, LatentMatchSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            cfg_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, cfg_interval=cfg_interval, **kwargs)

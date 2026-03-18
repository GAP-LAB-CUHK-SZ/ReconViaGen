from typing import *
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from .base import Sampler
from .classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin
import trellis.modules.sparse as sp
from trellis.modules.spatial import patchify, unpatchify

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
    
    def _xstart_to_x_t(self, x_0, t, eps):
        assert x_0.shape == eps.shape
        return (1-t) * x_0 + (self.sigma_min + (1 - self.sigma_min) * t) * eps

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
        return model(x_t.to(torch.float32), t, cond, **kwargs)

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
    
    @torch.no_grad()
    def sample_once_featurevolume(
        self,
        model,
        cond_model,
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
        if isinstance(cond, sp.SparseTensor):
            t_tmp = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=x_t.dtype)
            t_embed = model.t_embedder(t_tmp).to(x_t.dtype)
            for block in cond_model:
                cond = block(cond, t_embed)
            if model.pe_mode == "ape":
                cond = cond + model.pos_embedder(cond.coords[:, 1:]).to(x_t.dtype)
            if 'neg_cond' in kwargs.keys():
                neg_cond = kwargs['neg_cond']
                for block in cond_model:
                    neg_cond = block(neg_cond, t_embed)
                if model.pe_mode == "ape":
                    neg_cond = neg_cond + model.pos_embedder(neg_cond.coords[:, 1:]).to(x_t.dtype)
                kwargs['neg_cond'] = neg_cond
        else:
            for block in cond_model:
                cond = block(cond)
            cond = patchify(cond, model.patch_size)
            cond = cond.view(*cond.shape[:2], -1).permute(0, 2, 1).contiguous()
            cond = cond + model.pos_emb[None].type(model.dtype)
            if 'neg_cond' in kwargs.keys():
                neg_cond = kwargs['neg_cond']
                for block in cond_model:
                    neg_cond = block(neg_cond)
                neg_cond = patchify(neg_cond, model.patch_size)
                neg_cond = neg_cond.view(*neg_cond.shape[:2], -1).permute(0, 2, 1).contiguous()
                neg_cond = neg_cond + model.pos_emb[None].type(model.dtype)
                kwargs['neg_cond'] = neg_cond
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0, "pred_eps": pred_eps})

    @torch.no_grad()
    def sample_featurevolume(
        self,
        model,
        cond_model,
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
            out = self.sample_once_featurevolume(model, cond_model, sample, t, t_prev, cond, **kwargs)
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

    @torch.no_grad()
    def sample_featurevolume(
        self,
        model,
        cond_model,
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
        return super().sample_featurevolume(model, cond_model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, cfg_interval=cfg_interval, **kwargs)

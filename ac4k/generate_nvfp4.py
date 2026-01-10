import os
import sys
import time
import warnings
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
import torch
import random
import argparse
import logging
from wan.utils.utils import save_video, str2bool
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS

warnings.filterwarnings('ignore')


EXAMPLE_PROMPT = {
    "t2v-A14B": {
        "prompt":
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "i2v-A14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
    },
}


def _validate_args(args):
    assert args.quantized_ckpt_dir is not None, "Please specify the quantized checkpoint directory (--quantized_ckpt_dir)."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
    if args.image is None and "image" in EXAMPLE_PROMPT[args.task]:
        args.image = EXAMPLE_PROMPT[args.task]["image"]

    if args.task == "i2v-A14B":
        assert args.image is not None, "Please specify the image path for i2v."

    cfg = WAN_CONFIGS[args.task]

    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift

    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale

    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    if not 's2v' in args.task:
        assert args.size in SUPPORTED_SIZES[
            args.
            task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a video from a text prompt using Wan2.2 with NVFP4 quantization"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-A14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames of video are generated. The number should be 4n+1"
    )
    parser.add_argument(
        "--quantized_ckpt_dir",
        type=str,
        default=None,
        help="The path to the quantized checkpoint directory containing DiT weights (low_noise and high_noise).")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the video from.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the video.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=None,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--convert_model_dtype",
        action="store_true",
        default=False,
        help="Whether to convert model paramerters dtype.")
    parser.add_argument(
        "--enable_hooks",
        action="store_true",
        default=False,
        help="Whether to enable hooks for runtime dumping."
    )
    args = parser.parse_args()
    _validate_args(args)

    return args


def _init_logging(rank):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


class WanT2VNVFP4:
    """
    WanT2V with NVFP4 quantization support.
    T5 and VAE use original data types, DiT models use NVFP4.
    """

    def __init__(
        self,
        config,
        quantized_ckpt_dir,
        device_id=0,
        rank=0,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
    ):
        """
        Initializes the Wan text-to-video generation model components with NVFP4 quantization.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            quantized_ckpt_dir (`str`):
                Path to directory containing quantized DiT model checkpoints (low_noise and high_noise)
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.boundary = config.boundary
        self.param_dtype = config.param_dtype

        from wan.modules.t5 import T5EncoderModel
        from wan.modules.vae2_1 import Wan2_1_VAE

        # Timing: T5 loading
        t_start = time.time()
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(
                quantized_ckpt_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(
                quantized_ckpt_dir, config.t5_tokenizer),
            shard_fn=None)
        t_t5 = time.time() - t_start
        logging.info(f"[TIMING] T5 model loading: {t_t5:.2f}s")

        # Timing: VAE loading
        t_start = time.time()
        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(quantized_ckpt_dir, config.vae_checkpoint),
            device=self.device)
        t_vae = time.time() - t_start
        logging.info(f"[TIMING] VAE model loading: {t_vae:.2f}s")

        logging.info(
            f"Loading quantized DiT models from: {quantized_ckpt_dir}")
        from wan.modules.ac4k_quant_model import create_quantized_wan_model

        # Timing: Low noise model loading
        t_start = time.time()
        self.low_noise_model = create_quantized_wan_model(
            quantized_ckpt_dir=quantized_ckpt_dir,
            subfolder=config.low_noise_checkpoint)
        t_low_load = time.time() - t_start
        logging.info(f"[TIMING] Low noise model loading: {t_low_load:.2f}s")

        t_start = time.time()
        self.low_noise_model = self._configure_model(
            model=self.low_noise_model,
            convert_model_dtype=convert_model_dtype)
        t_low_config = time.time() - t_start
        logging.info(
            f"[TIMING] Low noise model configuration: {t_low_config:.2f}s")

        # Timing: High noise model loading
        t_start = time.time()
        self.high_noise_model = create_quantized_wan_model(
            quantized_ckpt_dir=quantized_ckpt_dir,
            subfolder=config.high_noise_checkpoint)
        t_high_load = time.time() - t_start
        logging.info(f"[TIMING] High noise model loading: {t_high_load:.2f}s")

        t_start = time.time()
        self.high_noise_model = self._configure_model(
            model=self.high_noise_model,
            convert_model_dtype=convert_model_dtype)
        t_high_config = time.time() - t_start
        logging.info(
            f"[TIMING] High noise model configuration: {t_high_config:.2f}s")

        logging.info(
            f"[TIMING] Total model initialization: {t_t5 + t_vae + t_low_load + t_low_config + t_high_load + t_high_config:.2f}s")

        self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt

    def _configure_model(self, model, convert_model_dtype=False):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.
        """
        model.eval().requires_grad_(False)

        if convert_model_dtype:
            model.to(self.param_dtype)
        if not self.init_on_cpu:
            model.to(self.device)

        return model

    def _prepare_model_for_timestep(self, t, boundary, offload_model):
        r"""
        Prepares and returns the required model for the current timestep.
        """
        t_start = time.time()
        if t.item() >= boundary:
            required_model_name = 'high_noise_model'
            offload_model_name = 'low_noise_model'
        else:
            required_model_name = 'low_noise_model'
            offload_model_name = 'high_noise_model'
        if offload_model or self.init_on_cpu:
            if next(getattr(
                    self,
                    offload_model_name).parameters()).device.type == 'cuda':
                getattr(self, offload_model_name).to('cpu')
            if next(getattr(
                    self,
                    required_model_name).parameters()).device.type == 'cpu':
                getattr(self, required_model_name).to(self.device)
        t_elapsed = time.time() - t_start
        if t_elapsed > 0.1:  # Only log if it takes significant time
            logging.info(
                f"[TIMING] Model switch ({required_model_name}): {t_elapsed:.3f}s")
        return getattr(self, required_model_name)

    def generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        """
        Generates video frames from text prompt using diffusion process with NVFP4 quantization.
        """
        import math
        from contextlib import contextmanager
        from tqdm import tqdm
        from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
        from wan.utils.fm_solvers import (
            FlowDPMSolverMultistepScheduler,
            get_sampling_sigmas,
            retrieve_timesteps,
        )
        torch.cuda.reset_peak_memory_stats()
        guide_scale = (guide_scale, guide_scale) if isinstance(
            guide_scale, float) else guide_scale
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        # Timing: T5 encoding
        t_start = time.time()
        if not self.t5_cpu:
            t_move = time.time()
            self.text_encoder.model.to(self.device)
            t_move_elapsed = time.time() - t_move
            if t_move_elapsed > 0.1:
                logging.info(
                    f"[TIMING] T5 model move to GPU: {t_move_elapsed:.3f}s")

            t_encode = time.time()
            context = self.text_encoder([input_prompt], self.device)
            t_encode_elapsed = time.time() - t_encode
            logging.info(f"[TIMING] T5 encode prompt: {t_encode_elapsed:.3f}s")

            t_encode_null = time.time()
            context_null = self.text_encoder([n_prompt], self.device)
            t_encode_null_elapsed = time.time() - t_encode_null
            logging.info(
                f"[TIMING] T5 encode negative prompt: {t_encode_null_elapsed:.3f}s")

            if offload_model:
                self.text_encoder.model.cpu()
        else:
            t_encode = time.time()
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]
            t_encode_elapsed = time.time() - t_encode
            logging.info(f"[TIMING] T5 encode (CPU): {t_encode_elapsed:.3f}s")

        t_t5_total = time.time() - t_start
        logging.info(f"[TIMING] T5 encoding total: {t_t5_total:.3f}s")

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_low_noise = getattr(self.low_noise_model, 'no_sync',
                                    noop_no_sync)
        no_sync_high_noise = getattr(self.high_noise_model, 'no_sync',
                                     noop_no_sync)

        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync_low_noise(),
                no_sync_high_noise(),
        ):
            boundary = self.boundary * self.num_train_timesteps

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            latents = noise

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            # Timing: Sampling loop
            t_sampling_start = time.time()
            total_model_time = 0.0
            total_cond_time = 0.0
            total_uncond_time = 0.0
            total_scheduler_time = 0.0

            for step_idx, t in enumerate(tqdm(timesteps, desc="Sampling")):
                t_step_start = time.time()
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                t_model_prep = time.time()
                model = self._prepare_model_for_timestep(
                    t, boundary, offload_model)
                t_model_prep_elapsed = time.time() - t_model_prep
                total_model_time += t_model_prep_elapsed

                sample_guide_scale = guide_scale[1] if t.item(
                ) >= boundary else guide_scale[0]

                t_cond = time.time()
                noise_pred_cond = model(
                    latent_model_input, t=timestep, **arg_c)[0]
                t_cond_elapsed = time.time() - t_cond
                total_cond_time += t_cond_elapsed

                t_uncond = time.time()
                noise_pred_uncond = model(
                    latent_model_input, t=timestep, **arg_null)[0]
                t_uncond_elapsed = time.time() - t_uncond
                total_uncond_time += t_uncond_elapsed

                noise_pred = noise_pred_uncond + sample_guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                t_scheduler = time.time()
                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                t_scheduler_elapsed = time.time() - t_scheduler
                total_scheduler_time += t_scheduler_elapsed

                latents = [temp_x0.squeeze(0)]

                t_step_elapsed = time.time() - t_step_start
                if step_idx % 10 == 0 or step_idx == len(timesteps) - 1:
                    logging.info(f"[TIMING] Step {step_idx}/{len(timesteps)-1}: "
                                 f"total={t_step_elapsed:.3f}s, "
                                 f"model_prep={t_model_prep_elapsed:.3f}s, "
                                 f"cond={t_cond_elapsed:.3f}s, "
                                 f"uncond={t_uncond_elapsed:.3f}s, "
                                 f"scheduler={t_scheduler_elapsed:.3f}s")

            t_sampling_total = time.time() - t_sampling_start
            logging.info(
                f"[TIMING] Sampling loop total: {t_sampling_total:.2f}s")
            logging.info(f"[TIMING] Sampling breakdown - "
                         f"model_prep: {total_model_time:.2f}s, "
                         f"cond: {total_cond_time:.2f}s, "
                         f"uncond: {total_uncond_time:.2f}s, "
                         f"scheduler: {total_scheduler_time:.2f}s")

            x0 = latents
            if offload_model:
                t_offload = time.time()
                self.low_noise_model.cpu()
                self.high_noise_model.cpu()
                torch.cuda.empty_cache()
                t_offload_elapsed = time.time() - t_offload
                logging.info(
                    f"[TIMING] Model offload to CPU: {t_offload_elapsed:.3f}s")

            # Timing: VAE decoding
            if self.rank == 0:
                t_vae_decode = time.time()
                videos = self.vae.decode(x0)
                t_vae_decode_elapsed = time.time() - t_vae_decode
                logging.info(
                    f"[TIMING] VAE decode: {t_vae_decode_elapsed:.3f}s")

        del noise, latents
        del sample_scheduler
        if offload_model:
            import gc
            gc.collect()
            torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated()
        logging.info(f"Peak GPU memory: {peak_memory / 1024**3:.2f} GB")
        return videos[0] if self.rank == 0 else None


class WanI2VNVFP4:
    """
    WanI2V with NVFP4 quantization support.
    T5 and VAE use original data types, DiT models use NVFP4.
    """

    def __init__(
        self,
        config,
        quantized_ckpt_dir,
        device_id=0,
        rank=0,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
    ):
        r"""
        Initializes the Wan image-to-video generation model components with NVFP4 quantization.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            quantized_ckpt_dir (`str`):
                Path to directory containing quantized DiT model checkpoints (low_noise and high_noise)
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.boundary = config.boundary
        self.param_dtype = config.param_dtype

        from wan.modules.t5 import T5EncoderModel
        from wan.modules.vae2_1 import Wan2_1_VAE

        # Timing: T5 loading
        t_start = time.time()
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(
                quantized_ckpt_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(
                quantized_ckpt_dir, config.t5_tokenizer),
            shard_fn=None)
        t_t5 = time.time() - t_start
        logging.info(f"[TIMING] T5 model loading: {t_t5:.2f}s")

        # Timing: VAE loading
        t_start = time.time()
        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(quantized_ckpt_dir, config.vae_checkpoint),
            device=self.device)
        t_vae = time.time() - t_start
        logging.info(f"[TIMING] VAE model loading: {t_vae:.2f}s")

        # Load quantized DiT models from quantized_ckpt_dir
        logging.info(
            f"Loading quantized DiT models from: {quantized_ckpt_dir}")
        from wan.modules.ac4k_quant_model import create_quantized_wan_model

        # Timing: Low noise model loading
        t_start = time.time()
        self.low_noise_model = create_quantized_wan_model(
            quantized_ckpt_dir=quantized_ckpt_dir,
            subfolder=config.low_noise_checkpoint)
        t_low_load = time.time() - t_start
        logging.info(f"[TIMING] Low noise model loading: {t_low_load:.2f}s")

        t_start = time.time()
        self.low_noise_model = self._configure_model(
            model=self.low_noise_model,
            convert_model_dtype=convert_model_dtype)
        t_low_config = time.time() - t_start
        logging.info(
            f"[TIMING] Low noise model configuration: {t_low_config:.2f}s")

        # Timing: High noise model loading
        t_start = time.time()
        self.high_noise_model = create_quantized_wan_model(
            quantized_ckpt_dir=quantized_ckpt_dir,
            subfolder=config.high_noise_checkpoint)
        t_high_load = time.time() - t_start
        logging.info(f"[TIMING] High noise model loading: {t_high_load:.2f}s")

        t_start = time.time()
        self.high_noise_model = self._configure_model(
            model=self.high_noise_model,
            convert_model_dtype=convert_model_dtype)
        t_high_config = time.time() - t_start
        logging.info(
            f"[TIMING] High noise model configuration: {t_high_config:.2f}s")

        logging.info(
            f"[TIMING] Total model initialization: {t_t5 + t_vae + t_low_load + t_low_config + t_high_load + t_high_config:.2f}s")

        self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt

    def _configure_model(self, model, convert_model_dtype=False):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.
        """
        model.eval().requires_grad_(False)

        if convert_model_dtype:
            model.to(self.param_dtype)
        if not self.init_on_cpu:
            model.to(self.device)

        return model

    def _prepare_model_for_timestep(self, t, boundary, offload_model):
        r"""
        Prepares and returns the required model for the current timestep.
        """
        t_start = time.time()
        if t.item() >= boundary:
            required_model_name = 'high_noise_model'
            offload_model_name = 'low_noise_model'
        else:
            required_model_name = 'low_noise_model'
            offload_model_name = 'high_noise_model'
        if offload_model or self.init_on_cpu:
            if next(getattr(
                    self,
                    offload_model_name).parameters()).device.type == 'cuda':
                getattr(self, offload_model_name).to('cpu')
            if next(getattr(
                    self,
                    required_model_name).parameters()).device.type == 'cpu':
                getattr(self, required_model_name).to(self.device)
        t_elapsed = time.time() - t_start
        if t_elapsed > 0.1:  # Only log if it takes significant time
            logging.info(
                f"[TIMING] Model switch ({required_model_name}): {t_elapsed:.3f}s")
        return getattr(self, required_model_name)

    def generate(self,
                 input_prompt,
                 img,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        """
        Generates video frames from input image and text prompt using diffusion process with NVFP4 quantization.
        """
        import math
        import numpy as np
        from contextlib import contextmanager
        from tqdm import tqdm
        import torchvision.transforms.functional as TF
        from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
        from wan.utils.fm_solvers import (
            FlowDPMSolverMultistepScheduler,
            get_sampling_sigmas,
            retrieve_timesteps,
        )
        torch.cuda.reset_peak_memory_stats()

        # preprocess
        guide_scale = (guide_scale, guide_scale) if isinstance(
            guide_scale, float) else guide_scale
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]

        max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16,
            (F - 1) // self.vae_stride[0] + 1,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ],
            dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # Timing: T5 encoding
        t_start = time.time()
        if not self.t5_cpu:
            t_move = time.time()
            self.text_encoder.model.to(self.device)
            t_move_elapsed = time.time() - t_move
            if t_move_elapsed > 0.1:
                logging.info(
                    f"[TIMING] T5 model move to GPU: {t_move_elapsed:.3f}s")

            t_encode = time.time()
            context = self.text_encoder([input_prompt], self.device)
            t_encode_elapsed = time.time() - t_encode
            logging.info(f"[TIMING] T5 encode prompt: {t_encode_elapsed:.3f}s")

            t_encode_null = time.time()
            context_null = self.text_encoder([n_prompt], self.device)
            t_encode_null_elapsed = time.time() - t_encode_null
            logging.info(
                f"[TIMING] T5 encode negative prompt: {t_encode_null_elapsed:.3f}s")

            if offload_model:
                self.text_encoder.model.cpu()
        else:
            t_encode = time.time()
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]
            t_encode_elapsed = time.time() - t_encode
            logging.info(f"[TIMING] T5 encode (CPU): {t_encode_elapsed:.3f}s")

        t_t5_total = time.time() - t_start
        logging.info(f"[TIMING] T5 encoding total: {t_t5_total:.3f}s")

        # Timing: VAE encoding
        t_vae_encode = time.time()
        y = self.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(
                    img[None].cpu(), size=(h, w), mode='bicubic').transpose(
                        0, 1),
                torch.zeros(3, F - 1, h, w)
            ],
                dim=1).to(self.device)
        ])[0]
        y = torch.concat([msk, y])
        t_vae_encode_elapsed = time.time() - t_vae_encode
        logging.info(f"[TIMING] VAE encode: {t_vae_encode_elapsed:.3f}s")

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_low_noise = getattr(self.low_noise_model, 'no_sync',
                                    noop_no_sync)
        no_sync_high_noise = getattr(self.high_noise_model, 'no_sync',
                                     noop_no_sync)

        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync_low_noise(),
                no_sync_high_noise(),
        ):
            boundary = self.boundary * self.num_train_timesteps

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latent = noise

            arg_c = {
                'context': [context[0]],
                'seq_len': max_seq_len,
                'y': [y],
            }

            arg_null = {
                'context': context_null,
                'seq_len': max_seq_len,
                'y': [y],
            }

            # Timing: Sampling loop
            t_sampling_start = time.time()
            total_model_time = 0.0
            total_cond_time = 0.0
            total_uncond_time = 0.0
            total_scheduler_time = 0.0

            for step_idx, t in enumerate(tqdm(timesteps, desc="Sampling")):
                t_step_start = time.time()
                latent_model_input = [latent.to(self.device)]
                timestep = [t]

                timestep = torch.stack(timestep).to(self.device)

                t_model_prep = time.time()
                model = self._prepare_model_for_timestep(
                    t, boundary, offload_model)
                t_model_prep_elapsed = time.time() - t_model_prep
                total_model_time += t_model_prep_elapsed

                sample_guide_scale = guide_scale[1] if t.item(
                ) >= boundary else guide_scale[0]

                t_cond = time.time()
                noise_pred_cond = model(
                    latent_model_input, t=timestep, **arg_c)[0]
                t_cond_elapsed = time.time() - t_cond
                total_cond_time += t_cond_elapsed
                if offload_model:
                    torch.cuda.empty_cache()

                t_uncond = time.time()
                noise_pred_uncond = model(
                    latent_model_input, t=timestep, **arg_null)[0]
                t_uncond_elapsed = time.time() - t_uncond
                total_uncond_time += t_uncond_elapsed
                if offload_model:
                    torch.cuda.empty_cache()

                noise_pred = noise_pred_uncond + sample_guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                t_scheduler = time.time()
                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                t_scheduler_elapsed = time.time() - t_scheduler
                total_scheduler_time += t_scheduler_elapsed

                latent = temp_x0.squeeze(0)
                x0 = [latent]
                del latent_model_input, timestep

                t_step_elapsed = time.time() - t_step_start
                if step_idx % 10 == 0 or step_idx == len(timesteps) - 1:
                    logging.info(f"[TIMING] Step {step_idx}/{len(timesteps)-1}: "
                                 f"total={t_step_elapsed:.3f}s, "
                                 f"model_prep={t_model_prep_elapsed:.3f}s, "
                                 f"cond={t_cond_elapsed:.3f}s, "
                                 f"uncond={t_uncond_elapsed:.3f}s, "
                                 f"scheduler={t_scheduler_elapsed:.3f}s")

            t_sampling_total = time.time() - t_sampling_start
            logging.info(
                f"[TIMING] Sampling loop total: {t_sampling_total:.2f}s")
            logging.info(f"[TIMING] Sampling breakdown - "
                         f"model_prep: {total_model_time:.2f}s, "
                         f"cond: {total_cond_time:.2f}s, "
                         f"uncond: {total_uncond_time:.2f}s, "
                         f"scheduler: {total_scheduler_time:.2f}s")

            if offload_model:
                t_offload = time.time()
                self.low_noise_model.cpu()
                self.high_noise_model.cpu()
                torch.cuda.empty_cache()
                t_offload_elapsed = time.time() - t_offload
                logging.info(
                    f"[TIMING] Model offload to CPU: {t_offload_elapsed:.3f}s")

            # Timing: VAE decoding
            if self.rank == 0:
                t_vae_decode = time.time()
                videos = self.vae.decode(x0)
                t_vae_decode_elapsed = time.time() - t_vae_decode
                logging.info(
                    f"[TIMING] VAE decode: {t_vae_decode_elapsed:.3f}s")

        del noise, latent, x0
        del sample_scheduler
        if offload_model:
            import gc
            gc.collect()
            torch.cuda.synchronize()

        peak_memory = torch.cuda.max_memory_allocated()
        logging.info(f"Peak GPU memory: {peak_memory / 1024**3:.2f} GB")
        return videos[0] if self.rank == 0 else None


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)

    cfg = WAN_CONFIGS[args.task]

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    logging.info(f"Input prompt: {args.prompt}")
    img = None
    if args.image is not None:
        img = Image.open(args.image).convert("RGB")
        logging.info(f"Input image: {args.image}")

    if "t2v" in args.task:
        # Timing: Pipeline initialization
        t_init_start = time.time()
        logging.info("Creating WanT2V pipeline with NVFP4 quantization.")
        wan_t2v = WanT2VNVFP4(
            config=cfg,
            quantized_ckpt_dir=args.quantized_ckpt_dir,
            device_id=device,
            rank=rank,
            convert_model_dtype=args.convert_model_dtype,
        )
        t_init_total = time.time() - t_init_start
        logging.info(
            f"[TIMING] Pipeline initialization total: {t_init_total:.2f}s")

        if args.enable_hooks and rank == 0:
            from wan.utils.hook_utils import FSDPRuntimeDumper
            dumper = FSDPRuntimeDumper()
            dumper.register_hooks(
                wan_t2v.text_encoder.model, prefix="t5_encoder")
            dumper.register_hooks(wan_t2v.vae.model, prefix="vae")
            dumper.register_hooks(wan_t2v.high_noise_model,
                                  prefix="high_noise_model")
            dumper.register_hooks(wan_t2v.low_noise_model,
                                  prefix="low_noise_model")

        # Timing: Video generation
        t_gen_start = time.time()
        logging.info(f"Generating video ...")
        video = wan_t2v.generate(
            args.prompt,
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)
        t_gen_total = time.time() - t_gen_start
        logging.info(f"[TIMING] Video generation total: {t_gen_total:.2f}s")
    elif "i2v" in args.task:
        # Timing: Pipeline initialization
        t_init_start = time.time()
        logging.info("Creating WanI2V pipeline with NVFP4 quantization.")
        wan_i2v = WanI2VNVFP4(
            config=cfg,
            quantized_ckpt_dir=args.quantized_ckpt_dir,
            device_id=device,
            rank=rank,
            convert_model_dtype=args.convert_model_dtype,
        )
        t_init_total = time.time() - t_init_start
        logging.info(
            f"[TIMING] Pipeline initialization total: {t_init_total:.2f}s")

        if args.enable_hooks and rank == 0:
            from wan.utils.hook_utils import FSDPRuntimeDumper
            dumper = FSDPRuntimeDumper()
            dumper.register_hooks(
                wan_i2v.text_encoder.model, prefix="t5_encoder")
            dumper.register_hooks(wan_i2v.vae.model, prefix="vae")
            dumper.register_hooks(wan_i2v.high_noise_model,
                                  prefix="high_noise_model")
            dumper.register_hooks(wan_i2v.low_noise_model,
                                  prefix="low_noise_model")

        # Timing: Video generation
        t_gen_start = time.time()
        logging.info(f"Generating video ...")
        video = wan_i2v.generate(
            args.prompt,
            img,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)
        t_gen_total = time.time() - t_gen_start
        logging.info(f"[TIMING] Video generation total: {t_gen_total:.2f}s")
    else:
        raise NotImplementedError(
            f"Task {args.task} is not yet supported in NVFP4 mode.")

    if rank == 0:
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/",
                                                                     "_")[:50]
            suffix = '.mp4'
            task_suffix = "_NVFP4" if "NVFP4" not in args.task else ""
            args.save_file = f"{args.task}{task_suffix}_{args.size.replace('*','x') if sys.platform=='win32' else args.size}_{formatted_prompt}_{formatted_time}" + suffix

        logging.info(f"Saving generated video to {args.save_file}")
        save_video(
            tensor=video[None],
            save_file=args.save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))
    del video

    torch.cuda.synchronize()
    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)

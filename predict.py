import os, sys
from cog import BasePredictor, Input, Path
sys.path.append('/content/Open-Sora')
os.chdir('/content/Open-Sora')

import os
import torch
from mmengine.runner import set_random_seed
from opensora.datasets import save_sample
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.misc import to_torch_dtype

class Predictor(BasePredictor):
    def setup(self) -> None:
        num_frames = 16
        self.fps = 24 // 3
        image_size = (512, 512)
        self.model = dict(
            type="STDiT-XL/2",
            space_scale=1.0,
            time_scale=1.0,
            enable_flashattn=True,
            enable_layernorm_kernel=True,
            from_pretrained="/content/Open-Sora/models/OpenSora-v1-HQ-16x512x512.pth"
        )
        self.vae = dict(
            type="VideoAutoencoderKL",
            from_pretrained="stabilityai/sd-vae-ft-ema",
            micro_batch_size=128,
        )
        self.text_encoder = dict(
            type="t5",
            from_pretrained="/content/Open-Sora/pretrained_models/t5_ckpts",
            model_max_length=120,
        )
        self.scheduler = dict(
            type="iddpm",
            num_sampling_steps=100,
            cfg_scale=7.0,
        )
        self.dtype = "fp16"

        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = to_torch_dtype(self.dtype)

        input_size = (num_frames, *image_size)
        self.vae = build_module(self.vae, MODELS)
        self.latent_size = self.vae.get_latent_size(input_size)
        self.text_encoder = build_module(self.text_encoder, MODELS, device=self.device)  # T5 must be fp32
        self.model = build_module(
            self.model,
            MODELS,
            input_size=self.latent_size,
            in_channels=self.vae.out_channels,
            caption_channels=self.text_encoder.output_dim,
            model_max_length=self.text_encoder.model_max_length,
            dtype=self.dtype,
            enable_sequence_parallelism=False,
        )
        self.text_encoder.y_embedder = self.model.y_embedder  # hack for classifier-free guidance
        self.vae = self.vae.to(self.device, self.dtype).eval()
        self.model = self.model.to(self.device, self.dtype).eval()
        self.scheduler = build_module(self.scheduler, SCHEDULERS)
    def predict(
        self,
        prompt: str = Input(default="A serene underwater scene featuring a sea turtle swimming through a coral reef. The turtle, with its greenish-brown shell, is the main focus of the video, swimming gracefully towards the right side of the frame. The coral reef, teeming with life, is visible in the background, providing a vibrant and colorful backdrop to the turtle's journey. Several small fish, darting around the turtle, add a sense of movement and dynamism to the scene. The video is shot from a slightly elevated angle, providing a comprehensive view of the turtle's surroundings. The overall style of the video is calm and peaceful, capturing the beauty and tranquility of the underwater world."),
        seed: int = Input(default=1234),
    ) -> Path:
        set_random_seed(seed=seed)
        prompts=[prompt]
        samples = self.scheduler.sample(
            self.model,
            self.text_encoder,
            z_size=(self.vae.out_channels, *self.latent_size),
            prompts=prompts,
            device=self.device
        )
        samples = self.vae.decode(samples.to(self.dtype))
        save_sample(samples[0], fps=self.fps, save_path='/content/output')
        return Path('/content/output.mp4')
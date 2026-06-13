"""
Interactive frame-by-frame world model interface.
Turns the Oasis diffusion model into a real-time playable experience.
"""

import torch
from dit import DiT_models
from vae import VAE_models
from utils import load_prompt, sigmoid_beta_schedule
from einops import rearrange
from torch import autocast
from safetensors.torch import load_model
import os
import numpy as np

assert torch.cuda.is_available()
device = "cuda:0"


class InteractiveWorld:
    def __init__(
        self,
        oasis_ckpt="oasis500m.safetensors",
        vae_ckpt="vit-l-20.safetensors",
        num_frames=128,
        ddim_steps=3,
        noise_abs_max=20,
        stabilization_level=15,
        compile=False,
    ):
        self.num_frames = num_frames
        self.ddim_steps = ddim_steps
        self.noise_abs_max = noise_abs_max
        self.stabilization_level = stabilization_level
        self.max_noise_level = 1000
        self.scaling_factor = 0.07843137255

        # load DiT
        self.model = DiT_models["DiT-S/2"]()
        print(f"[InteractiveWorld] loading DiT from {oasis_ckpt}...")
        if oasis_ckpt.endswith(".pt"):
            ckpt = torch.load(oasis_ckpt, weights_only=True)
            self.model.load_state_dict(ckpt, strict=False)
        elif oasis_ckpt.endswith(".safetensors"):
            load_model(self.model, oasis_ckpt)
        self.model = self.model.to(device).eval()

        # load VAE
        self.vae = VAE_models["vit-l-20-shallow-encoder"]()
        print(f"[InteractiveWorld] loading VAE from {vae_ckpt}...")
        if vae_ckpt.endswith(".pt"):
            vae_ckpt_data = torch.load(vae_ckpt, weights_only=True)
            self.vae.load_state_dict(vae_ckpt_data)
        elif vae_ckpt.endswith(".safetensors"):
            load_model(self.vae, vae_ckpt)
        self.vae = self.vae.to(device).eval()

        if compile:
            try:
                print("[InteractiveWorld] torch.compile on DiT + VAE ...")
                self.model = torch.compile(self.model, mode="reduce-overhead")
                self.vae = torch.compile(self.vae, mode="reduce-overhead")
                print("[InteractiveWorld] compile done.")
            except Exception as e:
                print(f"[InteractiveWorld] torch.compile failed ({e}), continuing without it.")

        # precompute noise schedule
        self.betas = sigmoid_beta_schedule(self.max_noise_level).float().to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod = rearrange(self.alphas_cumprod, "T -> T 1 1 1")

        self.noise_range = torch.linspace(
            -1, self.max_noise_level - 1, self.ddim_steps + 1, device=device
        )

        self.reset_state()

    def reset_state(self):
        self.x = None  # latent frame buffer (B, T, C, H, W)
        self.actions = None  # action buffer (B, T, D)
        self.frame_count = 0
        self.initialized = False

    @torch.inference_mode()
    def initialize(self, prompt_path, n_prompt_frames=1):
        """Load an image/video prompt and encode it into latents."""
        self.reset_state()

        x = load_prompt(prompt_path, n_prompt_frames=n_prompt_frames)
        x = x.to(device)

        # VAE encode
        B = x.shape[0]
        H, W = x.shape[-2:]
        x = rearrange(x, "b t c h w -> (b t) c h w")
        with autocast("cuda", dtype=torch.half):
            x = self.vae.encode(x * 2 - 1).mean * self.scaling_factor
        x = rearrange(
            x,
            "(b t) (h w) c -> b t c h w",
            t=n_prompt_frames,
            h=H // self.vae.patch_size,
            w=W // self.vae.patch_size,
        )

        self.x = x
        self.frame_count = n_prompt_frames
        self.initialized = True

        # decode first frame for display
        frame = self._decode_frame(self.x[:, :1])
        return frame

    def _decode_frame(self, x_latent):
        """Decode a single (or batch of) latent frame(s) to RGB."""
        x = rearrange(x_latent, "b t c h w -> (b t) (h w) c")
        with autocast("cuda", dtype=torch.half):
            x = (self.vae.decode(x / self.scaling_factor) + 1) / 2
        x = rearrange(x, "(b t) c h w -> b t h w c", t=x_latent.shape[1])
        x = torch.clamp(x, 0, 1)
        x = (x * 255).byte()
        return x[0, -1].cpu().numpy()  # return last frame as HWC uint8 numpy

    @torch.inference_mode()
    def step(self, action_one_hot: torch.Tensor):
        """
        Generate one new frame conditioned on `action_one_hot`.
        action_one_hot: (25,) tensor of Minecraft actions.
        Returns: (H, W, 3) uint8 numpy array of the new frame.
        """
        if not self.initialized:
            raise RuntimeError("World not initialized. Call initialize() first.")

        B = self.x.shape[0]
        i = self.frame_count

        # append zero action for prompt frames, then current action
        if self.actions is None:
            # prompt frames have zero action
            prompt_actions = torch.zeros(
                B, self.x.shape[1], 25, device=device, dtype=torch.float32
            )
            self.actions = prompt_actions

        action_one_hot = action_one_hot.to(device).half().unsqueeze(0).unsqueeze(0)  # (1, 1, 25)
        self.actions = torch.cat([self.actions, action_one_hot], dim=1)

        # append noise chunk for new frame
        chunk = torch.randn((B, 1, *self.x.shape[-3:]), device=device, dtype=torch.half)
        chunk = torch.clamp(chunk, -self.noise_abs_max, +self.noise_abs_max)
        x = torch.cat([self.x, chunk], dim=1)

        curr_len = self.x.shape[1]
        start_frame = max(0, (curr_len + 1) - self.model.max_frames)

        # precompute full timestep tensors once per step (shared across noise iterations)
        t_ctx = torch.full((B, curr_len), self.stabilization_level - 1, dtype=torch.long, device=device)
        t_base = torch.cat([t_ctx, torch.zeros((B, 1), dtype=torch.long, device=device)], dim=1)
        t_next_base = torch.cat([t_ctx, torch.zeros((B, 1), dtype=torch.long, device=device)], dim=1)
        act_slice = self.actions[:, start_frame : curr_len + 1]

        # DDIM denoising loop for the new frame only
        for noise_idx in reversed(range(1, self.ddim_steps + 1)):
            t_base[:, -1] = self.noise_range[noise_idx]
            t_next_base[:, -1] = self.noise_range[noise_idx - 1]
            t_next_base[:, -1] = torch.where(t_next_base[:, -1] < 0, t_base[:, -1], t_next_base[:, -1])

            t_full = t_base[:, start_frame:]
            t_next_full = t_next_base[:, start_frame:]
            x_curr = x[:, start_frame:]

            with autocast("cuda", dtype=torch.half):
                v = self.model(x_curr, t_full, act_slice)

            acp_t = self.alphas_cumprod[t_full]
            acp_t_next = self.alphas_cumprod[t_next_full]
            acp_t_next[:, :-1] = 1.0
            if noise_idx == 1:
                acp_t_next[:, -1:] = 1.0

            x_start = acp_t.sqrt() * x_curr - (1 - acp_t).sqrt() * v
            x_noise = ((1 / acp_t).sqrt() * x_curr - x_start) / (1 / acp_t - 1).sqrt()
            x[:, -1:] = (acp_t_next.sqrt() * x_start + x_noise * (1 - acp_t_next).sqrt())[:, -1:]

        self.x = x
        self.frame_count += 1

        # prune old frames to keep within max_frames context window
        if self.x.shape[1] > self.model.max_frames:
            self.x = self.x[:, -self.model.max_frames :]
            self.actions = self.actions[:, -self.model.max_frames :]
            # adjust frame_count is tricky because it's used for indexing.
            # actually frame_count is only used for start_frame calculation which
            # uses i+1 - max_frames, so we can keep it as is.

        frame = self._decode_frame(self.x[:, -1:])
        return frame

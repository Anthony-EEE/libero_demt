import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from libero.lifelong.models.base_policy import BasePolicy


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_groups):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
            ),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size, n_groups):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups),
            ]
        )
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels * 2),
            nn.Unflatten(-1, (-1, 1)),
        )
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond):
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).reshape(x.shape[0], 2, self.out_channels, 1)
        out = embed[:, 0] * out + embed[:, 1]
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
    ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        cond_dim = diffusion_step_embed_dim + global_cond_dim

        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        self.down_modules = nn.ModuleList()
        for idx, (dim_in, dim_out) in enumerate(in_out):
            is_last = idx >= len(in_out) - 1
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in, dim_out, cond_dim, kernel_size, n_groups
                        ),
                        ConditionalResidualBlock1D(
                            dim_out, dim_out, cond_dim, kernel_size, n_groups
                        ),
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1)
                        if not is_last
                        else nn.Identity(),
                    ]
                )
            )

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim, mid_dim, cond_dim, kernel_size, n_groups
                ),
                ConditionalResidualBlock1D(
                    mid_dim, mid_dim, cond_dim, kernel_size, n_groups
                ),
            ]
        )

        self.up_modules = nn.ModuleList()
        for idx, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = idx >= len(in_out) - 2
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out + dim_in,
                            dim_in,
                            cond_dim,
                            kernel_size,
                            n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in, dim_in, cond_dim, kernel_size, n_groups
                        ),
                        nn.ConvTranspose1d(dim_in, dim_in, 4, 2, 1)
                        if not is_last
                        else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size, n_groups),
            nn.Conv1d(down_dims[0], input_dim, 1),
        )

    def forward(self, sample, timestep, global_cond):
        x = sample.moveaxis(-1, -2)
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=x.device)
        timestep = timestep.to(device=x.device, dtype=torch.long)
        if timestep.ndim == 0:
            timestep = timestep[None].expand(x.shape[0])
        global_feature = torch.cat(
            [self.diffusion_step_encoder(timestep), global_cond], dim=-1
        )

        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            if idx < len(self.down_modules) - 1:
                h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for resnet, resnet2, upsample in self.up_modules:
            skip = h.pop()
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1], mode="nearest")
            x = torch.cat((x, skip), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        return x.moveaxis(-1, -2)


def replace_bn_with_gn(module, features_per_group=16):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_groups = max(1, child.num_features // features_per_group)
            setattr(module, name, nn.GroupNorm(num_groups, child.num_features))
        else:
            replace_bn_with_gn(child, features_per_group)
    return module


class ResNet18Encoder(nn.Module):
    def __init__(self, output_dim, pretrained=False):
        super().__init__()
        resnet = torchvision.models.resnet18(pretrained=pretrained)
        resnet.fc = nn.Identity()
        self.backbone = replace_bn_with_gn(resnet)
        self.proj = nn.Linear(512, output_dim)

    def forward(self, x):
        return self.proj(self.backbone(x))


class DiffusionPolicy(BasePolicy):
    def __init__(self, cfg, shape_meta):
        super().__init__(cfg, shape_meta)
        policy_cfg = cfg.policy
        self.obs_horizon = policy_cfg.obs_horizon
        self.pred_horizon = min(policy_cfg.pred_horizon, cfg.data.seq_len)
        self.action_horizon = min(
            policy_cfg.action_horizon,
            max(1, self.pred_horizon - self.obs_horizon + 1),
        )
        self.action_dim = shape_meta["ac_dim"]
        self.num_train_timesteps = policy_cfg.get(
            "num_train_timesteps", policy_cfg.num_diffusion_iters
        )
        self.num_inference_iters = policy_cfg.get(
            "num_inference_iters", policy_cfg.num_diffusion_iters
        )
        self.inference_scheduler_name = policy_cfg.get("inference_scheduler", "ddpm")
        self.beta_schedule = policy_cfg.get("beta_schedule", "squaredcos_cap_v2")
        self.prediction_type = policy_cfg.get("prediction_type", "epsilon")
        self.ddim_set_alpha_to_one = policy_cfg.get("ddim_set_alpha_to_one", True)
        self.ddim_steps_offset = policy_cfg.get("ddim_steps_offset", 0)
        self.enforce_action_bounds = policy_cfg.get("enforce_action_bounds", True)
        self.ema_power = policy_cfg.ema_power
        self.use_ema = policy_cfg.use_ema

        self.image_encoders = nn.ModuleDict()
        for name in cfg.data.obs.modality.rgb:
            self.image_encoders[name] = ResNet18Encoder(
                output_dim=policy_cfg.vision_feature_dim,
                pretrained=policy_cfg.pretrained_resnet,
            )

        low_dim = 0
        if cfg.data.use_gripper:
            low_dim += shape_meta["all_shapes"]["gripper_states"][0]
        if cfg.data.use_joint:
            low_dim += shape_meta["all_shapes"]["joint_states"][0]
        if cfg.data.use_ee and "ee_states" in shape_meta["all_shapes"]:
            low_dim += shape_meta["all_shapes"]["ee_states"][0]
        self.low_dim = low_dim
        obs_dim = policy_cfg.vision_feature_dim * len(self.image_encoders) + low_dim

        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=obs_dim * self.obs_horizon,
            diffusion_step_embed_dim=policy_cfg.diffusion_step_embed_dim,
            down_dims=tuple(policy_cfg.down_dims),
            kernel_size=policy_cfg.kernel_size,
            n_groups=policy_cfg.n_groups,
        )
        self.train_noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            beta_schedule=self.beta_schedule,
            clip_sample=policy_cfg.clip_sample,
            prediction_type=self.prediction_type,
        )
        if self.inference_scheduler_name == "ddpm":
            self.inference_noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.num_train_timesteps,
                beta_schedule=self.beta_schedule,
                clip_sample=policy_cfg.clip_sample,
                prediction_type=self.prediction_type,
            )
        elif self.inference_scheduler_name == "ddim":
            self.inference_noise_scheduler = DDIMScheduler(
                num_train_timesteps=self.num_train_timesteps,
                beta_schedule=self.beta_schedule,
                clip_sample=policy_cfg.clip_sample,
                set_alpha_to_one=self.ddim_set_alpha_to_one,
                steps_offset=self.ddim_steps_offset,
                prediction_type=self.prediction_type,
            )
        else:
            raise ValueError(
                "Unsupported diffusion inference scheduler "
                f"{self.inference_scheduler_name!r}; expected 'ddpm' or 'ddim'."
            )

        self.obs_queue = []
        self.action_queue = None
        self.ema_params = None
        self.action_bounds_checked = False

    def _encode_obs(self, data):
        obs_features = []
        for name, encoder in self.image_encoders.items():
            x = data["obs"][name]
            b, t, c, h, w = x.shape
            obs_features.append(encoder(x.reshape(b * t, c, h, w)).view(b, t, -1))

        low_dim_features = []
        if self.cfg.data.use_gripper:
            low_dim_features.append(data["obs"]["gripper_states"])
        if self.cfg.data.use_joint:
            low_dim_features.append(data["obs"]["joint_states"])
        if self.cfg.data.use_ee and "ee_states" in data["obs"]:
            low_dim_features.append(data["obs"]["ee_states"])
        if low_dim_features:
            obs_features.append(torch.cat(low_dim_features, dim=-1).float())
        return torch.cat(obs_features, dim=-1)

    def _obs_cond(self, data):
        obs_features = self._encode_obs(data)
        obs_features = obs_features[:, : self.obs_horizon]
        return obs_features.flatten(start_dim=1)

    def _training_actions(self, data):
        action = data["actions"][:, : self.pred_horizon].float()
        if self.enforce_action_bounds and not self.action_bounds_checked:
            in_range = ((action >= -1.0) & (action <= 1.0)).all().item()
            if not in_range:
                action_min = action.min().item()
                action_max = action.max().item()
                raise ValueError(
                    "DiffusionPolicy expects normalized actions in [-1, 1], "
                    f"but saw range [{action_min:.3f}, {action_max:.3f}]. "
                    "Normalize the HDF5 actions or set "
                    "policy.enforce_action_bounds=false for debugging."
                )
            self.action_bounds_checked = True
        return action.clamp(-1, 1)

    def forward(self, data):
        action = self._training_actions(data)
        timesteps = torch.zeros(
            (action.shape[0],), device=action.device, dtype=torch.long
        )
        return self.noise_pred_net(
            action,
            timesteps,
            global_cond=self._obs_cond(data),
        )

    def compute_loss(self, data, reduction="mean"):
        data = self.preprocess_input(data, train_mode=True)
        action = self._training_actions(data)
        noise = torch.randn_like(action)
        timesteps = torch.randint(
            0,
            self.train_noise_scheduler.config.num_train_timesteps,
            (action.shape[0],),
            device=action.device,
        ).long()
        noisy_action = self.train_noise_scheduler.add_noise(action, noise, timesteps)
        noise_pred = self.noise_pred_net(
            noisy_action,
            timesteps,
            global_cond=self._obs_cond(data),
        )
        return F.mse_loss(noise_pred, noise, reduction=reduction)

    def update_ema(self):
        if not self.use_ema:
            return
        params = [p.detach() for p in self.parameters() if p.requires_grad]
        if self.ema_params is None:
            self.ema_params = [p.clone() for p in params]
            return
        decay = self.ema_power
        for avg, param in zip(self.ema_params, params):
            avg.mul_(decay).add_(param, alpha=1.0 - decay)

    @contextlib.contextmanager
    def ema_scope(self):
        if not self.use_ema or self.ema_params is None:
            yield
            return
        params = [p for p in self.parameters() if p.requires_grad]
        backup = [p.detach().clone() for p in params]
        for param, avg in zip(params, self.ema_params):
            param.data.copy_(avg.data)
        try:
            yield
        finally:
            for param, old in zip(params, backup):
                param.data.copy_(old.data)

    def _stack_obs_queue(self):
        data = {"obs": {}, "task_emb": self.obs_queue[-1]["task_emb"]}
        for key in self.obs_queue[-1]["obs"].keys():
            data["obs"][key] = torch.cat([x["obs"][key] for x in self.obs_queue], dim=1)
        return data

    def _append_obs(self, data):
        data = self.preprocess_input(data, train_mode=False)
        if not self.obs_queue:
            self.obs_queue = [data for _ in range(self.obs_horizon)]
        else:
            self.obs_queue.append(data)
            self.obs_queue = self.obs_queue[-self.obs_horizon :]

    def _sample_action_plan(self, data):
        b = next(iter(data["obs"].values())).shape[0]
        action = torch.randn(
            b, self.pred_horizon, self.action_dim, device=self.cfg.device
        )
        self.inference_noise_scheduler.set_timesteps(self.num_inference_iters)
        for timestep in self.inference_noise_scheduler.timesteps.to(action.device):
            noise_pred = self.noise_pred_net(
                action,
                timestep.expand(b),
                global_cond=self._obs_cond(data),
            )
            action = self.inference_noise_scheduler.step(
                noise_pred, timestep, action
            ).prev_sample
        start = self.obs_horizon - 1
        end = start + self.action_horizon
        return action[:, start:end].clamp(-1, 1)

    def get_action(self, data):
        self.eval()
        with torch.no_grad(), self.ema_scope():
            self._append_obs(data)
            if self.action_queue is None or self.action_queue.shape[1] == 0:
                self.action_queue = self._sample_action_plan(self._stack_obs_queue())
            action = self.action_queue[:, 0]
            self.action_queue = self.action_queue[:, 1:]
        return action.detach().cpu().numpy()

    def reset(self):
        self.obs_queue = []
        self.action_queue = None

    def get_extra_state(self):
        return {"ema_params": self.ema_params}

    def set_extra_state(self, state):
        self.ema_params = state.get("ema_params", None) if state is not None else None

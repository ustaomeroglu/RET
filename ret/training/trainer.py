from __future__ import annotations

import math
from pathlib import Path

import torch
from accelerate import Accelerator
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader

from ret.config import ExperimentConfig, LLMConfig
from ret.extraction.hooks import HiddenStateExtractor
from ret.models.encoders import build_encoder
from ret.models.predictor import Predictor
from ret.models.predictor import TokenProjector
from ret.models.ema import EMATeacher
from ret.losses.prediction import build_prediction_loss, masked_prediction_loss
from ret.losses.vicreg import vicreg_loss
from ret.losses.sigreg import SIGReg, sigreg_loss
from ret.losses.policy import policy_loss as compute_policy_loss
from ret.losses.supervision import supervision_reward_loss as compute_reward_loss
from ret.losses.supervision import supervision_value_loss as compute_value_loss
from ret.models.policy import PolicyNetwork
from ret.models.supervision import SupervisionRewardHead, SupervisionValueHead
from ret.utils.checkpointing import save_checkpoint, load_checkpoint
from ret.utils.logging import log_metrics, LocalLogger
from ret.utils.prompt_logger import PromptLogger


class JEPATrainer:
    """Main JEPA/BYOL-style training loop."""

    def __init__(self, config: ExperimentConfig, accelerator: Accelerator,
                 prompt_logger: PromptLogger | None = None):
        self.config = config
        self.accelerator = accelerator
        self.device = accelerator.device
        self.prompt_logger = prompt_logger

        # Resolve warmup_steps=-1 → 10% of max_steps (use local vars, don't mutate config)
        max_steps = config.training.max_steps
        training_warmup = config.training.warmup_steps
        if training_warmup == -1:
            training_warmup = max(1, max_steps // 10)
        self._resolved_training_warmup = training_warmup

        ema_warmup = config.ema.warmup_steps
        if ema_warmup == -1:
            ema_warmup = max(1, max_steps // 10)

        # Frozen LLM (not prepared — no gradients) — skipped in cached mode
        self._cached_mode = config.data.mode == "cached"
        if self._cached_mode:
            self.extractor = None
            self._embedding_layer = (
                self._load_embedding_only(config.llm)
                if config.predictor.mode == "controlled" else None
            )
            from transformers import AutoTokenizer
            tok_kwargs = {}
            if config.llm.trust_remote_code:
                tok_kwargs["trust_remote_code"] = True
            self._tokenizer = AutoTokenizer.from_pretrained(
                config.llm.name, **tok_kwargs
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        else:
            self.extractor = HiddenStateExtractor(config.llm, device=self.device)
            self._embedding_layer = None
            self._tokenizer = None

        # Student encoder
        d_h = config.llm.hidden_dim
        # Multi-layer: compute concat_dim from extractor (which resolved "all")
        if self.extractor is not None and self.extractor.num_extract_layers > 1:
            concat_dim = self.extractor.num_extract_layers * d_h
        else:
            concat_dim = None
        self.encoder = build_encoder(d_h, config.encoder, concat_dim=concat_dim).to(self.device)

        # EMA teacher — only if enabled; created BEFORE prepare() so it copies raw weights
        self.ema_teacher: EMATeacher | None = None
        if config.ema.enabled:
            self.ema_teacher = EMATeacher(
                self.encoder,
                momentum=config.ema.momentum,
                warmup_momentum=config.ema.warmup_momentum,
                warmup_steps=ema_warmup,
            ).to(self.device)

        # Predictor
        self.predictor = Predictor(config.encoder.z_dim, config.predictor).to(self.device)

        # Token projector (for controlled mode)
        self.token_projector: TokenProjector | None = None
        if config.predictor.mode == "controlled":
            self.token_projector = TokenProjector(d_h, config.predictor).to(self.device)

        # Policy network (optional: z_t -> next-token logits)
        self.policy_network: PolicyNetwork | None = None
        if config.policy.enabled:
            self.policy_network = PolicyNetwork(
                z_dim=config.encoder.z_dim,
                vocab_size=config.llm.vocab_size,
                hidden_dim=config.policy.hidden_dim,
                num_layers=config.policy.num_layers,
            ).to(self.device)

        # Supervision heads (optional, TD-MPC2-inspired)
        sv = config.supervision
        if sv.value_enabled and not sv.reward_enabled:
            raise ValueError("supervision.value_enabled requires reward_enabled "
                             "(TD target needs rewards)")

        self.supervision_reward: SupervisionRewardHead | None = None
        self.supervision_value: SupervisionValueHead | None = None
        self.supervision_value_ema: SupervisionValueHead | None = None
        if sv.reward_enabled:
            self.supervision_reward = SupervisionRewardHead(
                z_dim=config.encoder.z_dim,
                hidden_dim=sv.reward_hidden_dim,
                num_layers=sv.reward_num_layers,
            ).to(self.device)
        if sv.value_enabled:
            import copy
            self.supervision_value = SupervisionValueHead(
                z_dim=config.encoder.z_dim,
                hidden_dim=sv.value_hidden_dim,
                num_layers=sv.value_num_layers,
            ).to(self.device)
            self.supervision_value_ema = copy.deepcopy(self.supervision_value)
            for p in self.supervision_value_ema.parameters():
                p.requires_grad_(False)

        # Loss
        self.pred_loss_fn = build_prediction_loss(config.loss.prediction_type)

        # SIGReg module (no learnable params, only buffers)
        self.sigreg: SIGReg | None = None
        if config.loss.regularizer == "sigreg":
            self.sigreg = SIGReg(
                knots=config.loss.sigreg_knots,
                num_proj=config.loss.sigreg_num_proj,
            ).to(self.device)

        # Optimizer: all trainable params
        params = list(self.encoder.parameters()) + list(self.predictor.parameters())
        if self.token_projector is not None:
            params += list(self.token_projector.parameters())
        if self.policy_network is not None:
            params += list(self.policy_network.parameters())
        if self.supervision_reward is not None:
            params += list(self.supervision_reward.parameters())
        if self.supervision_value is not None:
            params += list(self.supervision_value.parameters())
        self.optimizer = torch.optim.AdamW(
            params, lr=config.training.lr, weight_decay=config.training.weight_decay
        )

        # Prepare trainable models and optimizer with accelerate
        to_prepare = [self.encoder, self.predictor]
        if self.token_projector is not None:
            to_prepare.append(self.token_projector)
        if self.policy_network is not None:
            to_prepare.append(self.policy_network)
        if self.supervision_reward is not None:
            to_prepare.append(self.supervision_reward)
        if self.supervision_value is not None:
            to_prepare.append(self.supervision_value)
        to_prepare.append(self.optimizer)
        prepared = accelerator.prepare(*to_prepare)
        self.optimizer = prepared[-1]
        idx = 0
        self.encoder = prepared[idx]; idx += 1
        self.predictor = prepared[idx]; idx += 1
        if self.token_projector is not None:
            self.token_projector = prepared[idx]; idx += 1
        if self.policy_network is not None:
            self.policy_network = prepared[idx]; idx += 1
        if self.supervision_reward is not None:
            self.supervision_reward = prepared[idx]; idx += 1
        if self.supervision_value is not None:
            self.supervision_value = prepared[idx]; idx += 1

        # LR schedule: built AFTER prepare so it references the wrapped optimizer
        self.lr_scheduler = self._build_lr_scheduler()

        self.global_step = 0

    def _resume_extra(self) -> dict:
        """Build the extra state dict for checkpoint resume support."""
        extra = {
            "ema_teacher": self.ema_teacher.state_dict() if self.ema_teacher else None,
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "global_step": self.global_step,
        }
        if self.supervision_value_ema is not None:
            extra["value_ema"] = self.supervision_value_ema.state_dict()
        return extra

    def resume_from_checkpoint(self, path: str | Path) -> None:
        """Restore training state from a checkpoint saved by this trainer.

        Loads encoder, predictor, optimizer, EMA teacher, LR scheduler,
        global_step, and optionally token_projector.
        """
        path = Path(path)
        print(f"Resuming from checkpoint: {path}")
        ckpt = load_checkpoint(path)

        # Core model weights
        self.accelerator.unwrap_model(self.encoder).load_state_dict(ckpt["encoder"])
        self.accelerator.unwrap_model(self.predictor).load_state_dict(ckpt["predictor"])
        self.optimizer.load_state_dict(ckpt["optimizer"])

        if self.token_projector is not None and "token_projector" in ckpt:
            self.accelerator.unwrap_model(self.token_projector).load_state_dict(
                ckpt["token_projector"]
            )

        if self.policy_network is not None and "policy_network" in ckpt:
            self.accelerator.unwrap_model(self.policy_network).load_state_dict(
                ckpt["policy_network"]
            )

        # Backward compat: try new key first, fall back to old key
        sv_reward_key = "supervision_reward" if "supervision_reward" in ckpt else "reward_network"
        if self.supervision_reward is not None and sv_reward_key in ckpt:
            self.accelerator.unwrap_model(self.supervision_reward).load_state_dict(
                ckpt[sv_reward_key]
            )
        sv_value_key = "supervision_value" if "supervision_value" in ckpt else "value_network"
        if self.supervision_value is not None and sv_value_key in ckpt:
            self.accelerator.unwrap_model(self.supervision_value).load_state_dict(
                ckpt[sv_value_key]
            )
        if self.supervision_value_ema is not None and "value_ema" in ckpt:
            self.supervision_value_ema.load_state_dict(ckpt["value_ema"])

        # Resume-critical state (may be absent in legacy checkpoints)
        if self.ema_teacher is not None:
            if "ema_teacher" in ckpt and ckpt["ema_teacher"] is not None:
                self.ema_teacher.load_state_dict(ckpt["ema_teacher"])
            else:
                print("  WARNING: checkpoint has no EMA state, re-copying from encoder")
                import copy
                self.ema_teacher.teacher.load_state_dict(
                    self.accelerator.unwrap_model(self.encoder).state_dict()
                )

        if "lr_scheduler" in ckpt:
            self.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        else:
            # Fast-forward scheduler to match global_step
            step = ckpt.get("global_step", ckpt.get("step", 0))
            for _ in range(step):
                self.lr_scheduler.step()
            print(f"  WARNING: no lr_scheduler state, fast-forwarded to step {step}")

        self.global_step = ckpt.get("global_step", ckpt.get("step", 0))
        ema_info = f"ema_momentum={self.ema_teacher.momentum:.4f}" if self.ema_teacher else "ema=disabled"
        print(f"  Resumed at global_step={self.global_step}, "
              f"lr={self.lr_scheduler.get_last_lr()[0]:.2e}, "
              f"{ema_info}")

    def _build_lr_scheduler(self):
        """Linear warmup then cosine decay to lr * lr_min_fraction (no bounce)."""
        warmup = self._resolved_training_warmup
        max_steps = self.config.training.max_steps
        min_frac = self.config.training.lr_min_fraction

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(warmup, 1)
            progress = (step - warmup) / max(max_steps - warmup, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_frac + (1.0 - min_frac) * cosine

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _load_embedding_only(self, llm_config: LLMConfig) -> nn.Module:
        """Load only the LLM's embedding table for controlled-mode token embeddings.

        Temporarily loads the full model to extract the embedding layer, then
        deletes the model to free memory.
        """
        import gc
        from transformers import AutoModelForCausalLM

        dtype = getattr(torch, llm_config.dtype)
        load_kwargs: dict = {"torch_dtype": dtype}
        if llm_config.trust_remote_code:
            load_kwargs["trust_remote_code"] = True

        model = AutoModelForCausalLM.from_pretrained(llm_config.name, **load_kwargs)
        embed = model.get_input_embeddings()
        embed = embed.to(self.device)
        embed.requires_grad_(False)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return embed

    def _compute_ar_loss(
        self,
        z_t: torch.Tensor,
        z_tgt: torch.Tensor,
        u_all: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Multi-step autoregressive prediction loss.

        Rolls out the predictor for H steps (feeding predictions back as input),
        comparing each step against the EMA teacher target with temporal discounting.

        Args:
            z_t: (B, T-1, d_z) — student encoder z at positions 0..T-2
            z_tgt: (B, T, d_z) — full EMA teacher targets (all T positions)
            u_all: (B, T, d_u) or None — full control signals (before slicing)
            attention_mask: (B, T) or None — 1 for valid, 0 for padding

        Returns:
            Scalar loss tensor
        """
        T = z_tgt.shape[1]
        H = min(self.config.loss.ar_horizon, T - 1)
        rho = self.config.loss.ar_rho
        loss_type = self.config.loss.prediction_type

        ar_loss = torch.tensor(0.0, device=z_t.device)
        z_rolled = z_t  # start from encoder output

        for k in range(1, H + 1):
            valid_len = T - k
            if valid_len <= 0:
                break

            z_src = z_rolled[:, :valid_len]
            u_k = u_all[:, k:k + valid_len] if u_all is not None else None
            z_pred = self.predictor(z_src, u_k)
            z_target = z_tgt[:, k:k + valid_len].detach()

            if attention_mask is not None:
                mask_k = (attention_mask[:, :valid_len] * attention_mask[:, k:k + valid_len]).float()
                step_loss = masked_prediction_loss(z_pred, z_target, mask_k, loss_type)
            else:
                step_loss = self.pred_loss_fn(z_pred, z_target)

            ar_loss = ar_loss + (rho ** (k - 1)) * step_loss
            z_rolled = z_pred  # feed back predictions (gradients flow through)

        return ar_loss / H

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Execute one training step (may be a gradient accumulation micro-step)."""
        _accum_models = [self.encoder, self.predictor]
        if self.policy_network is not None:
            _accum_models.append(self.policy_network)
        if self.supervision_reward is not None:
            _accum_models.append(self.supervision_reward)
        if self.supervision_value is not None:
            _accum_models.append(self.supervision_value)
        with self.accelerator.accumulate(*_accum_models):
            input_ids = batch["input_ids"]  # (B, T)
            attention_mask = batch.get("attention_mask")  # (B, T) or None

            # 1. Extract hidden states from frozen LLM (or read from batch if cached)
            if "hidden_states" in batch:
                h = batch["hidden_states"].to(self.device)  # (B, T, d_h) float32
            else:
                h = self.extractor.extract(input_ids, attention_mask=attention_mask)  # (B, T, d_h)

            # 2. Student encoder
            z, enc_aux = self.encoder(h, attention_mask=attention_mask)  # (B, T, d_z), dict

            # 3. Prediction target
            if self.ema_teacher is not None:
                with torch.no_grad():
                    z_tgt = self.ema_teacher(h, attention_mask=attention_mask)  # (B, T, d_z)
            else:
                # No EMA: target is student encoder's own output.
                # Gradients flow through z_tgt — regularizer prevents collapse.
                z_tgt = z

            # 4. Temporal split
            z_t = z[:, :-1]  # (B, T-1, d_z)

            # 5. Control signal (if controlled)
            u_all = None
            if self.token_projector is not None:
                with torch.no_grad():
                    if self._embedding_layer is not None:
                        token_emb = self._embedding_layer(
                            input_ids.to(self.device)
                        ).float()
                    else:
                        token_emb = self.extractor.get_token_embeddings(input_ids)  # (B, T, d_h)
                u_all = self.token_projector(token_emb)  # (B, T, d_u)

            # 6. Multi-step AR prediction loss
            pred_loss = self._compute_ar_loss(z_t, z_tgt, u_all, attention_mask)

            # 7. Regularization loss (VICReg / SIGReg / none)
            z_for_reg = enc_aux.get("z_pre", z)
            vreg_loss, vreg_metrics = self._compute_reg_loss(z_for_reg, attention_mask)
            # 8. Policy loss (z_t -> next-token prediction)
            policy_metrics = {}
            policy_loss_val = torch.tensor(0.0, device=self.device)
            if (self.policy_network is not None
                    and self.global_step >= self.config.policy.warmup_steps):
                next_tokens = input_ids[:, 1:].to(self.device)
                policy_mask = None
                if attention_mask is not None:
                    policy_mask = (
                        attention_mask[:, :-1] * attention_mask[:, 1:]
                    ).float()
                z_policy = z_t.detach() if self.config.policy.detach_z else z_t
                logits = self.policy_network(z_policy)
                policy_loss_val, policy_metrics = compute_policy_loss(
                    logits, next_tokens, policy_mask,
                )
                policy_loss_val = self.config.policy.lambda_policy * policy_loss_val

            # 8b. Reward + value loss (TD-MPC2-inspired)
            rv_metrics = {}
            sv_loss = torch.tensor(0.0, device=self.device)
            if (self.supervision_reward is not None
                    and "rewards" in batch
                    and self.global_step >= self.config.supervision.warmup_steps):
                rewards = batch["rewards"].to(self.device)  # (B, T)
                r_mask = batch.get("reward_mask")
                if r_mask is not None:
                    r_mask = r_mask.to(self.device)
                else:
                    r_mask = attention_mask  # fallback: all valid tokens
                sv_cfg = self.config.supervision

                # Reward prediction — masked to supervised tokens only
                r_pred = self.supervision_reward(z).squeeze(-1)  # (B, T)
                r_loss, r_metrics = compute_reward_loss(r_pred, rewards, r_mask)
                sv_loss = sv_cfg.lambda_reward * r_loss
                rv_metrics = {**r_metrics}

                # Value prediction + TD target (only if value head is present)
                if self.supervision_value is not None:
                    import torch.nn.functional as _F
                    v_pred = self.supervision_value(z).squeeze(-1)  # (B, T)
                    with torch.no_grad():
                        v_next_padded = torch.zeros_like(rewards)
                        v_next = self.supervision_value_ema(z[:, 1:].detach()).squeeze(-1)  # (B, T-1)
                        v_next_padded[:, :-1] = v_next

                        if attention_mask is not None:
                            is_nonterminal = _F.pad(
                                attention_mask[:, :-1] * attention_mask[:, 1:],
                                (0, 1), value=0,
                            ).float()
                        else:
                            is_nonterminal = torch.ones_like(rewards)
                            is_nonterminal[:, -1] = 0

                        v_target = rewards + is_nonterminal * v_next_padded

                    v_loss, v_metrics = compute_value_loss(v_pred, v_target, attention_mask)
                    sv_loss = sv_loss + sv_cfg.lambda_value * v_loss
                    rv_metrics.update(v_metrics)

            # 8c. Temporal smoothness loss (TV regularization on z)
            smooth_loss = torch.tensor(0.0, device=self.device)
            smooth_metrics = {}
            if self.config.loss.lambda_smooth > 0:
                from ret.losses.temporal import temporal_smoothness_loss
                sdims = None if self.config.loss.smooth_dims < 0 else self.config.loss.smooth_dims
                smooth_loss, smooth_metrics = temporal_smoothness_loss(
                    z, attention_mask,
                    loss_type=self.config.loss.smooth_loss_type,
                    smooth_dims=sdims,
                )
                smooth_loss = self.config.loss.lambda_smooth * smooth_loss

            total_loss = (pred_loss + vreg_loss + smooth_loss
                          + enc_aux.get("aux_loss", 0)
                          + policy_loss_val + sv_loss)

            # 9. Backward + optimize (accumulate handles zero_grad/step skipping)
            self.accelerator.backward(total_loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(
                    self._trainable_params(), self.config.training.grad_clip
                )
                grad_norm = self._compute_grad_norm()
            else:
                grad_norm = 0.0
            self.optimizer.step()
            self.optimizer.zero_grad()

        # 9. EMA update + LR step only on sync steps (after actual optimizer step)
        # LambdaLR is NOT wrapped by accelerate, so it must be guarded manually
        # to avoid stepping once per micro-step instead of per optimizer step.
        if self.accelerator.sync_gradients:
            self.lr_scheduler.step()
            if self.ema_teacher is not None:
                self.ema_teacher.update(self.accelerator.unwrap_model(self.encoder))
            # Value EMA (Polyak averaging for stable TD targets)
            if self.supervision_value_ema is not None:
                tau = self.config.supervision.value_ema_momentum
                for p_ema, p in zip(
                    self.supervision_value_ema.parameters(),
                    self.accelerator.unwrap_model(self.supervision_value).parameters(),
                ):
                    p_ema.data.mul_(tau).add_(p.data, alpha=1.0 - tau)
            self.global_step += 1

        metrics = {
            "train/total_loss": total_loss.item(),
            "train/pred_loss": pred_loss.item(),
            "train/grad_norm": grad_norm,
            "train/lr": self.lr_scheduler.get_last_lr()[0],
            "train/ema_momentum": self.ema_teacher.momentum if self.ema_teacher else 0.0,
            **vreg_metrics,
            **smooth_metrics,
            **policy_metrics,
            **rv_metrics,
        }
        # Log scalar encoder aux metrics (e.g. sparse_loss, vq_loss, codebook_perplexity)
        for k, v in enc_aux.items():
            if k in ("aux_loss", "z_pre", "codebook_indices"):
                continue
            if isinstance(v, torch.Tensor) and v.numel() != 1:
                continue
            metrics[f"encoder/{k}"] = v.item() if isinstance(v, torch.Tensor) else v
        return metrics

    def _compute_grad_norm(self) -> float:
        """Compute total grad norm across all trainable params."""
        total_norm_sq = 0.0
        for p in self._trainable_params():
            if p.grad is not None:
                total_norm_sq += p.grad.data.norm(2).item() ** 2
        return total_norm_sq ** 0.5

    def _compute_reg_loss(
        self, z: torch.Tensor, attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute regularization loss (VICReg, SIGReg, or none)."""
        reg = self.config.loss.regularizer
        if reg == "none":
            return torch.tensor(0.0, device=self.device), {"reg/mean_std": 0.0}

        # Handle masking: extract valid tokens
        if attention_mask is not None:
            z_valid = z[attention_mask.bool()]
            if z_valid.shape[0] <= 1:
                return torch.tensor(0.0, device=self.device), {"reg/mean_std": 0.0}
            z_input = z_valid.unsqueeze(0)
        else:
            z_input = z

        if reg == "sigreg":
            return sigreg_loss(
                z_input, self.sigreg,
                lambda_sigreg=self.config.loss.lambda_sigreg,
            )
        else:  # vicreg
            return vicreg_loss(
                z_input,
                lambda_var=self.config.loss.lambda_var,
                lambda_cov=self.config.loss.lambda_cov,
                var_target=self.config.loss.var_target,
            )

    @torch.no_grad()
    def eval_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Evaluate on a single batch."""
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        if "hidden_states" in batch:
            h = batch["hidden_states"].to(self.device)
        else:
            h = self.extractor.extract(input_ids, attention_mask=attention_mask)

        self.encoder.eval()
        if self.token_projector is not None:
            self.token_projector.eval()
        if self.policy_network is not None:
            self.policy_network.eval()
        if self.supervision_reward is not None:
            self.supervision_reward.eval()
        if self.supervision_value is not None:
            self.supervision_value.eval()
        self.predictor.eval()
        try:
            z, enc_aux = self.encoder(h, attention_mask=attention_mask)
            if self.ema_teacher is not None:
                z_tgt = self.ema_teacher(h, attention_mask=attention_mask)
            else:
                z_tgt = z

            z_t = z[:, :-1]

            u_all = None
            if self.token_projector is not None:
                if self._embedding_layer is not None:
                    token_emb = self._embedding_layer(
                        input_ids.to(self.device)
                    ).float()
                else:
                    token_emb = self.extractor.get_token_embeddings(input_ids)
                u_all = self.token_projector(token_emb)

            pred_loss = self._compute_ar_loss(z_t, z_tgt, u_all, attention_mask)

            z_for_reg = enc_aux.get("z_pre", z)
            _, vreg_metrics = self._compute_reg_loss(z_for_reg, attention_mask)

            # Policy eval
            eval_policy_metrics = {}
            if self.policy_network is not None:
                next_tokens = input_ids[:, 1:].to(self.device)
                policy_mask = None
                if attention_mask is not None:
                    policy_mask = (
                        attention_mask[:, :-1] * attention_mask[:, 1:]
                    ).float()
                logits = self.policy_network(z_t)
                _, eval_policy_metrics = compute_policy_loss(
                    logits, next_tokens, policy_mask,
                )
                eval_policy_metrics = {
                    f"val/{k.split('/')[-1]}": v
                    for k, v in eval_policy_metrics.items()
                }

            # Reward/value eval
            eval_rv_metrics = {}
            if (self.supervision_reward is not None
                    and "rewards" in batch):
                rewards = batch["rewards"].to(self.device)
                r_mask = batch.get("reward_mask")
                if r_mask is not None:
                    r_mask = r_mask.to(self.device)
                else:
                    r_mask = attention_mask

                r_pred = self.supervision_reward(z).squeeze(-1)
                _, r_metrics = compute_reward_loss(r_pred, rewards, r_mask)
                all_rv = {**r_metrics}

                if self.supervision_value is not None:
                    import torch.nn.functional as _F
                    v_pred = self.supervision_value(z).squeeze(-1)
                    v_next_padded = torch.zeros_like(rewards)
                    v_next = self.supervision_value_ema(z[:, 1:]).squeeze(-1)
                    v_next_padded[:, :-1] = v_next
                    if attention_mask is not None:
                        is_nonterminal = _F.pad(
                            attention_mask[:, :-1] * attention_mask[:, 1:],
                            (0, 1), value=0,
                        ).float()
                    else:
                        is_nonterminal = torch.ones_like(rewards)
                        is_nonterminal[:, -1] = 0
                    v_target = rewards + is_nonterminal * v_next_padded
                    _, v_metrics = compute_value_loss(v_pred, v_target, attention_mask)
                    all_rv.update(v_metrics)

                eval_rv_metrics = {
                    f"val/{k.split('/')[-1]}": v
                    for k, v in all_rv.items()
                }
        finally:
            self.encoder.train()
            if self.token_projector is not None:
                self.token_projector.train()
            if self.policy_network is not None:
                self.policy_network.train()
            if self.supervision_reward is not None:
                self.supervision_reward.train()
            if self.supervision_value is not None:
                self.supervision_value.train()
            self.predictor.train()

        return {
            "val/pred_loss": pred_loss.item(),
            **{f"val/{k}": v for k, v in vreg_metrics.items()},
            **eval_policy_metrics,
            **eval_rv_metrics,
        }

    def evaluate(self, val_loader: DataLoader, max_batches: int = 20) -> dict[str, float]:
        """Run evaluation over multiple batches."""
        all_metrics: dict[str, list[float]] = {}
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            metrics = self.eval_step(batch)
            for k, v in metrics.items():
                all_metrics.setdefault(k, []).append(v)

        return {k: sum(v) / len(v) for k, v in all_metrics.items()}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        use_wandb: bool = True,
    ) -> None:
        """Main training loop."""
        cfg = self.config.training
        log_cfg = self.config.logging
        output_dir = Path(log_cfg.output_dir)
        is_main = self.accelerator.is_main_process

        # Prepare dataloaders (adds DistributedSampler for multi-GPU)
        train_loader = self.accelerator.prepare(train_loader)
        if val_loader is not None:
            val_loader = self.accelerator.prepare(val_loader)

        local_logger = LocalLogger(output_dir) if is_main else None
        train_iter = iter(train_loader)

        accum_steps = cfg.gradient_accumulation_steps
        total_micro_steps = cfg.max_steps * accum_steps
        run_name = log_cfg.wandb_run_name or output_dir.name
        pbar = tqdm(
            total=cfg.max_steps - self.global_step,
            desc=run_name,
            dynamic_ncols=True,
            disable=not is_main,
        )
        micro_step = 0
        while self.global_step < cfg.max_steps:
            # Get batch (restart iterator if exhausted)
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            # Log first N prompts (main process only)
            if (is_main and self.prompt_logger
                    and not self.prompt_logger.is_full):
                tokenizer = (
                    self._tokenizer if self._cached_mode
                    else self.extractor.tokenizer
                )
                self.prompt_logger.log_forward_pass_batch(
                    batch["input_ids"],
                    tokenizer,
                    attention_mask=batch.get("attention_mask"),
                    source="document" if "attention_mask" in batch else "concat_chunk",
                )

            prev_step = self.global_step
            metrics = self.train_step(batch)
            micro_step += 1

            # Only log/eval/checkpoint on actual optimizer steps
            if not self.accelerator.sync_gradients:
                continue

            step = self.global_step
            pbar.update(step - prev_step)

            if is_main:
                # Update progress bar postfix
                pbar.set_postfix(
                    loss=f"{metrics['train/total_loss']:.4f}",
                    pred=f"{metrics['train/pred_loss']:.4f}",
                    std=f"{metrics.get('reg/mean_std', 0):.3f}",
                    lr=f"{metrics['train/lr']:.1e}",
                )

                # Log
                if step % log_cfg.log_every == 0:
                    local_logger.log_train(metrics, step)
                    if use_wandb:
                        log_metrics(metrics, step=step)

            # Eval
            if val_loader is not None and step % cfg.eval_every == 0 and step > 0:
                val_metrics = self.evaluate(val_loader)
                if is_main:
                    pbar.write(f"  [val step {step}] pred_loss={val_metrics['val/pred_loss']:.4f}")
                    local_logger.log_val(val_metrics, step)
                    if use_wandb:
                        log_metrics(val_metrics, step=step)

            # Checkpoint (main process only, unwrap models for clean state dicts)
            if step % cfg.checkpoint_every == 0 and step > 0:
                if is_main:
                    ckpt_path = output_dir / f"checkpoint_{step}.pt"
                    save_checkpoint(
                        ckpt_path,
                        step=step,
                        encoder=self.accelerator.unwrap_model(self.encoder),
                        predictor=self.accelerator.unwrap_model(self.predictor),
                        optimizer=self.optimizer,
                        config=self.config,
                        token_projector=(
                            self.accelerator.unwrap_model(self.token_projector)
                            if self.token_projector is not None else None
                        ),
                        policy_network=(
                            self.accelerator.unwrap_model(self.policy_network)
                            if self.policy_network is not None else None
                        ),
                        supervision_reward=(
                            self.accelerator.unwrap_model(self.supervision_reward)
                            if self.supervision_reward is not None else None
                        ),
                        supervision_value=(
                            self.accelerator.unwrap_model(self.supervision_value)
                            if self.supervision_value is not None else None
                        ),
                        extra=self._resume_extra(),
                    )
                    pbar.write(f"  Saved checkpoint: {ckpt_path}")
                self.accelerator.wait_for_everyone()

        pbar.close()

        # Final checkpoint
        if is_main:
            save_checkpoint(
                output_dir / "checkpoint_final.pt",
                step=self.global_step,
                encoder=self.accelerator.unwrap_model(self.encoder),
                predictor=self.accelerator.unwrap_model(self.predictor),
                optimizer=self.optimizer,
                config=self.config,
                token_projector=(
                    self.accelerator.unwrap_model(self.token_projector)
                    if self.token_projector is not None else None
                ),
                policy_network=(
                    self.accelerator.unwrap_model(self.policy_network)
                    if self.policy_network is not None else None
                ),
                supervision_reward=(
                    self.accelerator.unwrap_model(self.supervision_reward)
                    if self.supervision_reward is not None else None
                ),
                supervision_value=(
                    self.accelerator.unwrap_model(self.supervision_value)
                    if self.supervision_value is not None else None
                ),
                extra=self._resume_extra(),
            )
        self.accelerator.wait_for_everyone()

        # Save local logs and generate plots
        if is_main:
            local_logger.save()
            if self.prompt_logger:
                self.prompt_logger.save()
            print(f"Saved metrics CSV + loss curves to {output_dir}")

    def _trainable_params(self):
        params = (
            list(self.accelerator.unwrap_model(self.encoder).parameters())
            + list(self.accelerator.unwrap_model(self.predictor).parameters())
        )
        if self.token_projector is not None:
            params += list(self.accelerator.unwrap_model(self.token_projector).parameters())
        if self.policy_network is not None:
            params += list(self.accelerator.unwrap_model(self.policy_network).parameters())
        if self.supervision_reward is not None:
            params += list(self.accelerator.unwrap_model(self.supervision_reward).parameters())
        if self.supervision_value is not None:
            params += list(self.accelerator.unwrap_model(self.supervision_value).parameters())
        return params

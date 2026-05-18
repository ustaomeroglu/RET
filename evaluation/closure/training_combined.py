"""Unified closure predictor training loop for model and SAE modes."""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from evaluation.closure.predictors import (
    build_closure_predictor,
    build_token_projector,
    compute_loss,
    compute_r2_streaming,
)
from evaluation.closure.config import make_train_iter


def train_all_variants(
    train_loader,
    extractor,
    encode_fn: Callable,
    val_data: dict,
    d_repr: int,
    d_h: int,
    config,
    loss_type: str,
    device: str,
    repr_label: str = "model",
    cfg_section=None,
) -> tuple[dict, nn.Module | None, float]:
    """Train autonomous and controlled closure variants using shared extraction.

    Parameters
    ----------
    train_loader:
        DataLoader for training data.
    extractor:
        HiddenStateExtractor providing h and token embeddings.
    encode_fn:
        Callable with signature ``(h, attention_mask) -> (repr_t, repr_next)``
        where both outputs are flat tensors of shape ``(N_pairs, d_repr)``.
        The function receives the full-sequence ``h`` (B, T, d_h) and
        the attention_mask, and is responsible for computing temporal pairs.
    val_data:
        Dict from ``collect_val_data``.  Must contain at least
        ``repr_t``, ``repr_next``, ``h_t``, ``token_emb_next``.
    d_repr:
        Dimensionality of the representation (z_dim or sae_dim).
    d_h:
        Hidden-state dimensionality.
    config:
        RETConfig.
    loss_type:
        ``"mse"`` or ``"cosine"``.
    device:
        Target device string.
    repr_label:
        Short label used in progress messages (default ``"model"``).
    cfg_section:
        Config sub-section that holds training hyper-params
        (``hidden_dim``, ``num_layers``, ``batch_size``,
        ``lr``, ``max_steps``, ``eval_every``, ``patience``).
        Defaults to ``config.closure``.

    Returns
    -------
    trained : dict
        Mapping from variant name to training artefact dict.
    projector : nn.Module | None
        Trained token projector for controlled variant.
    target_variance : float
        Variance of ``repr_next`` on the validation set.
    """
    from tqdm import tqdm
    from torch.utils.data import DataLoader, TensorDataset

    if cfg_section is None:
        cfg_section = config.closure
    ccfg = cfg_section

    control_dim = config.closure.control_dim
    closure_batch_size = ccfg.batch_size

    # Val representation tensors (encoded by caller or pre-encoded in val_data)
    val_repr_t = val_data["repr_t"]
    val_repr_next = val_data["repr_next"]
    val_h_t = val_data["h_t"]
    val_te_next = val_data["token_emb_next"]

    target_variance = val_repr_next.to(device).var(dim=0).mean().item()
    print(f"  Target variance ({repr_label}_next): {target_variance:.6f}")

    hidden_dim = ccfg.hidden_dim

    # Build predictors + optimizers for autonomous and controlled variants
    predictors = {
        "autonomous": build_closure_predictor(d_repr, d_repr, hidden_dim, ccfg.num_layers).to(device),
        "controlled": build_closure_predictor(d_repr + control_dim, d_repr, hidden_dim, ccfg.num_layers).to(device),
    }
    projector = build_token_projector(d_h, control_dim).to(device)

    optimizers = {
        "autonomous": torch.optim.Adam(predictors["autonomous"].parameters(), lr=ccfg.lr),
        "controlled": torch.optim.Adam(
            list(predictors["controlled"].parameters()) + list(projector.parameters()), lr=ccfg.lr,
        ),
    }

    # Val inputs per variant
    val_inputs = {
        "autonomous": (val_repr_t, val_repr_next),
    }
    # Ctrl val inputs stored as (repr, h, te, y) for projector application
    val_ctrl_inputs = {
        "controlled": (val_repr_t, val_h_t, val_te_next, val_repr_next),
    }

    # Training state
    best_states = {name: None for name in predictors}
    best_proj_state = None
    best_val_losses = {name: float("inf") for name in predictors}
    evals_without_improvement = {name: 0 for name in predictors}
    active_variants = set(predictors.keys())
    train_loss_accum = {name: [0.0, 0] for name in predictors}  # [sum, count]
    all_train_losses = {name: [] for name in predictors}
    all_val_losses = {name: [] for name in predictors}
    all_val_r2 = {name: [] for name in predictors}

    train_it = make_train_iter(train_loader)

    max_steps = ccfg.max_steps
    eval_every = ccfg.eval_every
    patience = ccfg.patience
    min_delta = 1e-6

    pbar = tqdm(range(1, max_steps + 1), desc=f"{repr_label} closure training")
    for step in pbar:
        if not active_variants:
            break

        # Shared extraction: one LLM forward pass per batch
        batch = next(train_it)
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        with torch.no_grad():
            if "hidden_states" in batch:
                h = batch["hidden_states"].to(device)
            else:
                h = extractor.extract(input_ids, attention_mask=attention_mask)
            # Move mask to same device as h (needed for multi-GPU / device_map)
            if attention_mask is not None:
                attention_mask = attention_mask.to(h.device)
            token_emb = extractor.get_token_embeddings(input_ids)  # (B, T, d_h)

            # Compute representation pairs via encode_fn
            repr_t, repr_next = encode_fn(h, attention_mask)

        # Compute h_t and te_next temporal pairs (stay on device)
        if attention_mask is not None:
            pair_mask = (attention_mask[:, :-1] * attention_mask[:, 1:]).bool()
            h_t = h[:, :-1][pair_mask]
            te_next = token_emb[:, 1:][pair_mask]
        else:
            B, T, _ = h.shape
            h_t = h[:, :-1].reshape(-1, d_h)
            te_next = token_emb[:, 1:].reshape(-1, d_h)

        # Sub-batch and train each active variant
        n_pairs = repr_t.shape[0]
        for start in range(0, n_pairs, closure_batch_size):
            end = min(start + closure_batch_size, n_pairs)
            rb = repr_t[start:end]
            rnb = repr_next[start:end]
            teb = te_next[start:end]

            for name in list(active_variants):
                model = predictors[name]
                model.train()

                if name == "autonomous":
                    x, y = rb, rnb
                elif name == "controlled":
                    projector.train()
                    u = projector(teb)
                    x, y = torch.cat([rb, u], dim=-1), rnb

                pred = model(x)
                loss = compute_loss(pred, y, loss_type)
                optimizers[name].zero_grad()
                loss.backward()
                optimizers[name].step()

                train_loss_accum[name][0] += loss.item()
                train_loss_accum[name][1] += 1

        # Periodic validation
        if step % eval_every == 0:
            for name in list(active_variants):
                s, c = train_loss_accum[name]
                avg_train = s / max(c, 1)
                all_train_losses[name].append(avg_train)
                train_loss_accum[name] = [0.0, 0]

                predictors[name].eval()

                with torch.no_grad():
                    if name == "autonomous":
                        vx, vy = val_inputs[name]
                        vds = TensorDataset(vx, vy)
                        vldr = DataLoader(vds, batch_size=closure_batch_size, shuffle=False)
                        val_loss_sum, val_n = 0.0, 0
                        ss_res_sum = 0.0
                        tgt_sum = tgt_sq_sum = None
                        for vxb, vyb in vldr:
                            vxb, vyb = vxb.to(device), vyb.to(device)
                            vpred = predictors[name](vxb)
                            val_loss_sum += compute_loss(vpred, vyb, loss_type).item() * vxb.shape[0]
                            val_n += vxb.shape[0]
                            res = (vpred - vyb).cpu()
                            ss_res_sum += res.pow(2).sum().item()
                            vyb_c = vyb.cpu()
                            if tgt_sum is None:
                                tgt_sum = vyb_c.sum(dim=0)
                                tgt_sq_sum = vyb_c.pow(2).sum(dim=0)
                            else:
                                tgt_sum += vyb_c.sum(dim=0)
                                tgt_sq_sum += vyb_c.pow(2).sum(dim=0)
                    else:
                        projector.eval()
                        vz, vh, vte, vy = val_ctrl_inputs[name]
                        vds = TensorDataset(vz, vh, vte, vy)
                        vldr = DataLoader(vds, batch_size=closure_batch_size, shuffle=False)
                        val_loss_sum, val_n = 0.0, 0
                        ss_res_sum = 0.0
                        tgt_sum = tgt_sq_sum = None
                        for vzb, vhb, vteb, vyb in vldr:
                            vzb, vhb, vteb, vyb = vzb.to(device), vhb.to(device), vteb.to(device), vyb.to(device)
                            vu = projector(vteb)
                            vx = torch.cat([vzb, vu], dim=-1)
                            vpred = predictors[name](vx)
                            val_loss_sum += compute_loss(vpred, vyb, loss_type).item() * vzb.shape[0]
                            val_n += vzb.shape[0]
                            res = (vpred - vyb).cpu()
                            ss_res_sum += res.pow(2).sum().item()
                            vyb_c = vyb.cpu()
                            if tgt_sum is None:
                                tgt_sum = vyb_c.sum(dim=0)
                                tgt_sq_sum = vyb_c.pow(2).sum(dim=0)
                            else:
                                tgt_sum += vyb_c.sum(dim=0)
                                tgt_sq_sum += vyb_c.pow(2).sum(dim=0)

                    val_loss = val_loss_sum / max(val_n, 1)
                    val_r2 = compute_r2_streaming(ss_res_sum, tgt_sum, tgt_sq_sum, val_n)

                all_val_losses[name].append(val_loss)
                all_val_r2[name].append(val_r2)

                if val_loss < best_val_losses[name] - min_delta:
                    best_val_losses[name] = val_loss
                    best_states[name] = {k: v.clone() for k, v in predictors[name].state_dict().items()}
                    if name == "controlled":
                        best_proj_state = {k: v.clone() for k, v in projector.state_dict().items()}
                    evals_without_improvement[name] = 0
                else:
                    evals_without_improvement[name] += 1

                if evals_without_improvement[name] >= patience:
                    active_variants.discard(name)
                    print(f"  {name}: early stop @ step {step}")

            # Progress bar shows best active variant
            if active_variants:
                best_name = min(active_variants, key=lambda n: best_val_losses[n])
                pbar.set_postfix(
                    active=len(active_variants),
                    best_R2=f"{all_val_r2[best_name][-1]:.4f}" if all_val_r2[best_name] else "?",
                )

    pbar.close()

    # Restore best states
    for name in predictors:
        if best_states[name] is not None:
            predictors[name].load_state_dict(best_states[name])
    if best_proj_state is not None:
        projector.load_state_dict(best_proj_state)

    # Assemble trained dict
    trained = {}
    input_dims = {
        "autonomous": d_repr,
        "controlled": d_repr + control_dim,
    }
    for name in predictors:
        val_r2_final = all_val_r2[name][-1] if all_val_r2[name] else None
        print(f"  {name}: val_R²={val_r2_final:.4f}" if val_r2_final is not None else f"  {name}: no eval")
        trained[name] = {
            "model": predictors[name],
            "input_dim": input_dims[name],
            "output_dim": d_repr,
            "loss_type": loss_type,
            "hidden_dim": hidden_dim,
            "num_layers": ccfg.num_layers,
            "train_losses": all_train_losses[name],
            "val_losses": all_val_losses[name],
            "val_r2_final": val_r2_final,
        }

    return trained, projector, target_variance


# ---------------------------------------------------------------------------
# Backward-compat wrappers
# ---------------------------------------------------------------------------

def train_all_variants_model(
    train_loader,
    extractor,
    encoder,
    val_data: dict[str, torch.Tensor],
    d_z: int,
    d_h: int,
    config,
    loss_type: str,
    device: str,
) -> tuple[dict, nn.Module, float]:
    """Train autonomous and controlled closure variants for an RET encoder checkpoint.

    Thin wrapper around :func:`train_all_variants` with the model encode_fn.

    Returns (trained_dict, projector, target_variance).
    """
    def encode_fn(h, attention_mask):
        z, _aux = (
            encoder(h, attention_mask=attention_mask)
            if attention_mask is not None else encoder(h)
        )
        if attention_mask is not None:
            pair_mask = (attention_mask[:, :-1] * attention_mask[:, 1:]).bool()
            z_t = z[:, :-1][pair_mask]
            z_next = z[:, 1:][pair_mask]
        else:
            z_t = z[:, :-1].reshape(-1, d_z)
            z_next = z[:, 1:].reshape(-1, d_z)
        return z_t, z_next

    # val_data already has z_t, z_next from collect_val_data (encoder was passed)
    adapted_val = {
        "repr_t": val_data["z_t"],
        "repr_next": val_data["z_next"],
        "h_t": val_data["h_t"],
        "token_emb_next": val_data["token_emb_next"],
    }

    return train_all_variants(
        train_loader=train_loader,
        extractor=extractor,
        encode_fn=encode_fn,
        val_data=adapted_val,
        d_repr=d_z,
        d_h=d_h,
        config=config,
        loss_type=loss_type,
        device=device,
        repr_label="model",
        cfg_section=config.closure,
    )


def train_all_variants_sae(
    train_loader,
    extractor,
    sae,
    val_data: dict[str, torch.Tensor],
    d_sae: int,
    d_h: int,
    config,
    loss_type: str,
    device: str,
) -> tuple[dict, nn.Module, float]:
    """Train autonomous and controlled SAE closure variants using shared extraction.

    Thin wrapper around :func:`train_all_variants` with the SAE encode_fn.

    Returns (trained_dict, projector, target_variance).
    """
    from evaluation.baselines.sae import encode_sae

    # Pre-encode val h through SAE
    val_h_t = val_data["h_t"]
    val_h_next = val_data["h_next"]
    val_sae_t = encode_sae(sae, val_h_t.to(device))       # returns CPU
    val_sae_next = encode_sae(sae, val_h_next.to(device))  # returns CPU

    def encode_fn(h, attention_mask):
        sae_acts = encode_sae(sae, h)  # (B, T, d_sae), CPU
        if attention_mask is not None:
            pair_mask = (attention_mask[:, :-1] * attention_mask[:, 1:]).bool()
            pair_mask_cpu = pair_mask.cpu()
            sae_t = sae_acts[:, :-1][pair_mask_cpu]
            sae_next = sae_acts[:, 1:][pair_mask_cpu]
        else:
            sae_t = sae_acts[:, :-1].reshape(-1, d_sae)
            sae_next = sae_acts[:, 1:].reshape(-1, d_sae)
        # Return on device for consistency with the training loop
        return sae_t.to(device), sae_next.to(device)

    adapted_val = {
        "repr_t": val_sae_t,
        "repr_next": val_sae_next,
        "h_t": val_data["h_t"],
        "token_emb_next": val_data["token_emb_next"],
    }

    return train_all_variants(
        train_loader=train_loader,
        extractor=extractor,
        encode_fn=encode_fn,
        val_data=adapted_val,
        d_repr=d_sae,
        d_h=d_h,
        config=config,
        loss_type=loss_type,
        device=device,
        repr_label="sae",
        cfg_section=config.closure.sae,
    )

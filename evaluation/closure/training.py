"""Training loops for closure predictors."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from evaluation.closure.predictors import (
    build_closure_predictor,
    build_token_projector,
    compute_loss,
    compute_r2,
)


def train_closure_predictor(
    input_dim: int,
    output_dim: int,
    train_iter,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    loss_type: str,
    hidden_dim: int = 256,
    num_layers: int = 2,
    device: str = "cuda",
    max_steps: int = 5000,
    lr: float = 1e-3,
    batch_size: int = 1024,
    patience: int = 15,
    eval_every: int = 200,
    min_delta: float = 1e-6,
) -> tuple[nn.Module, list[float], list[float], list[float]]:
    """Train a closure predictor with streaming data and step-based early stopping.

    Args:
        train_iter: infinite iterator yielding (x_batch, y_batch) pairs on the
            target device. Each x_batch: (N, input_dim), y_batch: (N, output_dim).
        val_x, val_y: pre-collected validation tensors for periodic eval.
        max_steps: total training steps.
        eval_every: steps between validation evaluations.
        patience: eval cycles without val improvement before early stopping.

    Returns:
        (best_model, train_losses, val_losses, val_r2_list) where losses/r2
        are recorded per eval cycle.
    """
    model = build_closure_predictor(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Validation batched eval (handles large high-dim tensors)
    val_dataset = TensorDataset(val_x, val_y)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_r2_list: list[float] = []
    best_val_loss = float("inf")
    best_state_dict = None
    evals_without_improvement = 0
    running_loss = 0.0
    n_train_batches = 0

    pbar = tqdm(range(1, max_steps + 1), desc="    Training", leave=True)
    for step in pbar:
        model.train()
        xb, yb = next(train_iter)
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = compute_loss(pred, yb, loss_type)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        n_train_batches += 1

        if step % eval_every == 0:
            avg_train_loss = running_loss / max(n_train_batches, 1)
            train_losses.append(avg_train_loss)
            running_loss = 0.0
            n_train_batches = 0

            model.eval()
            with torch.no_grad():
                val_loss_sum = 0.0
                val_n = 0
                all_vpred, all_vtgt = [], []
                for vxb, vyb in val_loader:
                    vxb, vyb = vxb.to(device), vyb.to(device)
                    vpred = model(vxb)
                    val_loss_sum += compute_loss(vpred, vyb, loss_type).item() * vxb.shape[0]
                    val_n += vxb.shape[0]
                    all_vpred.append(vpred)
                    all_vtgt.append(vyb)
                val_loss = val_loss_sum / max(val_n, 1)
                val_r2 = compute_r2(torch.cat(all_vpred), torch.cat(all_vtgt))
            val_losses.append(val_loss)
            val_r2_list.append(val_r2)

            pbar.set_postfix(
                train=f"{avg_train_loss:.5f}", val=f"{val_loss:.5f}",
                best=f"{best_val_loss:.5f}", R2=f"{val_r2:.4f}",
            )

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
                evals_without_improvement = 0
            else:
                evals_without_improvement += 1

            if evals_without_improvement >= patience:
                pbar.set_description(f"    Early stop @ step {step}")
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model, train_losses, val_losses, val_r2_list


def train_closure_predictor_with_projector(
    z_dim: int,
    d_h: int,
    output_dim: int,
    train_iter,
    val_z: torch.Tensor,
    val_h: torch.Tensor,
    val_token_emb: torch.Tensor,
    val_y: torch.Tensor,
    loss_type: str,
    control_dim: int = 32,
    hidden_dim: int = 256,
    num_layers: int = 2,
    device: str = "cuda",
    max_steps: int = 5000,
    lr: float = 1e-3,
    batch_size: int = 1024,
    patience: int = 15,
    eval_every: int = 200,
    min_delta: float = 1e-6,
) -> tuple[nn.Module, nn.Module, list[float], list[float], list[float]]:
    """Train a closure predictor jointly with a fresh TokenProjector for the controlled variant.

    Input = [z_t, u_next] where u_next = TokenProjector(token_emb_next).

    Args:
        train_iter: infinite iterator yielding (z_batch, h_batch, token_emb_batch, y_batch)
            tuples on the target device.
        val_z, val_h, val_token_emb, val_y: pre-collected validation tensors.
        max_steps: total training steps.
        eval_every: steps between validation evaluations.
        patience: eval cycles without val improvement before early stopping.

    Returns:
        (best_predictor, best_projector, train_losses, val_losses, val_r2_list).
    """
    input_dim = z_dim + control_dim

    predictor = build_closure_predictor(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)

    projector = build_token_projector(d_h, control_dim).to(device)

    optimizer = torch.optim.Adam(
        list(predictor.parameters()) + list(projector.parameters()), lr=lr
    )

    val_ds = TensorDataset(val_z, val_h, val_token_emb, val_y)
    v_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_r2_list: list[float] = []
    best_val_loss = float("inf")
    best_pred_state = None
    best_proj_state = None
    evals_without_improvement = 0
    running_loss = 0.0
    n_train_batches = 0

    pbar = tqdm(range(1, max_steps + 1), desc="    Training", leave=True)
    for step in pbar:
        predictor.train()
        projector.train()

        zb, hb, teb, yb = next(train_iter)
        zb, hb, teb, yb = zb.to(device), hb.to(device), teb.to(device), yb.to(device)
        u = projector(teb)
        x = torch.cat([zb, u], dim=-1)
        pred = predictor(x)
        loss = compute_loss(pred, yb, loss_type)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        n_train_batches += 1

        if step % eval_every == 0:
            avg_train_loss = running_loss / max(n_train_batches, 1)
            train_losses.append(avg_train_loss)
            running_loss = 0.0
            n_train_batches = 0

            predictor.eval()
            projector.eval()
            with torch.no_grad():
                val_loss_sum = 0.0
                val_n = 0
                all_vpred, all_vtgt = [], []
                for vzb, vhb, vteb, vyb in v_loader:
                    vzb, vhb, vteb, vyb = vzb.to(device), vhb.to(device), vteb.to(device), vyb.to(device)
                    vu = projector(vteb)
                    vx = torch.cat([vzb, vu], dim=-1)
                    vpred = predictor(vx)
                    val_loss_sum += compute_loss(vpred, vyb, loss_type).item() * vzb.shape[0]
                    val_n += vzb.shape[0]
                    all_vpred.append(vpred)
                    all_vtgt.append(vyb)
                val_loss = val_loss_sum / max(val_n, 1)
                val_r2 = compute_r2(torch.cat(all_vpred), torch.cat(all_vtgt))
            val_losses.append(val_loss)
            val_r2_list.append(val_r2)

            pbar.set_postfix(
                train=f"{avg_train_loss:.5f}", val=f"{val_loss:.5f}",
                best=f"{best_val_loss:.5f}", R2=f"{val_r2:.4f}",
            )

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_pred_state = {k: v.clone() for k, v in predictor.state_dict().items()}
                best_proj_state = {k: v.clone() for k, v in projector.state_dict().items()}
                evals_without_improvement = 0
            else:
                evals_without_improvement += 1

            if evals_without_improvement >= patience:
                pbar.set_description(f"    Early stop @ step {step}")
                break

    if best_pred_state is not None:
        predictor.load_state_dict(best_pred_state)
    if best_proj_state is not None:
        projector.load_state_dict(best_proj_state)

    return predictor, projector, train_losses, val_losses, val_r2_list

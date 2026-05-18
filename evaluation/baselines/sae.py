"""Unified TopK Sparse Autoencoder (SAE) encoder.

Provides the canonical TopKSAE class plus general-purpose helpers used by
any experiment that needs a pre-trained SAE (closure probes, sae-steering,
embedding/trajectory analyses, baseline-aware probing).

Public API
----------
TopKSAE
    TopK Sparse Autoencoder encoder (encode-only).

_load_sae_from_state_dict(state_dict, device)
    Create a TopKSAE from a raw state dict, handling three checkpoint formats.

load_sae_baseline(llm_config, device, sae_release, sae_id)
    Load a pre-trained SAE for a given LLM + layer from andyrdt /
    SAELens / EleutherAI HF (whichever is available).

encode_sae(sae, h, batch_size, device)
    Batched encoding of hidden states to SAE latent activations.

DEFAULT_SAE_HF_REPO, DEFAULT_SAE_FILENAME, DEFAULT_SAE_K
    Default constants for the GPT-OSS-20B layer-15 SAE on HuggingFace.
"""
from __future__ import annotations

import json
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# Default HuggingFace repo for GPT-OSS-20B layer 15 SAE
DEFAULT_SAE_HF_REPO = "Goodfire/Hackathon-gpt-oss-20b-SAE-l15"
DEFAULT_SAE_FILENAME = "topk_sae_l15_exp16_k32.pt"
DEFAULT_SAE_K = 32


class TopKSAE(nn.Module):
    """TopK Sparse Autoencoder encoder.

    Implements encode-only path: hidden state -> sparse feature activations.
    Only the top-k pre-activations are kept (with ReLU), rest are zeroed.

    Parameters
    ----------
    encoder_weight : (num_latents, d_in) — encoder projection matrix
    encoder_bias   : (num_latents,) — encoder bias
    b_dec          : (d_in,) — decoder bias (used for input centering)
    k              : number of top activations to keep
    """

    def __init__(self, encoder_weight: torch.Tensor, encoder_bias: torch.Tensor,
                 b_dec: torch.Tensor, k: int):
        super().__init__()
        self.register_buffer("encoder_weight", encoder_weight)   # (num_latents, d_in)
        self.register_buffer("encoder_bias", encoder_bias)       # (num_latents,)
        self.register_buffer("b_dec", b_dec)                     # (d_in,)
        self.k = k
        self.d_sae = encoder_weight.shape[0]   # output dimension (num_latents)
        self.d_in = encoder_weight.shape[1]    # input dimension (hidden_dim)

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        """Encode hidden states to sparse feature activations.

        Args:
            h: (..., d_in) hidden states

        Returns:
            (..., num_latents) sparse feature activations (only top-k nonzero)
        """
        h_centered = h - self.b_dec
        pre_acts = h_centered @ self.encoder_weight.T + self.encoder_bias
        topk_vals, topk_idx = pre_acts.topk(self.k, dim=-1)
        topk_vals = torch.relu(topk_vals)
        acts = torch.zeros_like(pre_acts)
        acts.scatter_(-1, topk_idx, topk_vals)
        return acts


def _load_sae_from_state_dict(state_dict: dict, device: str) -> TopKSAE:
    """Create TopKSAE from a state dict, handling multiple checkpoint formats.

    Supported formats:
    1. Goodfire/EleutherAI HF format: OrderedDict with
       'encoder.weight', 'encoder.bias', 'b_dec'
    2. Our sparse_autoencoder format: Autoencoder state dict with
       'encoder.weight', 'latent_bias', 'pre_bias', activation info
    """
    # Format 1: Goodfire HF format (encoder_linear.weight, decoder_bias, no encoder bias)
    if "encoder_linear.weight" in state_dict and "decoder_bias" in state_dict:
        encoder_weight = state_dict["encoder_linear.weight"]  # (d_sae, d_in)
        b_dec = state_dict["decoder_bias"]                    # (d_in,)
        # No encoder bias in this format — use zeros
        encoder_bias = torch.zeros(encoder_weight.shape[0])
        k = state_dict.get("k", DEFAULT_SAE_K)
        sae = TopKSAE(encoder_weight, encoder_bias, b_dec, k=k)
        sae.to(device)
        sae.eval()
        return sae

    # Format 2: EleutherAI-style (encoder.weight, encoder.bias, b_dec)
    if "encoder.weight" in state_dict and "encoder.bias" in state_dict and "b_dec" in state_dict:
        encoder_weight = state_dict["encoder.weight"]
        encoder_bias = state_dict["encoder.bias"]
        b_dec = state_dict["b_dec"]
        k = DEFAULT_SAE_K
        if "activation_state_dict" in state_dict:
            act_sd = state_dict["activation_state_dict"]
            if "k" in act_sd:
                k = act_sd["k"]
        elif "k" in state_dict:
            k = state_dict["k"]
        sae = TopKSAE(encoder_weight, encoder_bias, b_dec, k=k)
        sae.to(device)
        sae.eval()
        return sae

    # Format 3: Our Autoencoder format (encoder.weight, latent_bias, pre_bias)
    if "encoder.weight" in state_dict and "latent_bias" in state_dict:
        encoder_weight = state_dict["encoder.weight"]
        encoder_bias = state_dict["latent_bias"]
        b_dec = state_dict["pre_bias"]
        k = DEFAULT_SAE_K
        if "activation_state_dict" in state_dict:
            act_sd = state_dict["activation_state_dict"]
            if "k" in act_sd:
                k = act_sd["k"]
        elif "activation" in state_dict and state_dict["activation"] == "TopK":
            pass  # k already set to default
        sae = TopKSAE(encoder_weight, encoder_bias, b_dec, k=k)
        sae.to(device)
        sae.eval()
        return sae

    raise ValueError(
        f"Unrecognized SAE checkpoint format. "
        f"Keys: {list(state_dict.keys())[:20]}... "
        f"Expected 'encoder_linear.weight'+'decoder_bias' or "
        f"'encoder.weight'+'encoder.bias'+'b_dec' or "
        f"'encoder.weight'+'latent_bias'+'pre_bias'."
    )


def load_sae_baseline(
    llm_config,
    device: str = "cuda",
    sae_release: str | None = None,
    sae_id: str | None = None,
):
    """Load a pre-trained SAE for the configured LLM + layer.

    Tries andyrdt (gpt-oss-20b), SAELens, then EleutherAI's HuggingFace SAEs.

    Returns (sae, release_str, id_str) or None if loading fails.
    """
    layer_idx = llm_config.layer_index

    # --- Attempt 0: andyrdt/saes-gpt-oss-20b (BatchTopK SAEs at layers 3,11,15,19,23) ---
    if llm_config.name == "openai/gpt-oss-20b" and sae_release is None:
        try:
            from huggingface_hub import hf_hub_download
            repo_id = "andyrdt/saes-gpt-oss-20b"
            folder = sae_id or f"resid_post_layer_{layer_idx}/trainer_0"
            cfg_path = hf_hub_download(repo_id, f"{folder}/config.json")
            ae_path = hf_hub_download(repo_id, f"{folder}/ae.pt")
            with open(cfg_path) as f:
                cfg = json.load(f)
            # map_location=cpu so unused decoder.weight / threshold / etc. don't
            # consume GPU memory before we extract only the encoder pieces we need.
            state = torch.load(ae_path, map_location="cpu", weights_only=False)
            sae = _load_sae_from_state_dict(state, device)
            d_sae = sae.d_sae
            d_in = sae.d_in
            del state
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"Loaded SAE (andyrdt): {repo_id} / {folder}")
            print(f"  d_in={d_in}, num_latents={d_sae}, k={sae.k}")
            return sae, repo_id, folder
        except Exception as e:
            print(f"andyrdt SAE load failed ({e}); trying other sources.")

    # --- Attempt 1: SAELens ---
    try:
        from sae_lens import SAE
        release = sae_release or f"sae_bench_{llm_config.name.replace('/', '-')}_sweep_topk_ctx128_0730"
        sid = sae_id or f"blocks.{layer_idx}.hook_resid_post__trainer_0"
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release=release, sae_id=sid, device=device,
        )
        sae.eval()
        print(f"Loaded SAE (SAELens): {release} / {sid}")
        return sae, release, sid
    except ImportError:
        pass  # sae_lens not installed, try fallback
    except Exception:
        pass  # release not found, try fallback

    # --- Attempt 2: EleutherAI HuggingFace SAEs ---
    # Available for pythia models at EleutherAI/sae-<model>-32k
    model_short = llm_config.name.split("/")[-1]  # e.g. "pythia-160m"
    repo_id = sae_release or f"EleutherAI/sae-{model_short}-32k"
    folder = sae_id or f"layers.{layer_idx}"

    try:
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open

        cfg_path = hf_hub_download(repo_id, f"{folder}/cfg.json")
        weights_path = hf_hub_download(repo_id, f"{folder}/sae.safetensors")

        with open(cfg_path) as f:
            cfg = json.load(f)

        with safe_open(weights_path, framework="pt", device=device) as sf:
            encoder_weight = sf.get_tensor("encoder.weight")
            encoder_bias = sf.get_tensor("encoder.bias")
            b_dec = sf.get_tensor("b_dec")

        k = cfg.get("k", 32)
        sae = TopKSAE(encoder_weight, encoder_bias, b_dec, k=k).to(device)
        sae.eval()

        d_sae = cfg.get("num_latents", encoder_weight.shape[0])
        d_in = cfg.get("d_in", encoder_weight.shape[1])
        print(f"Loaded SAE (EleutherAI HF): {repo_id} / {folder}")
        print(f"  d_in={d_in}, num_latents={d_sae}, k={k}")
        return sae, repo_id, folder

    except Exception as e:
        print(f"WARNING: Could not load SAE: {e}")
        print(f"  Tried SAELens and HuggingFace ({repo_id} / {folder})")
        print("  Skipping SAE baseline.")
        return None


def encode_sae(sae, h: torch.Tensor, batch_size: int = 4096, device: str | None = None) -> torch.Tensor:
    """Encode hidden states through SAE, returning SAE latent activations.

    Processes in batches, moving each chunk onto the SAE's own device before
    encoding (so ``h`` can sit on CPU while the SAE lives on GPU). Output
    accumulates on ``device`` if given, else on CPU (for very large batches
    that wouldn't fit on GPU). Keep ``device`` set to the SAE's device during
    training steps to avoid costly CPU round-trips.
    """
    sae_device = sae.encoder_weight.device
    original_shape = h.shape[:-1]
    d_h = h.shape[-1]
    h_flat = h.reshape(-1, d_h)

    keep_on_gpu = device is not None
    chunks = []
    with torch.no_grad():
        for i in range(0, h_flat.shape[0], batch_size):
            chunk = h_flat[i : i + batch_size].to(sae_device)
            out = sae.encode(chunk)
            chunks.append(out if keep_on_gpu else out.cpu())
    feature_acts = torch.cat(chunks, dim=0)

    return feature_acts.reshape(*original_shape, -1)

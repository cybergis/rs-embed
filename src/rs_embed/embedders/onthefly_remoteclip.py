# src/rs_embed/embedders/onthefly_remoteclip.py
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any

import numpy as np
import xarray as xr

from ..core.embedding import Embedding
from ..core.errors import ModelError
from ..core.registry import register
from ..core.specs import (
    ModelInputSpec,
    OutputSpec,
    SensorSpec,
    SpatialSpec,
    TemporalSpec,
)
from ..providers import ProviderBase

# -----------------------------
# Provider: Fetch S2 RGB
# -----------------------------
from ..providers.fetch import (
    fetch_s2_rgb_chw as _fetch_s2_rgb_chw_shared,
)
from ..providers.resolution import (
    is_provider_backend,
)
from ..tools.runtime import (
    resolve_device_auto_torch,
)
from ..tools.shape import (
    crop_grid_to_roi,
    geo_roi_from_meta,
    roi_is_full,
    square_fetch_batch,
)
from ..tools.spatial import square_spatial
from .base import EmbedderBase
from .meta import build_meta, temporal_to_range


def _s2_rgb_u8_from_chw(s2_chw):
    x = np.clip(np.asarray(s2_chw, dtype=np.float32) / 10000.0, 0.0, 1.0)
    return (x.transpose(1, 2, 0) * 255.0).astype(np.uint8)


def _fetch_s2_rgb_chw(
    provider: ProviderBase,
    spatial: SpatialSpec,
    temporal: TemporalSpec,
    *,
    scale_m: int = 10,
    cloudy_pct: int = 30,
    composite: str = "median",
) -> np.ndarray:
    return _fetch_s2_rgb_chw_shared(
        provider,
        spatial=spatial,
        temporal=temporal,
        scale_m=int(scale_m),
        cloudy_pct=int(cloudy_pct),
        composite=str(composite),
    )


# -----------------------------
# HF weight management (strict)
# -----------------------------
def _find_weight_file(path: str) -> str | None:
    for fn in (
        "model.safetensors",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    ):
        p = os.path.join(path, fn)
        if os.path.exists(p):
            return p
    return None


def _ensure_hf_weights(
    repo_id_or_path: str,
    *,
    auto_download: bool = True,
    require_pretrained: bool = True,
    cache_dir: str | None = None,
    min_bytes: int = 50 * 1024 * 1024,  # 50MB: below this is almost surely pointer/metadata
) -> tuple[str, str | None]:
    """
    Ensure pretrained weights are present locally.
    Returns (local_dir, weight_file_path).
    """
    if os.path.exists(repo_id_or_path):
        wf = _find_weight_file(repo_id_or_path)
        if require_pretrained:
            if wf is None:
                raise ModelError(
                    f"Local ckpt path '{repo_id_or_path}' has no weights file "
                    "(expected model.safetensors or pytorch_model.bin)."
                )
            if wf.endswith(".safetensors") and os.path.getsize(wf) < min_bytes:
                raise ModelError(
                    f"Local '{wf}' is too small to be real weights (size={os.path.getsize(wf)}). "
                    "It looks like a pointer/placeholder."
                )
        return repo_id_or_path, wf

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise ModelError(
            "Install huggingface_hub to download/verify weights: pip install huggingface_hub"
        ) from e

    if auto_download:
        local_dir = snapshot_download(
            repo_id=repo_id_or_path,
            cache_dir=cache_dir,
            local_files_only=False,
        )
    else:
        local_dir = snapshot_download(
            repo_id=repo_id_or_path,
            cache_dir=cache_dir,
            local_files_only=True,
        )

    wf = _find_weight_file(local_dir)
    if require_pretrained:
        if wf is None:
            raise ModelError(
                f"Downloaded snapshot for '{repo_id_or_path}' but no weights file found in {local_dir}."
            )
        if wf.endswith(".safetensors") and os.path.getsize(wf) < min_bytes:
            raise ModelError(
                f"Found '{wf}' but it's only {os.path.getsize(wf)} bytes — likely a xet/LFS pointer, not real weights.\n"
                "Fix:\n"
                "  pip install -U hf_xet\n"
                '  (optional) pip install -U "huggingface_hub[hf_transfer]"\n'
                "Then delete the cached snapshot and re-run.\n"
            )
    return local_dir, wf


def _assert_weights_loaded(model) -> dict[str, float]:
    """Best-effort sanity check that weights are loaded (do not trust rshf warnings)."""
    import torch

    core = getattr(model, "model", model)
    p = None
    for _, param in core.named_parameters():
        if param is not None and param.numel() > 0:
            p = param.detach()
            break
    if p is None:
        raise ModelError("RemoteCLIP model has no parameters; cannot verify weights.")
    if not torch.isfinite(p).all():
        raise ModelError("RemoteCLIP parameters contain NaN/Inf; load likely failed.")

    p_f = p.float()
    std = float(p_f.std().cpu())
    mx = float(p_f.abs().max().cpu())
    mean = float(p_f.mean().cpu())
    if std < 1e-6 and mx < 1e-5:
        raise ModelError("RemoteCLIP parameters look uninitialized (near-zero stats).")
    return {"param_mean": mean, "param_std": std, "param_absmax": mx}


@contextmanager
def _suppress_rshf_pretrained_init_warning():
    """
    Suppress a known misleading rshf/open_clip warning emitted during model construction:
      "No pretrained weights loaded ... Model initialized randomly."
    We verify actual loaded weights immediately after construction.
    """
    root_logger = logging.getLogger()
    suppressed: dict[str, Any] = {"count": 0, "messages": []}

    class _KnownWarningFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            is_target = (
                "No pretrained weights loaded for model" in msg
                and "Model initialized randomly" in msg
            )
            if is_target:
                suppressed["count"] += 1
                suppressed["messages"].append(msg)
                return False
            return True

    filt = _KnownWarningFilter()
    root_logger.addFilter(filt)
    try:
        yield suppressed
    finally:
        root_logger.removeFilter(filt)


def _load_rshf_remoteclip(
    ckpt: str,
    *,
    auto_download: bool = True,
    require_pretrained: bool = True,
    cache_dir: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load rshf RemoteCLIP with explicit weight checks. Returns (model, weight_meta)."""
    try:
        from rshf.remoteclip import RemoteCLIP
    except Exception as e:
        raise ModelError("RemoteCLIP requires rshf. Install: pip install rshf") from e

    local_dir, weight_file = _ensure_hf_weights(
        ckpt,
        auto_download=auto_download,
        require_pretrained=require_pretrained,
        cache_dir=cache_dir,
    )

    with _suppress_rshf_pretrained_init_warning() as suppressed_warning:
        model = RemoteCLIP.from_pretrained(local_dir if os.path.exists(local_dir) else ckpt)
    stats = _assert_weights_loaded(model)

    meta = {
        "ckpt_input": ckpt,
        "ckpt_local_dir": local_dir,
        "weight_file": weight_file,
        "weight_file_size": (
            os.path.getsize(weight_file) if (weight_file and os.path.exists(weight_file)) else None
        ),
        "weights_verified": True,
        "init_warning_suppressed_count": int(suppressed_warning.get("count", 0)),
        **stats,
    }
    return model, meta


# -----------------------------
# Token -> grid helpers
# -----------------------------
def _is_perfect_square(n: int) -> bool:
    r = int(np.sqrt(n))
    return r * r == n


def _tokens_to_grid_dhw(tokens_nd: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """
    tokens_nd: [N, D] or [N+1, D] (maybe with CLS)
    Returns: grid_dhw: [D, Ht, Wt] and meta about CLS handling and grid size.
    """
    if tokens_nd.ndim != 2:
        raise ModelError(f"Expected tokens [N,D], got {tokens_nd.shape}")
    n, d = tokens_nd.shape

    cls_removed = False
    if _is_perfect_square(n):
        ht = wt = int(np.sqrt(n))
        tok = tokens_nd
    elif _is_perfect_square(n - 1):
        # common: first token is CLS
        cls_removed = True
        tok = tokens_nd[1:, :]
        ht = wt = int(np.sqrt(n - 1))
    else:
        raise ModelError(
            f"Token count N={n} is not a square (or N-1). Cannot reshape into HxW grid."
        )

    grid_hwd = tok.reshape(ht, wt, d)  # [Ht, Wt, D]
    grid_dhw = np.transpose(grid_hwd, (2, 0, 1))  # [D, Ht, Wt]
    meta = {"token_count": n, "dim": d, "grid_hw": (ht, wt), "cls_removed": cls_removed}
    return grid_dhw.astype(np.float32), meta


def _first_nchw_grid(out: Any):
    """Pull the final patch-grid feature map out of an open_clip
    ``forward_intermediates()`` return value as a ``[B, D, h, w]`` tensor.

    open_clip returns the per-block visual features as ``image_intermediates``:
    a list of NCHW tensors (e.g. ``[1, 768, 7, 7]`` for ViT-B/32 @ 224). We take
    the last block's output (the deepest dense features). Returns ``None`` when no
    NCHW grid is present so callers can fall back to other extraction paths.
    """
    import torch

    def _last_nchw(seq: Any):
        if isinstance(seq, (list, tuple)) and seq:
            last = seq[-1]
            if torch.is_tensor(last) and last.ndim == 4:
                return last
        return None

    if isinstance(out, dict):
        g = _last_nchw(out.get("image_intermediates"))
        if g is not None:
            return g
        for v in out.values():
            g = _last_nchw(v)
            if g is not None:
                return g
            if torch.is_tensor(v) and v.ndim == 4:
                return v
    elif isinstance(out, (list, tuple)):
        for v in out:
            g = _last_nchw(v)
            if g is not None:
                return g
            if torch.is_tensor(v) and v.ndim == 4:
                return v
    return None


# -----------------------------
# RemoteCLIP inference adapter
# -----------------------------
def _remoteclip_encode_tokens(
    model,
    rgb_u8: np.ndarray,
    *,
    image_size: int = 224,
    device: str = "auto",
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Return tokens if possible; else return pooled vector.

    Priority:
      1) forward_encoder -> tokens [N,D] (not available in your current rshf)
      2) open_clip CLIP.forward_intermediates -> tokens (if returns)
      3) hook core.visual.transformer -> capture tokens while running core.encode_image(x)
      4) fallback encode_image -> pooled only

    Returns:
      - tokens: [N,D]  (tokens_kind='tokens' or 'tokens_hook' or 'tokens_intermediates')
      - pooled: [D]    (tokens_kind='pooled')
    """
    import torch
    from PIL import Image
    from torchvision import transforms

    dev = (
        "cuda"
        if (device == "auto" and torch.cuda.is_available())
        else ("cpu" if device == "auto" else device)
    )
    model = model.to(dev).eval()
    core = getattr(model, "model", model)

    # --- preprocess to tensor ---
    if hasattr(model, "transform") and callable(model.transform):
        x = model.transform(rgb_u8.astype(np.float32), image_size).unsqueeze(0)
    else:
        preprocess = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        img = Image.fromarray(rgb_u8, mode="RGB")
        x = preprocess(img).unsqueeze(0)

    x = x.to(dev)

    with torch.no_grad():
        # 1) forward_encoder (not in your current wrapper)
        fe = None
        if hasattr(model, "forward_encoder"):
            fe = model.forward_encoder
        elif hasattr(core, "forward_encoder"):
            fe = core.forward_encoder

        if fe is not None:
            try:
                out = fe(x, mask_ratio=0.0)
            except TypeError:
                out = fe(x)
            toks = out[0] if isinstance(out, (tuple, list)) else out
            if toks.ndim == 3:
                return toks[0].detach().float().cpu().numpy().astype(np.float32), {
                    "tokens_kind": "tokens"
                }
            if toks.ndim == 2:
                return toks[0].detach().float().cpu().numpy().astype(np.float32), {
                    "tokens_kind": "pooled"
                }
            raise ModelError(f"Unexpected forward_encoder output shape: {tuple(toks.shape)}")

        # 2) open_clip: forward_intermediates -> image_intermediates as NCHW patch
        #    grids ([B, D, h, w]). This is the canonical dense-feature path; we take
        #    the deepest block and flatten it to row-major tokens [N, D].
        if hasattr(core, "forward_intermediates"):
            try:
                out = core.forward_intermediates(x)
                grid_t = _first_nchw_grid(out)
                if grid_t is not None:
                    g = grid_t[0]  # [D, h, w]
                    d, gh, gw = int(g.shape[0]), int(g.shape[1]), int(g.shape[2])
                    # row-major (row*gw + col) order so _tokens_to_grid_dhw round-trips.
                    toks = g.reshape(d, gh * gw).permute(1, 0).contiguous()  # [N, D]
                    return toks.detach().float().cpu().numpy().astype(np.float32), {
                        "tokens_kind": "tokens_intermediates"
                    }
            except Exception as _e:
                # If forward_intermediates exists but signature/return differs, fall back to hook
                pass

        # 3) hook vision transformer output to get tokens (fallback for open_clip
        #    builds without forward_intermediates).
        if hasattr(core, "visual") and hasattr(core.visual, "transformer"):
            captured = {}

            def _hook(_module, _inp, outp):
                captured["tokens"] = outp

            handle = core.visual.transformer.register_forward_hook(_hook)
            try:
                # run a normal encode_image forward; hook captures tokens
                _ = core.encode_image(x) if hasattr(core, "encode_image") else core.forward(x)
            finally:
                handle.remove()

            t = captured.get("tokens")
            if torch.is_tensor(t) and t.ndim == 3:
                # We always run a single image (B=1). open_clip transformers may be
                # batch-first ([B, N, D]) or sequence-first ([N, B, D]); the axis of
                # size 1 is the batch dim, so collapse it to recover [N, D] either way.
                if t.shape[0] == 1 and t.shape[1] != 1:
                    toks = t[0]
                elif t.shape[1] == 1 and t.shape[0] != 1:
                    toks = t[:, 0, :]
                else:
                    bf = bool(getattr(core.visual.transformer, "batch_first", True))
                    toks = t[0] if bf else t[:, 0, :]
                return toks.detach().float().cpu().numpy().astype(np.float32), {
                    "tokens_kind": "tokens_hook"
                }

        # 4) pooled fallback only
        if hasattr(core, "encode_image"):
            v = core.encode_image(x)
            return v[0].detach().float().cpu().numpy().astype(np.float32), {"tokens_kind": "pooled"}
        if hasattr(core, "visual") and callable(getattr(core.visual, "forward", None)):
            v = core.visual(x)
            if v.ndim == 3:
                v = v.mean(dim=1)
            return v[0].detach().float().cpu().numpy().astype(np.float32), {"tokens_kind": "pooled"}

        raise ModelError("RemoteCLIP exposes neither token sequence nor pooled encoding methods.")


def _remoteclip_encode_pooled_batch(
    model,
    rgb_u8_batch: list[np.ndarray],
    *,
    image_size: int = 224,
    device: str = "auto",
) -> np.ndarray:
    """Fast pooled-only batch encoding. Returns [B,D]."""
    if not rgb_u8_batch:
        return np.zeros((0, 0), dtype=np.float32)

    import torch
    from PIL import Image
    from torchvision import transforms

    dev = (
        "cuda"
        if (device == "auto" and torch.cuda.is_available())
        else ("cpu" if device == "auto" else device)
    )
    model = model.to(dev).eval()
    core = getattr(model, "model", model)

    xs = []
    if hasattr(model, "transform") and callable(model.transform):
        for rgb_u8 in rgb_u8_batch:
            x = model.transform(rgb_u8.astype(np.float32), image_size)
            if not torch.is_tensor(x):
                raise ModelError("RemoteCLIP transform did not return torch.Tensor.")
            if x.ndim != 3:
                raise ModelError(
                    f"RemoteCLIP transform returned shape={tuple(x.shape)}; expected [C,H,W]."
                )
            xs.append(x)
    else:
        preprocess = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        for rgb_u8 in rgb_u8_batch:
            img = Image.fromarray(rgb_u8, mode="RGB")
            xs.append(preprocess(img))

    xb = torch.stack(xs, dim=0).to(dev)  # [B,3,H,W]
    with torch.inference_mode():
        if hasattr(core, "encode_image"):
            vec = core.encode_image(xb)
        elif hasattr(model, "encode_image"):
            vec = model.encode_image(xb)
        elif hasattr(core, "visual") and callable(getattr(core.visual, "forward", None)):
            vec = core.visual(xb)
            if vec.ndim == 3:
                vec = vec.mean(dim=1)
        else:
            raise ModelError("RemoteCLIP batch pooled path requires encode_image/visual.forward.")

    if vec.ndim != 2:
        raise ModelError(f"RemoteCLIP batch pooled expected [B,D], got {tuple(vec.shape)}")
    arr = vec.detach().float().cpu().numpy().astype(np.float32)
    if int(arr.shape[0]) != len(rgb_u8_batch):
        raise ModelError(
            f"RemoteCLIP batch mismatch: got B={arr.shape[0]}, expected {len(rgb_u8_batch)}"
        )
    return arr


@register("remoteclip")
class RemoteCLIPS2RGBEmbedder(EmbedderBase):
    """
    ROI -> (provider S2 SR Harmonized RGB composite) -> RemoteCLIP -> pooled or token-grid embedding

    - OutputSpec.pooled(): returns vec [D]
    - OutputSpec.grid(): returns token grid [D, Ht, Wt] (ViT patch grid, NOT pixel grid)
    """

    # RemoteCLIP needs a square input (Resize+CenterCrop would otherwise drop the
    # long edge of a rectangular ROI) → base.fetch_input enlarges the ROI to a
    # square of real imagery; the token grid is cropped back to the ROI.
    _requires_square_input = True
    DEFAULT_FETCH_WORKERS = 8
    DEFAULT_BATCH_CPU = 8
    DEFAULT_BATCH_CUDA = 64
    _allow_auto_backend = True

    input_spec = ModelInputSpec(
        collection="COPERNICUS/S2_SR_HARMONIZED",
        bands=("B4", "B3", "B2"),
        scale_m=10,
        cloudy_pct=30,
        image_size=224,
        expected_channels=3,
    )

    def describe(self) -> dict[str, Any]:
        return {
            "type": "on_the_fly",
            "backend": ["provider"],
            "inputs": {
                "collection": self.input_spec.collection,
                "bands": list(self.input_spec.bands),
            },
            "temporal": {"mode": "range"},
            "output": ["pooled", "grid"],
            "defaults": {
                "scale_m": self.input_spec.scale_m,
                "cloudy_pct": self.input_spec.cloudy_pct,
                "composite": self.input_spec.composite,
                "ckpt": "MVRL/remote-clip-vit-base-patch32",
                "image_size": self.input_spec.image_size,
            },
            "notes": "grid output is ViT token grid (patch-level), typically 7x7 for ViT-B/32 at 224px.",
        }

    def __init__(self) -> None:
        super().__init__()
        # key: (ckpt, cache_dir, resolved_device) -> (model, weight_meta)
        self._model_cache: dict[tuple[str, str, str], tuple[Any, dict[str, Any]]] = {}

    def _resolve_device(self, device: str) -> str:
        return resolve_device_auto_torch(device)

    def _get_model(
        self, *, ckpt: str, cache_dir: str | None, device: str
    ) -> tuple[Any, dict[str, Any], str]:
        dev = self._resolve_device(device)
        cache_dir_s = cache_dir or ""
        key = (ckpt, cache_dir_s, dev)
        if key in self._model_cache:
            m, wmeta = self._model_cache[key]
            return m, wmeta, dev

        model, wmeta = _load_rshf_remoteclip(
            ckpt,
            auto_download=True,
            require_pretrained=True,
            cache_dir=cache_dir,
        )
        try:
            model = model.to(dev).eval()
        except Exception as _e:
            pass
        self._model_cache[key] = (model, wmeta)
        return model, wmeta, dev

    @staticmethod
    def _resolve_fetch_workers(n_items: int) -> int:
        v = int(
            os.environ.get(
                "RS_EMBED_REMOTECLIP_FETCH_WORKERS",
                str(RemoteCLIPS2RGBEmbedder.DEFAULT_FETCH_WORKERS),
            )
        )
        return max(1, min(int(n_items), v))

    @staticmethod
    def _resolve_infer_batch(dev: str) -> int:
        default_bs = (
            RemoteCLIPS2RGBEmbedder.DEFAULT_BATCH_CUDA
            if str(dev).startswith("cuda")
            else RemoteCLIPS2RGBEmbedder.DEFAULT_BATCH_CPU
        )
        v = int(os.environ.get("RS_EMBED_REMOTECLIP_BATCH_SIZE", str(default_bs)))
        return max(1, v)

    def get_embedding(
        self,
        *,
        spatial: SpatialSpec,
        temporal: TemporalSpec | None,
        sensor: SensorSpec | None,
        output: OutputSpec,
        backend: str,
        device: str = "auto",
        input_chw: np.ndarray | None = None,
        fetch_meta: dict[str, Any] | None = None,
    ) -> Embedding:
        if not is_provider_backend(backend, allow_auto=True):
            raise ModelError("remoteclip_s2rgb expects a provider backend (or 'auto').")
        if temporal is None:
            raise ModelError("remoteclip_s2rgb requires TemporalSpec.range(start,end).")
        temporal.validate()
        if temporal.mode != "range":
            raise ModelError("remoteclip_s2rgb requires TemporalSpec.range in v0.1.")
        t = temporal_to_range(temporal)

        provider = self._get_provider(backend)

        # overrides via SensorSpec
        scale_m = sensor.scale_m if sensor else 10
        cloudy_pct = sensor.cloudy_pct if sensor else 30
        composite = sensor.composite if sensor else "median"

        ckpt = "MVRL/remote-clip-vit-base-patch32"
        # v0.1 convention: sensor.collection="hf:<repo_id_or_local_path>"
        if sensor and isinstance(sensor.collection, str) and sensor.collection.startswith("hf:"):
            ckpt = sensor.collection.replace("hf:", "", 1).strip()

        image_size = 224

        # Fetch-square ROI window: from the direct fetch, or carried in fetch_meta
        # when the API prefetched a square. The token grid is cropped back to it.
        geo_roi = geo_roi_from_meta(fetch_meta)

        # fetch image (optionally reuse pre-fetched raw patch)
        if input_chw is None:
            spatial, geo_roi = square_spatial(spatial)  # enlarge rectangle to square
            s2_rgb_chw = _fetch_s2_rgb_chw(
                provider,
                spatial,
                t,
                scale_m=scale_m,
                cloudy_pct=cloudy_pct,
                composite=composite,
            )
        else:
            # input_chw is expected to be raw S2 SR values in the order (B4,B3,B2)
            if input_chw.ndim != 3 or input_chw.shape[0] != 3:
                raise ModelError(
                    f"input_chw must be CHW with 3 bands for remoteclip_s2rgb, got {getattr(input_chw, 'shape', None)}"
                )
            # Keep raw S2 SR values here, exactly like the self-fetch path:
            # ``_s2_rgb_u8_from_chw`` is the single normalization step (/10000).
            # Pre-dividing here would double-normalize and yield a black image.
            s2_rgb_chw = np.asarray(input_chw, dtype=np.float32)

        # Optional: inspect on-the-fly provider input
        from ..tools.inspection import (
            checks_save_dir,
            checks_should_raise,
            maybe_inspect_chw,
            save_quicklook_rgb,
        )

        extra_checks: dict[str, Any] = {}
        report = maybe_inspect_chw(
            s2_rgb_chw,
            sensor=sensor,
            name="provider_s2_rgb_chw",
            expected_channels=3,
            # s2_rgb_chw holds raw S2 SR DN (~0..10000) on both the self-fetch
            # and input_chw paths, so the range check must match raw values.
            value_range=(0.0, 10000.0),
            fill_value=0.0,
            meta=extra_checks,
        )
        if report is not None and (not report.get("ok", True)) and checks_should_raise(sensor):
            raise ModelError(
                "Provider input inspection failed: " + "; ".join(report.get("issues", []))
            )
        sd = checks_save_dir(sensor)
        if sd and report is not None:
            try:
                import uuid

                fn = f"remoteclip_s2_rgb_{uuid.uuid4().hex[:8]}.png"
                save_quicklook_rgb(
                    s2_rgb_chw,
                    path=os.path.join(sd, fn),
                    bands=(0, 1, 2),
                    # s2_rgb_chw is raw S2 SR DN; stretch to its native range.
                    vmin=0.0,
                    vmax=10000.0,
                )
                extra_checks.setdefault("input_checks_artifacts", []).append(
                    {"name": "quicklook_rgb", "path": os.path.join(sd, fn)}
                )
            except Exception as _e:
                # Never fail embedding because quicklook saving failed.
                extra_checks.setdefault("input_checks_artifacts", []).append(
                    {"name": "quicklook_rgb", "error": repr(_e)}
                )

        rgb_u8 = _s2_rgb_u8_from_chw(s2_rgb_chw)

        # HF cache dir
        cache_dir = (
            os.environ.get("HUGGINGFACE_HUB_CACHE")
            or os.environ.get("HF_HOME")
            or os.environ.get("HUGGINGFACE_HOME")
        )

        # load model once per (ckpt, cache_dir, device)
        model, wmeta, dev = self._get_model(ckpt=ckpt, cache_dir=cache_dir, device=device)

        sensor_meta = {
            "collection": "COPERNICUS/S2_SR_HARMONIZED",
            "bands": ("B4", "B3", "B2"),
            "scale_m": scale_m,
            "cloudy_pct": cloudy_pct,
            "composite": composite,
        }

        def _build_base_meta(tmeta: dict[str, Any]) -> dict[str, Any]:
            extra = {
                "bands": sensor_meta["bands"],
                "scale_m": scale_m,
                "cloudy_pct": cloudy_pct,
                "composite": composite,
                "start": t.start,
                "end": t.end,
                "ckpt": ckpt,
                "device": dev,
                "pretrained_required": True,
                "auto_download": True,
                "hf_cache_dir": cache_dir,
                **wmeta,
                **tmeta,
                **extra_checks,
            }
            return build_meta(
                model=self.model_name,
                kind="on_the_fly",
                backend=str(backend).lower(),
                source="COPERNICUS/S2_SR_HARMONIZED",
                sensor=sensor_meta,
                temporal=t,
                image_size=image_size,
                extra=extra,
            )

        cropped_to_roi = not roi_is_full(geo_roi)

        # ---- pooled output ----
        if output.mode == "pooled":
            if cropped_to_roi:
                # The fetch enlarged a rectangle to a square; pool ONLY the ROI's
                # patch tokens instead of the global CLIP image embedding (which
                # would include the real-neighborhood context fetched to square it).
                tokens_or_vec, tmeta = _remoteclip_encode_tokens(
                    model, rgb_u8, image_size=image_size, device=dev
                )
                if tokens_or_vec.ndim == 2:
                    grid_dhw, gmeta = _tokens_to_grid_dhw(tokens_or_vec)
                    g = crop_grid_to_roi(grid_dhw, geo_roi)
                    reduce = g.max if output.pooling == "max" else g.mean
                    vec = reduce(axis=(1, 2)).astype(np.float32)
                    base_meta = _build_base_meta(
                        {**tmeta, **gmeta, "pooling": f"roi_grid_{output.pooling}"}
                    )
                    return Embedding(data=vec, meta=base_meta)
                # No token grid available (wrapper only exposes pooled vectors):
                # fall back to the global CLIP embedding (cannot crop a 1-D vector).
            # Use the projected CLIP image embedding (encode_image), identical to the
            # batch path, so pooled output is the canonical RemoteCLIP representation
            # with a consistent dimensionality regardless of the single/batch/tiled path.
            vec = _remoteclip_encode_pooled_batch(
                model, [rgb_u8], image_size=image_size, device=dev
            )[0].astype(np.float32)
            base_meta = _build_base_meta({"tokens_kind": "pooled"})
            return Embedding(data=vec, meta=base_meta)

        # ---- grid output ----
        if output.mode == "grid":
            tokens_or_vec, tmeta = _remoteclip_encode_tokens(
                model, rgb_u8, image_size=image_size, device=dev
            )
            base_meta = _build_base_meta(tmeta)
            if tokens_or_vec.ndim != 2:
                raise ModelError(
                    "grid output requires token sequence [N,D]. "
                    "Your RemoteCLIP wrapper only provides pooled vectors (no forward_encoder tokens)."
                )

            grid_dhw, gmeta = _tokens_to_grid_dhw(tokens_or_vec)
            if cropped_to_roi:
                grid_dhw = crop_grid_to_roi(grid_dhw, geo_roi)
                gmeta = {**gmeta, "grid_hw": (int(grid_dhw.shape[1]), int(grid_dhw.shape[2]))}
            meta = {
                **base_meta,
                **gmeta,
                "grid_type": "vit_tokens",
            }  # patch grid, not pixel grid

            da = xr.DataArray(
                grid_dhw,
                dims=("d", "y", "x"),
                coords={
                    "d": np.arange(grid_dhw.shape[0]),
                    "y": np.arange(grid_dhw.shape[1]),
                    "x": np.arange(grid_dhw.shape[2]),
                },
                name="embedding",
                attrs=meta,
            )
            return Embedding(data=da, meta=meta)

        raise ModelError(f"Unknown output mode: {output.mode}")

    def get_embeddings_batch(
        self,
        *,
        spatials: list[SpatialSpec],
        temporal: TemporalSpec | None = None,
        sensor: SensorSpec | None = None,
        output: OutputSpec = OutputSpec.pooled(),
        backend: str = "auto",
        device: str = "auto",
    ) -> list[Embedding]:
        if not spatials:
            return []
        if not is_provider_backend(backend, allow_auto=True):
            raise ModelError("remoteclip_s2rgb expects a provider backend (or 'auto').")
        if temporal is None:
            raise ModelError("remoteclip_s2rgb requires TemporalSpec.range(start,end).")
        temporal.validate()
        if temporal.mode != "range":
            raise ModelError("remoteclip_s2rgb requires TemporalSpec.range in v0.1.")

        t = temporal_to_range(temporal)
        provider = self._get_provider(backend)
        scale_m = sensor.scale_m if sensor else 10
        cloudy_pct = sensor.cloudy_pct if sensor else 30
        composite = sensor.composite if sensor else "median"

        # Square-fetch each ROI; the per-item ROI window rides in geo_rois and is
        # forwarded as _roi_windows_geo so each output is cropped back to it.
        raw_inputs, geo_rois = square_fetch_batch(
            spatials,
            lambda sq: np.clip(
                _fetch_s2_rgb_chw(
                    provider,
                    sq,
                    t,
                    scale_m=scale_m,
                    cloudy_pct=cloudy_pct,
                    composite=composite,
                ),
                0.0,
                10000.0,
            ).astype(np.float32),
            max_workers=self._resolve_fetch_workers(len(spatials)),
        )
        return self.get_embeddings_batch_from_inputs(
            spatials=spatials,
            input_chws=raw_inputs,
            temporal=temporal,
            sensor=sensor,
            output=output,
            backend=backend,
            device=device,
            _roi_windows_geo=geo_rois,
        )

    def get_embeddings_batch_from_inputs(
        self,
        *,
        spatials: list[SpatialSpec],
        input_chws: list[np.ndarray],
        temporal: TemporalSpec | None = None,
        sensor: SensorSpec | None = None,
        output: OutputSpec = OutputSpec.pooled(),
        backend: str = "auto",
        device: str = "auto",
        _roi_windows_geo: list[tuple[float, float, float, float] | None] | None = None,
    ) -> list[Embedding]:
        if not is_provider_backend(backend, allow_auto=True):
            raise ModelError("remoteclip_s2rgb expects a provider backend (or 'auto').")
        if temporal is None:
            raise ModelError("remoteclip_s2rgb requires TemporalSpec.range(start,end).")
        temporal.validate()
        if temporal.mode != "range":
            raise ModelError("remoteclip_s2rgb requires TemporalSpec.range in v0.1.")
        if len(spatials) != len(input_chws):
            raise ModelError(
                f"spatials/input_chws length mismatch: {len(spatials)} != {len(input_chws)}"
            )
        if not spatials:
            return []

        t = temporal_to_range(temporal)
        scale_m = sensor.scale_m if sensor else 10
        cloudy_pct = sensor.cloudy_pct if sensor else 30
        composite = sensor.composite if sensor else "median"
        image_size = 224

        ckpt = "MVRL/remote-clip-vit-base-patch32"
        if sensor and isinstance(sensor.collection, str) and sensor.collection.startswith("hf:"):
            ckpt = sensor.collection.replace("hf:", "", 1).strip()

        cache_dir = (
            os.environ.get("HUGGINGFACE_HUB_CACHE")
            or os.environ.get("HF_HOME")
            or os.environ.get("HUGGINGFACE_HOME")
        )
        model, wmeta, dev = self._get_model(ckpt=ckpt, cache_dir=cache_dir, device=device)
        infer_bs = self._resolve_infer_batch(str(dev))

        rgb_u8_all: list[np.ndarray] = []
        for i, input_chw in enumerate(input_chws):
            if input_chw.ndim != 3 or input_chw.shape[0] != 3:
                raise ModelError(
                    f"input_chw must be CHW with 3 bands for remoteclip_s2rgb, got {getattr(input_chw, 'shape', None)} at index={i}"
                )
            # Raw S2 SR values; ``_s2_rgb_u8_from_chw`` normalizes once (/10000).
            s2_rgb_chw = np.asarray(input_chw, dtype=np.float32)
            rgb_u8_all.append(_s2_rgb_u8_from_chw(s2_rgb_chw))

        sensor_meta = {
            "collection": "COPERNICUS/S2_SR_HARMONIZED",
            "bands": ("B4", "B3", "B2"),
            "scale_m": scale_m,
            "cloudy_pct": cloudy_pct,
            "composite": composite,
        }

        out: list[Embedding | None] = [None] * len(spatials)
        n = len(spatials)

        def _roi_at(i: int) -> tuple[float, float, float, float]:
            if _roi_windows_geo is None:
                return geo_roi_from_meta(None)
            return geo_roi_from_meta({"roi_window_geo": _roi_windows_geo[i]})

        def _base_meta_for(extra_kind: dict[str, Any]) -> dict[str, Any]:
            extra = {
                "bands": sensor_meta["bands"],
                "scale_m": scale_m,
                "cloudy_pct": cloudy_pct,
                "composite": composite,
                "start": t.start,
                "end": t.end,
                "ckpt": ckpt,
                "device": dev,
                "pretrained_required": True,
                "auto_download": True,
                "hf_cache_dir": cache_dir,
                "input_override": True,
                **wmeta,
                **extra_kind,
            }
            return build_meta(
                model=self.model_name,
                kind="on_the_fly",
                backend=str(backend).lower(),
                source="COPERNICUS/S2_SR_HARMONIZED",
                sensor=sensor_meta,
                temporal=t,
                image_size=image_size,
                extra=extra,
            )

        if output.mode == "pooled":
            for s0 in range(0, n, infer_bs):
                s1 = min(n, s0 + infer_bs)
                vecs = _remoteclip_encode_pooled_batch(
                    model,
                    rgb_u8_all[s0:s1],
                    image_size=image_size,
                    device=dev,
                )  # [B,D]
                for j in range(s1 - s0):
                    i = s0 + j
                    geo_roi = _roi_at(i)
                    if not roi_is_full(geo_roi):
                        # Pool only the ROI's patch tokens (see get_embedding).
                        tokens_or_vec, tmeta = _remoteclip_encode_tokens(
                            model, rgb_u8_all[i], image_size=image_size, device=dev
                        )
                        if tokens_or_vec.ndim == 2:
                            grid_dhw, gmeta = _tokens_to_grid_dhw(tokens_or_vec)
                            g = crop_grid_to_roi(grid_dhw, geo_roi)
                            reduce = g.max if output.pooling == "max" else g.mean
                            vec = reduce(axis=(1, 2)).astype(np.float32)
                            meta = _base_meta_for(
                                {
                                    **tmeta,
                                    **gmeta,
                                    "pooling": f"roi_grid_{output.pooling}",
                                    "batch_infer": False,
                                }
                            )
                            out[i] = Embedding(data=vec, meta=meta)
                            continue
                        # No token grid: fall back to the global pooled vector below.
                    meta = _base_meta_for({"tokens_kind": "pooled_batch", "batch_infer": True})
                    out[i] = Embedding(data=vecs[j].astype(np.float32), meta=meta)
        elif output.mode == "grid":
            for i in range(n):
                geo_roi = _roi_at(i)
                tokens_or_vec, tmeta = _remoteclip_encode_tokens(
                    model,
                    rgb_u8_all[i],
                    image_size=image_size,
                    device=dev,
                )
                if tokens_or_vec.ndim != 2:
                    raise ModelError(
                        "grid output requires token sequence [N,D]. "
                        "Your RemoteCLIP wrapper only provides pooled vectors (no forward_encoder tokens)."
                    )
                grid_dhw, gmeta = _tokens_to_grid_dhw(tokens_or_vec)
                if not roi_is_full(geo_roi):
                    grid_dhw = crop_grid_to_roi(grid_dhw, geo_roi)
                    gmeta = {**gmeta, "grid_hw": (int(grid_dhw.shape[1]), int(grid_dhw.shape[2]))}
                meta = _base_meta_for(
                    {**tmeta, **gmeta, "batch_infer": False, "grid_type": "vit_tokens"}
                )
                da = xr.DataArray(
                    grid_dhw,
                    dims=("d", "y", "x"),
                    coords={
                        "d": np.arange(grid_dhw.shape[0]),
                        "y": np.arange(grid_dhw.shape[1]),
                        "x": np.arange(grid_dhw.shape[2]),
                    },
                    name="embedding",
                    attrs=meta,
                )
                out[i] = Embedding(data=da, meta=meta)
        else:
            raise ModelError(f"Unknown output mode: {output.mode}")

        if any(e is None for e in out):
            raise ModelError("remoteclip_s2rgb batch inference produced incomplete outputs.")
        return [e for e in out if e is not None]

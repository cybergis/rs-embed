# src/rs_embed/embedders/onthefly_remoteclip.py
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

from ..core.registry import register
from ..core.embedding import Embedding
from ..core.errors import ModelError
from ..core.specs import SpatialSpec, TemporalSpec, SensorSpec, OutputSpec
from ..providers import ProviderBase
from .base import EmbedderBase
from .meta_utils import build_meta, temporal_to_range, temporal_midpoint_str
from .runtime_utils import (
    fetch_s2_rgb_chw as _fetch_s2_rgb_chw_shared,
    is_provider_backend,
    resolve_device_auto_torch,
)


# -----------------------------
# Provider: Fetch S2 RGB
# -----------------------------
from ._vit_mae_utils import _s2_rgb_u8_from_chw


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
def _find_weight_file(path: str) -> Optional[str]:
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
    cache_dir: Optional[str] = None,
    min_bytes: int = 50
    * 1024
    * 1024,  # 50MB: below this is almost surely pointer/metadata
) -> Tuple[str, Optional[str]]:
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


def _assert_weights_loaded(model) -> Dict[str, float]:
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
    suppressed: Dict[str, Any] = {"count": 0, "messages": []}

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
    cache_dir: Optional[str] = None,
) -> Tuple[Any, Dict[str, Any]]:
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
        model = RemoteCLIP.from_pretrained(
            local_dir if os.path.exists(local_dir) else ckpt
        )
    stats = _assert_weights_loaded(model)

    meta = {
        "ckpt_input": ckpt,
        "ckpt_local_dir": local_dir,
        "weight_file": weight_file,
        "weight_file_size": (
            os.path.getsize(weight_file)
            if (weight_file and os.path.exists(weight_file))
            else None
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


def _tokens_to_grid_dhw(tokens_nd: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
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


# -----------------------------
# RemoteCLIP inference adapter
# -----------------------------
def _remoteclip_encode_tokens(
    model,
    rgb_u8: np.ndarray,
    *,
    image_size: int = 224,
    device: str = "auto",
) -> Tuple[np.ndarray, Dict[str, Any]]:
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
    from torchvision import transforms
    from PIL import Image

    dev = (
        "cuda"
        if (device == "auto" and torch.cuda.is_available())
        else ("cpu" if device == "auto" else device)
    )
    model = model.to(dev).eval()
    core = getattr(model, "model", model)

    # --- preprocess to tensor ---
    if hasattr(model, "transform") and callable(getattr(model, "transform")):
        x = model.transform(rgb_u8.astype(np.float32), image_size).unsqueeze(0)
    else:
        preprocess = transforms.Compose(
            [
                transforms.Resize(
                    image_size, interpolation=transforms.InterpolationMode.BICUBIC
                ),
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
            raise ModelError(
                f"Unexpected forward_encoder output shape: {tuple(toks.shape)}"
            )

        # 2) open_clip: forward_intermediates (best if it returns tokens cleanly)
        if hasattr(core, "forward_intermediates"):
            try:
                out = core.forward_intermediates(x)
                # open_clip versions differ: out may be dict-like or tuple-like.
                # We'll search for the first tensor shaped [B,N,D] as tokens.
                tokens_t = None

                if isinstance(out, dict):
                    # common keys (varies by version)
                    candidates = []
                    for k in (
                        "image_intermediates",
                        "intermediates",
                        "image_tokens",
                        "tokens",
                    ):
                        if k in out:
                            candidates.append(out[k])
                    # also scan values
                    candidates += list(out.values())
                    for v in candidates:
                        if torch.is_tensor(v) and v.ndim == 3:
                            tokens_t = v
                            break
                        if isinstance(v, (list, tuple)):
                            for vv in v:
                                if torch.is_tensor(vv) and vv.ndim == 3:
                                    tokens_t = vv
                                    break
                            if tokens_t is not None:
                                break

                elif isinstance(out, (tuple, list)):
                    # scan elements
                    for v in out:
                        if torch.is_tensor(v) and v.ndim == 3:
                            tokens_t = v
                            break
                        if isinstance(v, (list, tuple)):
                            for vv in v:
                                if torch.is_tensor(vv) and vv.ndim == 3:
                                    tokens_t = vv
                                    break
                            if tokens_t is not None:
                                break

                if tokens_t is not None:
                    return tokens_t[0].detach().float().cpu().numpy().astype(
                        np.float32
                    ), {"tokens_kind": "tokens_intermediates"}
            except Exception:
                # If forward_intermediates exists but signature/return differs, fall back to hook
                pass

        # 3) hook vision transformer transformer output to get tokens
        if hasattr(core, "visual") and hasattr(core.visual, "transformer"):
            captured = {}

            def _hook(_module, _inp, outp):
                # outp is typically [B, N, D]
                captured["tokens"] = outp

            handle = core.visual.transformer.register_forward_hook(_hook)
            try:
                # run a normal encode_image forward; hook captures tokens
                _ = (
                    core.encode_image(x)
                    if hasattr(core, "encode_image")
                    else core.forward(x)
                )
            finally:
                handle.remove()

            if (
                "tokens" in captured
                and torch.is_tensor(captured["tokens"])
                and captured["tokens"].ndim == 3
            ):
                toks = captured["tokens"]
                return toks[0].detach().float().cpu().numpy().astype(np.float32), {
                    "tokens_kind": "tokens_hook"
                }

        # 4) pooled fallback only
        if hasattr(core, "encode_image"):
            v = core.encode_image(x)
            return v[0].detach().float().cpu().numpy().astype(np.float32), {
                "tokens_kind": "pooled"
            }
        if hasattr(core, "visual") and callable(getattr(core.visual, "forward", None)):
            v = core.visual(x)
            if v.ndim == 3:
                v = v.mean(dim=1)
            return v[0].detach().float().cpu().numpy().astype(np.float32), {
                "tokens_kind": "pooled"
            }

        raise ModelError(
            "RemoteCLIP exposes neither token sequence nor pooled encoding methods."
        )


def _remoteclip_encode_pooled_batch(
    model,
    rgb_u8_batch: List[np.ndarray],
    *,
    image_size: int = 224,
    device: str = "auto",
) -> np.ndarray:
    """Fast pooled-only batch encoding. Returns [B,D]."""
    if not rgb_u8_batch:
        return np.zeros((0, 0), dtype=np.float32)

    import torch
    from torchvision import transforms
    from PIL import Image

    dev = (
        "cuda"
        if (device == "auto" and torch.cuda.is_available())
        else ("cpu" if device == "auto" else device)
    )
    model = model.to(dev).eval()
    core = getattr(model, "model", model)

    xs = []
    if hasattr(model, "transform") and callable(getattr(model, "transform")):
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
                transforms.Resize(
                    image_size, interpolation=transforms.InterpolationMode.BICUBIC
                ),
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
        elif hasattr(core, "visual") and callable(
            getattr(core.visual, "forward", None)
        ):
            vec = core.visual(xb)
            if vec.ndim == 3:
                vec = vec.mean(dim=1)
        else:
            raise ModelError(
                "RemoteCLIP batch pooled path requires encode_image/visual.forward."
            )

    if vec.ndim != 2:
        raise ModelError(
            f"RemoteCLIP batch pooled expected [B,D], got {tuple(vec.shape)}"
        )
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

    DEFAULT_FETCH_WORKERS = 8
    DEFAULT_BATCH_CPU = 8
    DEFAULT_BATCH_CUDA = 64
    _allow_auto_backend = False

    def describe(self) -> Dict[str, Any]:
        return {
            "type": "on_the_fly",
            "backend": ["provider"],
            "inputs": {
                "collection": "COPERNICUS/S2_SR_HARMONIZED",
                "bands": ["B4", "B3", "B2"],
            },
            "temporal": {"mode": "range"},
            "output": ["pooled", "grid"],
            "defaults": {
                "scale_m": 10,
                "cloudy_pct": 30,
                "composite": "median",
                "ckpt": "MVRL/remote-clip-vit-base-patch32",
                "image_size": 224,
            },
            "notes": "grid output is ViT token grid (patch-level), typically 7x7 for ViT-B/32 at 224px.",
        }

    def __init__(self) -> None:
        super().__init__()
        # key: (ckpt, cache_dir, resolved_device) -> (model, weight_meta)
        self._model_cache: Dict[Tuple[str, str, str], Tuple[Any, Dict[str, Any]]] = {}

    def _resolve_device(self, device: str) -> str:
        return resolve_device_auto_torch(device)

    def _get_model(
        self, *, ckpt: str, cache_dir: Optional[str], device: str
    ) -> Tuple[Any, Dict[str, Any], str]:
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
        except Exception:
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
        temporal: Optional[TemporalSpec],
        sensor: Optional[SensorSpec],
        output: OutputSpec,
        backend: str,
        device: str = "auto",
        input_chw: Optional[np.ndarray] = None,
    ) -> Embedding:
        if not is_provider_backend(backend, allow_auto=False):
            raise ModelError(
                "remoteclip_s2rgb only supports a provider backend in v0.1."
            )
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
        if (
            sensor
            and isinstance(sensor.collection, str)
            and sensor.collection.startswith("hf:")
        ):
            ckpt = sensor.collection.replace("hf:", "", 1).strip()

        image_size = 224

        # fetch image (optionally reuse pre-fetched raw patch)
        if input_chw is None:
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
            s2_rgb_chw = np.clip(input_chw.astype(np.float32) / 10000.0, 0.0, 1.0)

        # Optional: inspect on-the-fly provider input
        from ..core.input_checks import (
            maybe_inspect_chw,
            checks_save_dir,
            checks_should_raise,
            save_quicklook_rgb,
        )

        extra_checks: Dict[str, Any] = {}
        report = maybe_inspect_chw(
            s2_rgb_chw,
            sensor=sensor,
            name="provider_s2_rgb_chw",
            expected_channels=3,
            value_range=(0.0, 1.0),
            fill_value=0.0,
            meta=extra_checks,
        )
        if (
            report is not None
            and (not report.get("ok", True))
            and checks_should_raise(sensor)
        ):
            raise ModelError(
                "Provider input inspection failed: "
                + "; ".join(report.get("issues", []))
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
                    vmin=0.0,
                    vmax=1.0,
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
        model, wmeta, dev = self._get_model(
            ckpt=ckpt, cache_dir=cache_dir, device=device
        )

        tokens_or_vec, tmeta = _remoteclip_encode_tokens(
            model, rgb_u8, image_size=image_size, device=dev
        )

        sensor_meta = {
            "collection": "COPERNICUS/S2_SR_HARMONIZED",
            "bands": ("B4", "B3", "B2"),
            "scale_m": scale_m,
            "cloudy_pct": cloudy_pct,
            "composite": composite,
        }

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

        base_meta = build_meta(
            model=self.model_name,
            kind="on_the_fly",
            backend=str(backend).lower(),
            source="COPERNICUS/S2_SR_HARMONIZED",
            sensor=sensor_meta,
            temporal=t,
            image_size=image_size,
            input_time=temporal_midpoint_str(t),
            extra=extra,
        )

        # ---- pooled output ----
        if output.mode == "pooled":
            if tokens_or_vec.ndim == 1:
                vec = tokens_or_vec.astype(np.float32)
            elif tokens_or_vec.ndim == 2:
                vec = tokens_or_vec.mean(axis=0).astype(np.float32)  # tokens mean
                base_meta["pooling"] = "token_mean"
            else:
                raise ModelError(
                    f"Unexpected tokens/vec shape for pooled: {tokens_or_vec.shape}"
                )
            return Embedding(data=vec, meta=base_meta)

        # ---- grid output ----
        if output.mode == "grid":
            if tokens_or_vec.ndim != 2:
                raise ModelError(
                    "grid output requires token sequence [N,D]. "
                    "Your RemoteCLIP wrapper only provides pooled vectors (no forward_encoder tokens)."
                )

            grid_dhw, gmeta = _tokens_to_grid_dhw(tokens_or_vec)
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
        temporal: Optional[TemporalSpec] = None,
        sensor: Optional[SensorSpec] = None,
        output: OutputSpec = OutputSpec.pooled(),
        backend: str = "auto",
        device: str = "auto",
    ) -> list[Embedding]:
        if not spatials:
            return []
        if not is_provider_backend(backend, allow_auto=False):
            raise ModelError(
                "remoteclip_s2rgb only supports a provider backend in v0.1."
            )
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

        n = len(spatials)
        prefetched_raw: List[Optional[np.ndarray]] = [None] * n

        def _fetch_one(i: int, sp: SpatialSpec) -> Tuple[int, np.ndarray]:
            s2_rgb_chw = _fetch_s2_rgb_chw(
                provider,
                sp,
                t,
                scale_m=scale_m,
                cloudy_pct=cloudy_pct,
                composite=composite,
            )
            # get_embedding(input_chw=...) expects raw SR in [0..10000]
            raw = np.clip(s2_rgb_chw * 10000.0, 0.0, 10000.0).astype(np.float32)
            return i, raw

        mw = self._resolve_fetch_workers(n)
        if mw == 1:
            for i, sp in enumerate(spatials):
                ii, raw = _fetch_one(i, sp)
                prefetched_raw[ii] = raw
        else:
            with ThreadPoolExecutor(max_workers=mw) as ex:
                futs = [ex.submit(_fetch_one, i, sp) for i, sp in enumerate(spatials)]
                for fut in as_completed(futs):
                    i, raw = fut.result()
                    prefetched_raw[i] = raw

        raw_inputs: List[np.ndarray] = []
        for i, raw in enumerate(prefetched_raw):
            if raw is None:
                raise ModelError(
                    f"Missing prefetched input at index={i} for remoteclip_s2rgb."
                )
            raw_inputs.append(raw)
        return self.get_embeddings_batch_from_inputs(
            spatials=spatials,
            input_chws=raw_inputs,
            temporal=temporal,
            sensor=sensor,
            output=output,
            backend=backend,
            device=device,
        )

    def get_embeddings_batch_from_inputs(
        self,
        *,
        spatials: list[SpatialSpec],
        input_chws: list[np.ndarray],
        temporal: Optional[TemporalSpec] = None,
        sensor: Optional[SensorSpec] = None,
        output: OutputSpec = OutputSpec.pooled(),
        backend: str = "auto",
        device: str = "auto",
    ) -> list[Embedding]:
        if not is_provider_backend(backend, allow_auto=False):
            raise ModelError(
                "remoteclip_s2rgb only supports a provider backend in v0.1."
            )
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
        if (
            sensor
            and isinstance(sensor.collection, str)
            and sensor.collection.startswith("hf:")
        ):
            ckpt = sensor.collection.replace("hf:", "", 1).strip()

        cache_dir = (
            os.environ.get("HUGGINGFACE_HUB_CACHE")
            or os.environ.get("HF_HOME")
            or os.environ.get("HUGGINGFACE_HOME")
        )
        model, wmeta, dev = self._get_model(
            ckpt=ckpt, cache_dir=cache_dir, device=device
        )
        infer_bs = self._resolve_infer_batch(str(dev))

        rgb_u8_all: List[np.ndarray] = []
        for i, input_chw in enumerate(input_chws):
            if input_chw.ndim != 3 or input_chw.shape[0] != 3:
                raise ModelError(
                    f"input_chw must be CHW with 3 bands for remoteclip_s2rgb, got {getattr(input_chw, 'shape', None)} at index={i}"
                )
            s2_rgb_chw = np.clip(input_chw.astype(np.float32) / 10000.0, 0.0, 1.0)
            rgb_u8_all.append(_s2_rgb_u8_from_chw(s2_rgb_chw))

        sensor_meta = {
            "collection": "COPERNICUS/S2_SR_HARMONIZED",
            "bands": ("B4", "B3", "B2"),
            "scale_m": scale_m,
            "cloudy_pct": cloudy_pct,
            "composite": composite,
        }

        out: List[Optional[Embedding]] = [None] * len(spatials)
        n = len(spatials)

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
                        "tokens_kind": "pooled_batch",
                        "batch_infer": True,
                        "input_override": True,
                        **wmeta,
                    }
                    meta = build_meta(
                        model=self.model_name,
                        kind="on_the_fly",
                        backend=str(backend).lower(),
                        source="COPERNICUS/S2_SR_HARMONIZED",
                        sensor=sensor_meta,
                        temporal=t,
                        image_size=image_size,
                        input_time=temporal_midpoint_str(t),
                        extra=extra,
                    )
                    out[i] = Embedding(data=vecs[j].astype(np.float32), meta=meta)
        elif output.mode == "grid":
            for i in range(n):
                tokens_or_vec, tmeta = _remoteclip_encode_tokens(
                    model,
                    rgb_u8_all[i],
                    image_size=image_size,
                    device=dev,
                )
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
                    "batch_infer": False,
                    "input_override": True,
                    **wmeta,
                    **tmeta,
                }
                base_meta = build_meta(
                    model=self.model_name,
                    kind="on_the_fly",
                    backend=str(backend).lower(),
                    source="COPERNICUS/S2_SR_HARMONIZED",
                    sensor=sensor_meta,
                    temporal=t,
                    image_size=image_size,
                    input_time=temporal_midpoint_str(t),
                    extra=extra,
                )
                if tokens_or_vec.ndim != 2:
                    raise ModelError(
                        "grid output requires token sequence [N,D]. "
                        "Your RemoteCLIP wrapper only provides pooled vectors (no forward_encoder tokens)."
                    )
                grid_dhw, gmeta = _tokens_to_grid_dhw(tokens_or_vec)
                meta = {**base_meta, **gmeta, "grid_type": "vit_tokens"}
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
            raise ModelError(
                "remoteclip_s2rgb batch inference produced incomplete outputs."
            )
        return [e for e in out if e is not None]

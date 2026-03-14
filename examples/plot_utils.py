import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json
from pathlib import Path


def _to_dhw(arr):
    if hasattr(arr, "values"):  # xarray
        arr = arr.values
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr[None, ...].astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D/3D array, got {arr.shape}")
    # HWD -> DHW if last dim looks like D
    if arr.shape[-1] in (32, 64, 128, 256, 512, 768, 1024) and arr.shape[0] not in (
        32,
        64,
        128,
        256,
        512,
        768,
        1024,
    ):
        arr = np.moveaxis(arr, -1, 0)
    return arr.astype(np.float32)


def _robust_scale01(x, lo=2.0, hi=98.0, eps=1e-8):
    """Scale array to [0,1] with percentile clipping."""
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).any():
        return np.zeros_like(x, dtype=np.float32)
    a = np.nanpercentile(x, lo)
    b = np.nanpercentile(x, hi)
    y = np.clip((x - a) / (b - a + eps), 0.0, 1.0)
    y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
    return y


def _stabilize_pca_sign(components: np.ndarray) -> np.ndarray:
    """
    Make PCA component signs deterministic.
    For each component row, enforce the element with max abs value to be positive.
    """
    comps = np.asarray(components, dtype=np.float32).copy()
    for i in range(comps.shape[0]):
        row = comps[i]
        if row.size == 0:
            continue
        j = int(np.argmax(np.abs(row)))
        if row[j] < 0:
            comps[i] = -row
    return comps


def fit_pca_rgb(
    emb,
    *,
    n_samples=100_000,
    seed=0,
    center=True,
):
    """
    Fit PCA on pixels of a (D,H,W) grid and return a dict with components for reuse.
    No sklearn dependency (uses SVD).
    """
    data = getattr(emb, "data", emb)
    dhw = _to_dhw(data)
    D, H, W = dhw.shape

    X = dhw.reshape(D, H * W).T  # [N, D]
    finite_rows = np.all(np.isfinite(X), axis=1)
    X = X[finite_rows]
    N = X.shape[0]

    if N == 0:
        raise ValueError("No finite pixels available for PCA fit.")

    rng = np.random.default_rng(seed)
    if n_samples is not None and N > n_samples:
        idx = rng.choice(N, size=int(n_samples), replace=False)
        Xs = X[idx]
    else:
        Xs = X

    # center
    mean = Xs.mean(axis=0) if center else np.zeros((D,), dtype=np.float32)
    Xc = Xs - mean

    if Xc.shape[0] < 2 or np.allclose(np.nanstd(Xc, axis=0), 0.0):
        comps = np.eye(D, dtype=np.float32)[:3]
    else:
        # SVD for PCA
        # Xc = U S Vt, rows are samples
        # PCs are rows of Vt
        try:
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            comps = Vt[:3].astype(np.float32)
        except np.linalg.LinAlgError:
            # Fallback for numerically unstable SVD cases.
            cov = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
            vals, vecs = np.linalg.eigh(cov)
            order = np.argsort(vals)[::-1]
            comps = vecs[:, order[:3]].T.astype(np.float32)
        comps = _stabilize_pca_sign(comps)

    return {
        "mean": mean.astype(np.float32),
        "components": comps,
        "center": bool(center),
    }


def transform_pca_rgb(
    emb,
    pca,
    *,
    robust_lo=2.0,
    robust_hi=98.0,
):
    """
    Apply fitted PCA to emb, return rgb in [0,1] as (H,W,3).
    """
    data = getattr(emb, "data", emb)
    dhw = _to_dhw(data)
    D, H, W = dhw.shape

    X = dhw.reshape(D, H * W).T  # [N, D]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mean = pca["mean"]
    comps = pca["components"]  # [3,D]

    Xc = X - mean if pca.get("center", True) else X
    Y = Xc @ comps.T  # [N,3]
    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

    # robust scale each channel to [0,1]
    rgb = np.zeros_like(Y, dtype=np.float32)
    for k in range(3):
        rgb[:, k] = _robust_scale01(Y[:, k], lo=robust_lo, hi=robust_hi)

    return rgb.reshape(H, W, 3)


def plot_embedding_pseudocolor(
    emb,
    emb2=None,
    *,
    title=None,
    title2=None,
    pca=None,
    n_samples=100_000,
    seed=0,
    robust_lo=2.0,
    robust_hi=98.0,
    figsize=(6, 5),
    show_colorbars=False,
):
    """
    Plot one or two PCA pseudocolor images.

    If `emb2` is provided, the two images are rendered side by side.
    When `pca` is provided, it is reused for all plots; otherwise each plot
    fits its own PCA independently.

    Returns fitted PCA for reuse across images. For two plots, returns a list.
    """
    embeddings = [emb] if emb2 is None else [emb, emb2]
    default_titles = [
        getattr(item, "meta", {}).get("model", "embedding PCA") for item in embeddings
    ]
    titles = default_titles.copy()
    if title is not None:
        titles[0] = title
    if emb2 is not None and title2 is not None:
        titles[1] = title2

    if pca is None:
        pcas = [fit_pca_rgb(item, n_samples=n_samples, seed=seed) for item in embeddings]
    elif isinstance(pca, (list, tuple)):
        if len(pca) != len(embeddings):
            raise ValueError("Length of pca must match number of embeddings.")
        pcas = list(pca)
    else:
        pcas = [pca] * len(embeddings)

    rgbs = [
        transform_pca_rgb(item, item_pca, robust_lo=robust_lo, robust_hi=robust_hi)
        for item, item_pca in zip(embeddings, pcas)
    ]

    width_scale = len(embeddings)
    fig, axes = plt.subplots(
        1,
        len(embeddings),
        figsize=(figsize[0] * width_scale, figsize[1]),
        squeeze=False,
    )
    for ax, rgb, item_title in zip(axes[0], rgbs, titles):
        ax.imshow(rgb)
        ax.set_title(item_title)
        ax.axis("off")

    fig.tight_layout()
    output_stem = "_vs_".join(item_title.replace(" ", "_") for item_title in titles)
    fig.savefig(f"{output_stem}_pca.png")
    plt.show()

    if show_colorbars:
        for item, item_pca, item_title in zip(embeddings, pcas, titles):
            data = getattr(item, "data", item)
            dhw = _to_dhw(data)
            D, H, W = dhw.shape
            X = dhw.reshape(D, H * W).T
            Xc = X - item_pca["mean"]
            Y = Xc @ item_pca["components"].T
            for k in range(3):
                img = _robust_scale01(Y[:, k], lo=robust_lo, hi=robust_hi).reshape(H, W)
                plt.figure(figsize=figsize)
                plt.imshow(img)
                plt.title(f"{item_title} | PC{k + 1}")
                plt.axis("off")
                plt.show()

    return pcas[0] if len(pcas) == 1 else pcas


def percentile_stretch(rgb_hwc: np.ndarray, p_low=1.0, p_high=99.0, gamma=1.0) -> np.ndarray:
    """Per-channel percentile stretch to [0,1]."""
    rgb = rgb_hwc.astype(np.float32)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)

    out = np.empty_like(rgb)
    for c in range(3):
        lo = np.percentile(rgb[..., c], p_low)
        hi = np.percentile(rgb[..., c], p_high)
        if hi <= lo + 1e-6:
            out[..., c] = 0.0
        else:
            out[..., c] = (rgb[..., c] - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    if gamma != 1.0:
        out = out ** (1.0 / gamma)
    return out


def show_input_chw(x_chw: np.ndarray, title: str, rgb_idx=(0, 1, 2), p_low=1, p_high=99):
    """Visualize CHW input. If C>=3 show RGB via indices; else show grayscale."""
    if x_chw.ndim != 3:
        print(f"Skip {title}: expected CHW, got shape={x_chw.shape}")
        return

    c, h, w = x_chw.shape
    print(
        f"{title:40s} shape={x_chw.shape} dtype={x_chw.dtype} min={x_chw.min():.3g} max={x_chw.max():.3g}"
    )

    plt.figure(figsize=(6, 6))
    if c >= 3:
        r, g, b = rgb_idx
        if max(rgb_idx) >= c:
            raise ValueError(f"rgb_idx={rgb_idx} out of range for C={c}")

        rgb = np.stack([x_chw[r], x_chw[g], x_chw[b]], axis=-1)  # HWC
        rgb = percentile_stretch(rgb, p_low=p_low, p_high=p_high, gamma=1.0)
        plt.imshow(rgb)
    else:
        plt.imshow(x_chw[0], cmap="gray")

    plt.title(f"{title}  (C={c}, H={h}, W={w})")
    plt.axis("off")
    plt.show()


def show_s1_vvvh_chw(
    x_chw: np.ndarray,
    *,
    band_names=("VV", "VH"),
    title_prefix="S1",
    p_low=2.0,
    p_high=98.0,
    figsize=(10, 4),
    cmap="gray",
    flipud: bool = True,
):
    """Visualize a Sentinel-1 VV/VH CHW patch as two stretched grayscale panels."""
    x = np.asarray(x_chw)
    if x.ndim != 3:
        raise ValueError(f"Expected CHW array, got shape={getattr(x, 'shape', None)}")
    if int(x.shape[0]) < 2:
        raise ValueError(f"Expected at least 2 channels for VV/VH, got C={int(x.shape[0])}")

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, band_img, band_name in zip(axes, x[:2], tuple(band_names)[:2]):
        band_img = np.asarray(band_img, dtype=np.float32)
        band_img = np.nan_to_num(band_img, nan=0.0, posinf=0.0, neginf=0.0)
        lo, hi = np.percentile(band_img, [p_low, p_high])
        img = np.clip((band_img - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        if bool(flipud):
            img = np.flipud(img)
        ax.imshow(img, cmap=cmap)
        ax.set_title(f"{title_prefix} {band_name}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def show_s1_vvvh_from_inspect(
    inspect_out: dict,
    *,
    title_prefix="S1",
    p_low=2.0,
    p_high=98.0,
    figsize=(10, 4),
    print_stats=True,
    flipud: bool = False,
):
    """Visualize Sentinel-1 VV/VH from inspect_gee_patch(..., return_array=True) output."""
    x_s1 = (inspect_out or {}).get("array_chw")
    if x_s1 is None:
        print("array_chw not found. Call inspect_gee_patch(..., return_array=True).")
        return

    if print_stats:
        sensor_meta = (inspect_out or {}).get("sensor") or {}
        print("ok:", (inspect_out or {}).get("ok"))
        print("bands:", sensor_meta.get("bands"))
        x = np.asarray(x_s1)
        print(
            "S1 CHW:",
            x.shape,
            x.dtype,
            "min=",
            float(np.nanmin(x)),
            "max=",
            float(np.nanmax(x)),
        )

    show_s1_vvvh_chw(
        x_s1,
        title_prefix=title_prefix,
        p_low=p_low,
        p_high=p_high,
        figsize=figsize,
        flipud=flipud,
    )


def print_band_quantiles_preview(report: dict, n_preview: int = 3):
    """Print a compact preview for report['band_quantiles']."""
    band_q = (report or {}).get("band_quantiles")
    if not band_q:
        print("band_quantiles not available")
        return
    preview = {k: v[:n_preview] for k, v in band_q.items()}
    print(f"band_quantiles (first {n_preview} bins):")
    print(preview)


def plot_histogram_from_report(report: dict, band_index: int = 0, figsize=(6, 3)):
    """Plot histogram for one band from inspect_gee_patch report."""
    hist = (report or {}).get("hist")
    if not hist or not hist.get("bins") or not hist.get("counts"):
        print("histogram not available")
        return

    bins = hist["bins"]
    counts = hist["counts"][band_index]
    plt.figure(figsize=figsize)
    plt.bar(bins[:-1], counts, width=bins[1] - bins[0], align="edge")
    plt.title(f"Band {band_index} histogram")
    plt.xlabel("DN")
    plt.ylabel("Count")
    plt.show()


def show_quicklook_artifact(
    artifacts: dict,
    *,
    flipud: bool = False,
    figsize=(5, 5),
    title: str = "quicklook_rgb",
):
    """Display quicklook image from inspect_gee_patch artifacts."""
    quicklook_path = (artifacts or {}).get("quicklook_rgb")
    if not quicklook_path:
        print("quicklook not saved; artifacts:", artifacts)
        return
    try:
        from pathlib import Path
    except Exception:
        print("pathlib is not available")
        return

    if Path(quicklook_path).exists():
        try:
            img = plt.imread(quicklook_path)
        except Exception as e:
            print(f"failed to read quicklook image: {e!r}")
            print("quicklook path:", quicklook_path)
            return

        if bool(flipud):
            img = np.flipud(img)

        plt.figure(figsize=figsize)
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")
        plt.show()
    else:
        print("quicklook not found on disk:", quicklook_path)


def infer_rgb_idx_from_sensor(sensor_meta, channels: int):
    """Infer RGB channel indices from sensor metadata when possible."""
    if channels < 3:
        return None

    bands = []
    if isinstance(sensor_meta, dict):
        bands = [str(b).upper() for b in (sensor_meta.get("bands") or [])]

    if bands:
        idx = {b: i for i, b in enumerate(bands)}
        if all(k in idx for k in ("B4", "B3", "B2")):
            return (idx["B4"], idx["B3"], idx["B2"])
        if all(k in idx for k in ("RED", "GREEN", "BLUE")):
            return (idx["RED"], idx["GREEN"], idx["BLUE"])

    return (0, 1, 2)


def visualize_manifest_inputs(manifest: dict, npz_obj, p_low: float = 1, p_high: float = 99):
    """Visualize exact model inputs using manifest.models[*].input.npz_key."""
    models = (manifest or {}).get("models") or []
    by_key = defaultdict(list)
    sensor_for_key = {}

    for m in models:
        model_name = m.get("model", "<unknown>")
        input_meta = m.get("input") or {}
        key = input_meta.get("npz_key")
        if not key:
            print(f"[skip] {model_name}: no saved input")
            continue
        by_key[key].append(model_name)
        sensor_for_key.setdefault(key, m.get("sensor") or {})

    if not by_key:
        print("No model inputs found in manifest.models[*].input.npz_key")
        return

    npz_keys = set(npz_obj.files) if hasattr(npz_obj, "files") else set(npz_obj.keys())
    print("\n=== Visualizing exact model inputs ===")
    for key, model_names in by_key.items():
        if key not in npz_keys:
            print(f"[missing] {key} not found in NPZ")
            continue

        x = npz_obj[key]
        c = int(x.shape[0]) if getattr(x, "ndim", 0) == 3 else 0
        rgb_idx = infer_rgb_idx_from_sensor(sensor_for_key.get(key), c)
        title = f"{key} <- {', '.join(model_names)}"

        if rgb_idx is None:
            show_input_chw(x, title=title, p_low=p_low, p_high=p_high)
        else:
            show_input_chw(x, title=title, rgb_idx=rgb_idx, p_low=p_low, p_high=p_high)


def load_export_npz(npz_path, json_path=None):
    """Load exported NPZ + sidecar manifest JSON."""
    npz_path = Path(npz_path)
    json_path = Path(json_path) if json_path is not None else npz_path.with_suffix(".json")

    if not npz_path.exists():
        raise FileNotFoundError(f"Missing: {npz_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Missing: {json_path}")

    z = np.load(npz_path)
    with open(json_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest, z


def print_export_summary(manifest: dict, npz_obj=None):
    """Print concise export summary from manifest."""
    print("=== Manifest summary ===")
    print("npz_path:", manifest.get("npz_path"))
    print("backend:", manifest.get("backend"))
    print("spatial:", manifest.get("spatial"))
    print("temporal:", manifest.get("temporal"))
    if npz_obj is not None:
        print("npz_keys:", getattr(npz_obj, "files", None))

    models = manifest.get("models") or []
    print("\n=== model -> input key ===")
    for m in models:
        model_name = m.get("model")
        input_meta = m.get("input") or {}
        print(f"{model_name}: {input_meta.get('npz_key')}")


def inspect_export_npz(npz_path, json_path=None, p_low: float = 1, p_high: float = 99):
    """One-call helper: load export, print summary, visualize exact model inputs."""
    manifest, z = load_export_npz(npz_path=npz_path, json_path=json_path)
    print_export_summary(manifest, z)
    visualize_manifest_inputs(manifest, z, p_low=p_low, p_high=p_high)

    artifacts = manifest.get("artifacts") or {}
    if artifacts:
        print("\n=== Artifacts ===")
        print(json.dumps(artifacts, ensure_ascii=False, indent=2))

    return manifest, z

# Aurora (`aurora`)

## Quick Facts

| Field              | Value                                                                                    |
| ------------------ | ---------------------------------------------------------------------------------------- |
| Model ID           | `aurora`                                                                                 |
| Family / Backbone  | Microsoft Aurora 0.25° (1.3B; Perceiver3D encoder + Swin3D backbone + decoder) — **rs-embed 只保留 encoder** |
| Adapter type       | `on-the-fly`                                                                             |
| Training alignment | High（输入即 ERA5 0.25° 原生网格、原始物理单位；归一化用官方内置统计量）                      |

!!! success "Aurora In 30 Seconds"
    Aurora 是 Microsoft 的大气基础模型（Nature 2025），输入是 ERA5 风格的天气场而非卫星影像。
    在 `rs-embed` 里它作为**天气/气候上下文 embedding** 模型运行：给定点/ROI 与时间窗，
    取该位置周边一个 0.25° 经纬网格窗口的 ERA5 状态（当前时刻 + 前 6 小时两帧），
    组装成 Aurora 官方 `Batch`，只跑 encoder，把 latent token 池化成 embedding。

    最重要的特性：

    - **三个数据源拼一个 Batch**：地表变量走现有 GEE provider，气压层变量从 ARCO-ERA5 公开 zarr 匿名读取，静态变量用官方 HF 仓库的 static pickle — 见 [Data Sources](#data-sources-alignment)
    - **encoder-only**：加载 checkpoint 后丢弃 backbone/decoder；encoder 无跨空间 attention，token 是逐 patch 的"大气柱"特征 — 见 [Encoder-only Inference](#encoder-only-inference)
    - temporal 语义是**时刻快照**（snap 到 6h 边界的 T=2 历史帧），不是影像模型的窗口合成 — 见 [Temporal Semantics](#temporal-semantics)
    - 输入必须对齐**全球 0.25° 经纬格点**（lat 递减、lon ∈ [0,360)），适配层负责 snap/重采样 — 见 [Data Sources](#data-sources-alignment)

---

## Input Contract

| Field                 | Value                                                                                     |
| --------------------- | ----------------------------------------------------------------------------------------- |
| Backend               | provider（`auto` 推荐；GEE 取地表变量）+ ARCO-ERA5 zarr（气压层）+ HF static pickle（静态） |
| `TemporalSpec`        | `range` 推荐 — 取窗口**末端前最后一个 6h 边界**（00/06/12/18 UTC）为当前时刻 t，历史帧 t−6h |
| Default collection    | `ECMWF/ERA5_HOURLY`（0.25°，逐小时，1940–present，滞后 ~5 天）                              |
| Default bands (order) | `temperature_2m, u_component_of_wind_10m, v_component_of_wind_10m, mean_sea_level_pressure`（↔ Aurora `2t, 10u, 10v, msl`） |
| Default fetch         | `scale_m=27830`，无云过滤（ERA5 无云概念），原始物理单位（K / m/s / Pa）                    |
| `input_chw`           | `TCHW`（`T=2, C=4`，地表 4 变量，原始单位）— 仅覆盖 GEE 部分；气压层/静态仍在线获取          |
| Side inputs           | 气压层 5 变量 × 13 层 × T=2（ARCO-ERA5）；静态 `lsm/slt/z`（HF pickle，进程内缓存）          |

空间窗口：以 ROI 中心为锚，在全球 0.25° 格点上取**固定 32×32**（≈ 8°×8°，patch_size=4 的倍数）；
靠近极点/0° 经线时窗口整体平移（不缩小），保证 ROI 在窗内且满足 Aurora 的 lat/lon 单调约束。
ROI 本身通过 `roi_window_geo` 语义参与输出裁剪（同其他模型）。
**不设 `_requires_square_input`**——窗口由 embedder 在经纬格点上自行推导，不走米制方形放大。

---

## Data Sources & Alignment

Aurora `Batch` 的三块输入及来源：

| Batch 字段     | 变量                                | 来源                                                                 | 形状 |
| ------------- | ----------------------------------- | -------------------------------------------------------------------- | ---- |
| `surf_vars`   | `2t, 10u, 10v, msl`                 | GEE `ECMWF/ERA5_HOURLY`（现有 provider 通路，2 个显式时间 bin）        | `(1, 2, H, W)` |
| `atmos_vars`  | `t, u, v, q, z` @ 13 hPa 层         | ARCO-ERA5 `gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3`（匿名，xarray+zarr+gcsfs） | `(1, 2, 13, H, W)` |
| `static_vars` | `lsm, slt, z`                       | HF `microsoft/aurora` 的 `aurora-0.25-static.pickle`（12.5 MB，一次下载全局缓存） | `(H, W)` |

对齐规则（`Batch` 硬性要求，官方文档）：

- lat **严格递减**，范围 [90, −90]；lon **递增**，范围 [0, 360)。
- 窗口边界 snap 到全球 0.25° 格点；GEE 返回的网格若与格点存在半像素/投影偏差，
  适配层做确定性双线性重采样到目标格点（纯 numpy 函数，单测锁定）。
- 三个来源同为 ERA5 0.25° 网格，对齐后逐格点一致；时间戳按 t−6h/t 精确选取（非合成）。
- 气压层固定 13 层（Aurora 预训练层）：`50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000` hPa。
- **ARCO 按时间步整球分块**：任何窗口读取都会下载完整全球场。适配层因此按时间戳缓存全球 slab
  （`_arco_global_slab`，~270 MB/时刻，进程内 LRU）——同一时间戳的一批点只下载一次，之后全是本地切片。
- 变量**不做任何归一化**（物理单位直接进 Batch）——归一化由 Aurora 内部 `batch.normalise()` 用官方统计量完成。
  因此**禁止**走 `coerce_input_to_tchw`/`normalize_s2`（两者裁剪 [0,10000]，会毁掉 Pa 量级与负风速）。

设计取舍记录（2026-08-05，与用户确认）：GEE 公开目录**没有任何气压层 3D 数据**（目录级限制），
候选方案里选了「GEE 地表 + ARCO-ERA5 补气压层」：最大程度沿用现有 GEE 通路，
缺口用同源（ERA5）、同网格（0.25°）、可匿名访问的 ARCO zarr 补齐。备选「全走 ARCO」「本地 NetCDF」被否。

---

## Temporal Semantics

- 入口统一 `temporal_to_range(temporal)`（`None` → 包默认窗口 + warning，同全仓约定）。
- 解析出 `[start, end)` 后：**t = end 之前最后一个 00/06/12/18 UTC 边界**，历史帧 t−6h；
  两帧各对应一个显式 GEE bin（`fetch_collection_binned_raw_tchw`，OlmoEarth 先例），ARCO 侧按同一时间戳精确选片。
- `start` 不参与取数（Aurora 是瞬时状态模型，不是窗口合成模型），meta 里记录已解析窗口与实际使用的两个时间戳。
- ERA5 发布滞后 ~5 天：窗口末端太新会导致 GEE 空集合报错（`ProviderError` 语义，报错信息提示改窗口）。

---

## Encoder-only Inference

- 权重：HF `microsoft/aurora`（MIT）。variant `pretrained`（默认）= `aurora-0.25-pretrained.ckpt`（5.03 GB，D=512）；
  variant `small` = `aurora-0.25-small-pretrained.ckpt`（451 MB，D=256，架构同契约，适合测试/低资源）。
  依赖 pip 包 `microsoft-aurora`（≥2.0，MIT），放 `aurora` extra，懒加载。
- 加载：实例化官方 `Aurora`/`AuroraSmallPretrained` → `load_checkpoint()` → **丢弃 backbone/decoder**（显存/内存占用降约一个数量级；瞬时峰值仍需 ~10 GB CPU RAM，small 版 ~1 GB）。
- 前处理不自己复刻：走官方 `model.forward` 的原生路径（dtype cast → `batch.normalise` → `batch.crop(patch_size)` → device），在 encoder 出口截获输出并中止后续阶段，保证与官方前处理零漂移、不随上游版本变化失同步。
- token 布局：`(B, latent_levels × H/4 × W/4, D)`。encoder 内只有 patch embed + 跨气压层的 Perceiver 聚合，
  **无跨空间 attention**——每个 token 是一个 1°×1° patch（4×4 格点） 的"大气柱"表征，窗口大小不影响单 token 数值（这正是 encoder-only 省算力的依据）。
  backbone 深层特征暂不暴露，留作后续 `stage` 配置项的扩展空间（本期不做）。

## Output Semantics

- 先把 token 重排为 `(latent_levels, H/4, W/4, D)`，**对 latent_levels 取均值** → `(D, H', W')` 的柱特征网格。
- `OutputSpec.pooled()`：ROI 裁剪后网格均值（`roi_grid_mean`，同全仓约定；点输入即窗口中心 token 邻域）；无 ROI 裁剪时全网格均值（`token_mean`）。Aurora 无 CLS token，均值池化是唯一合理默认。
- `OutputSpec.grid()`：`grid_to_dataarray` 输出 `(D, H', W')`，`grid_kind="aurora_column_tokens"`，north-up（lat 递减天然满足）。
- 未知 `output.mode`/`pooling` → `raise ModelError`（不静默回退）。

## Model Config

| key       | default        | 说明                                              |
| --------- | -------------- | ------------------------------------------------- |
| `variant` | `"pretrained"` | `pretrained` \| `small`；非法值 `raise ModelError` |

设计取舍：`window_px` / `atmos_levels` 在 v1 固定为常量（32 / 13 层预训练全集），**不开放配置**。
原因：两者都影响取数结果，而共享层 `fetch_input_extras_from_model_config` 只转发 `temporal_mode`
——开放它们会让 prefetch 路径与直连路径取到不同输入，破坏单点=批量=导出的等价性契约
（`test_export_input_prep_consistency`）。若未来需要，应先在共享层扩展 fetch-affecting
config 的通用转发机制（独立 PR），再放开这两个键。`variant` 不影响取数，安全开放。

## 与现有模型的关系

- 结构上最像 **OlmoEarth**（覆写 `fetch_input`、显式时间 bin、多源路由）；输出 builder 参照 **TerraMind** 的 `_build_embedding` 静态方法模式。
- 是全仓第一个非影像输入模型；ARCO/HF-static 侧不属于 GEE provider，但走 embedder 内封装的独立小函数（先例：tessera 的 geotessera、copernicus 的本地 GeoTIFF 均为非 GEE 源）。

## Examples

```python
from rs_embed import get_embedding, PointBuffer, TemporalSpec, OutputSpec

emb = get_embedding(
    "aurora",
    spatial=PointBuffer(lon=-88.2, lat=40.1, buffer_m=2048),
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    output=OutputSpec.pooled(),
    backend="auto",
)
# → (512,)，2022-08-31 18:00 UTC 大气状态的柱特征

emb_small = get_embedding(
    "aurora",
    spatial=PointBuffer(lon=-88.2, lat=40.1, buffer_m=2048),
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    output=OutputSpec.grid(),
    backend="auto",
    variant="small",
)
# → xarray (256, 8, 8)
```

## Common Failure Modes / Debugging

| 报错/现象 | 原因与处理 |
| --------- | ---------- |
| `No hourly ... asset for [...] (ERA5 publishes with ~5 days delay)` | 时间窗末端太新，GEE 还没有该小时的 ERA5 资产——把窗口末端往前挪几天。 |
| `Timestamp ... is outside ARCO-ERA5 coverage` | ARCO 镜像比 ERA5 滞后数月；窗口末端改早一些。GEE 与 ARCO 的可用期不同步，以两者交集为准。 |
| `Failed to open the ARCO-ERA5 zarr store` | 缺 gcsfs/zarr（`pip install 'rs-embed[aurora]'`）或无法访问 GCS 匿名端点。 |
| `Aurora requires the optional dependency 'microsoft-aurora'` | 未装 aurora extra。 |
| 首次调用很慢/内存暴涨 | 三重冷启动：checkpoint 下载（pretrained 5 GB）、加载期瞬时 CPU RAM 峰值 ~10 GB（encoder-only 化后回落）、每个新时间戳的 ARCO 全球 slab ~270 MB×2。small variant + 复用时间戳可显著缓解。 |
| `lat/lon-grid fetch ... returned shape ...` | crsTransform 采样返回形状与请求不符——窗口越界或集合不是经纬网格产品；这是 loud-fail 设计，报 issue 时附完整信息。 |
| GEE 未认证 | 与其他模型一致：`earthengine authenticate` / `EE_PROJECT`。 |

## Reproducibility Notes

- 权重 pinned：`microsoft-aurora` 包内各变体带固定的 HF revision（`AuroraPretrained.default_checkpoint_revision`），`load_checkpoint()` 无参调用即锁定版本。
- 输入确定性：窗口/时间戳均为 (spatial, temporal) 的纯函数（格点 snap + 6h 边界 snap）；同请求同输入。
- ERA5 本身有 ERA5T→final 的后订正（约 3 个月内的数据可能被替换）；对近期时间窗，GEE 与 ARCO 的取值可能属于不同订正批次，跨源一致性测试用 `atol=0.5`（K/Pa 量级远低于变量动态范围）。
- meta 里记录 `time_pair`、`window`、`atmos_levels`、`hf_id`、`encoder_only`，足以复现一次取数与前向。

## Source of Truth (Code Pointers)

- Embedder：`src/rs_embed/embedders/onthefly_aurora.py`（窗口几何 `_lattice_window`/`_window_roi`、时间 snap `resolve_time_pair`、ARCO slab 缓存 `_arco_global_slab`、encoder 截获 `_encoder_tokens`）。
- 共享层原语：`src/rs_embed/providers/base.py` + `gee.py` 的 `fetch_latlon_grid_chw`（EPSG:4326 crsTransform 采样，通用、无模型名）、`src/rs_embed/providers/fetch.py` 的 `fetch_latlon_grid_bins_tchw`（小时精度 bin + NaN 哨兵）。
- 测试：`tests/test_aurora.py`（几何/时间/通道布局/输出语义 + `RS_EMBED_LIVE_AURORA=1` 门控的 GEE↔ARCO 配准校验与真权重 smoke）。

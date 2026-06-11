<div align="center">

# <img src="https://raw.githubusercontent.com/cybergis/rs-embed/main/docs/assets/icon.png" width="35" alt="icon" /> rs-embed

**One line code to get Any Remote Sensing Foundation Model (RSFM) embeddings for Any Place and Any Time**

[![arXiv](https://img.shields.io/badge/arXiv-2602.23678-b31b1b.svg)](https://arxiv.org/abs/2602.23678)
[![Docs](https://img.shields.io/badge/docs-online-brightgreen)](https://cybergis.github.io/rs-embed/)
![Python](https://img.shields.io/badge/python-3.12-blue?logo=python)
[![PyTorch 2.2](https://img.shields.io/badge/torch-2.2-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

![Visitors](https://visitor-badge.laobi.icu/badge?page_id=cybergis.rs-embed)
![Last Commit](https://img.shields.io/github/last-commit/cybergis/rs-embed)
![License](https://img.shields.io/github/license/cybergis/rs-embed)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/rs-embed?period=total&units=INTERNATIONAL_SYSTEM&left_color=GREY&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/rs-embed)

[Docs](https://cybergis.github.io/rs-embed/) · [StartNow](https://github.com/cybergis/rs-embed/blob/main/examples/playground.ipynb) · [Releases](https://github.com/cybergis/rs-embed/releases) · [Changelog](https://github.com/cybergis/rs-embed/blob/main/CHANGELOG.md) · [UseCase](https://github.com/cybergis/rs-embed/blob/main/examples/demo.ipynb) · [Paper](https://arxiv.org/abs/2602.23678)

</div>

> Get Start on [I-GUIDE](https://platform.i-guide.io/notebooks/a013ce97-d963-4262-9f21-edd09976181a) Today!

<img src="https://raw.githubusercontent.com/cybergis/rs-embed/main/docs/assets/background.png" />

## TL;DR

```python
emb = get_embedding("prithvi", spatial=..., temporal=..., output=...)
```

## Install

```bash
# base install (always use the latest version for best experience)
pip install --upgrade rs-embed

# add [terratorch] only if you use terramind
pip install --upgrade "rs-embed[terratorch]"
```


For local development:

```bash
git clone https://github.com/cybergis/rs-embed.git
cd rs-embed
pip install -e .  # use -e ".[terratorch]" if you need terramind
```

If this is your first time using Google Earth Engine, authenticate once:

```bash
earthengine authenticate
```



## Quick Example

```python
from rs_embed import PointBuffer, TemporalSpec, OutputSpec, get_embedding

spatial = PointBuffer(lon=121.5, lat=31.2, buffer_m=2048)
temporal = TemporalSpec.range(
    "2022-06-01",
    "2022-09-01",
)

emb = get_embedding(
    "prithvi",
    spatial=spatial,
    temporal=temporal,
    output=OutputSpec.pooled(),
)

```

> **Tip:** Default settings are designed as a trade-off between compute cost and embedding quality. If you have sufficient compute resources, check [Choosing Settings](https://cybergis.github.io/rs-embed/0.1.3/choosing_settings/) and individual [model pages](https://cybergis.github.io/rs-embed/0.1.3/models/) to get the best results.

See the visualization helper and end-to-end notebook in the repository:

- [`examples/plot_utils.py`](https://github.com/cybergis/rs-embed/blob/main/examples/plot_utils.py)
- [`examples/playground.ipynb`](https://github.com/cybergis/rs-embed/blob/main/examples/playground.ipynb)

<img src="https://raw.githubusercontent.com/cybergis/rs-embed/main/docs/assets/vis.png" width=750 />

## Main API

For new users, start with these primary APIs:

- `get_embedding(...)`: one ROI -> one embedding
- `get_embeddings_batch(...)`: many ROIs, same model
- `export_batch(...)`: export datasets / experiments (single or multiple ROIs)
- `inspect_provider_patch(...)`: inspect raw provider patches before inference

## Supported Models

This is a convenience index with basic model info only (for quick scanning / links). For detailed I/O behavior and preprocessing notes, see [Supported Models](https://cybergis.github.io/rs-embed/models/).

### Precomputed Embeddings

| Model ID            | Resolution | Time Coverage | Publication                                     |
| ------------------- | ---------- | ------------- | ----------------------------------------------- |
| `tessera`           | 10m        | 2017-2025     | [CVPR 2026](https://arxiv.org/abs/2506.20380v4) |
| `gse` (Alpha Earth) | 10 m       | 2017-2024     | [arXiv 2025](https://arxiv.org/abs/2507.22291)  |
| `copernicus`        | 0.25°      | 2021          | [ICCV 2025](https://arxiv.org/abs/2503.11849)   |

### On-the-fly Foundation Models

| Model ID          | Primary Input            | Resolution(Default) | Publication                                                                     | Link                                                         |
| ----------------- | ------------------------ | ------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `satmae`          | S2 RGB                   | 10m                 | [NeurIPS 2022](https://arxiv.org/abs/2207.08051)                                | [link](https://github.com/sustainlab-group/SatMAE)           |
| `satmaepp`        | S2 RGB                   | 10m                 | [CVPR 2024](https://arxiv.org/abs/2403.05419)                                   | [link](https://github.com/techmn/satmae_pp)                  |
| `satmaepp_s2_10b` | S2 SR 10-band            | 10m                 | [CVPR 2024](https://arxiv.org/abs/2403.05419)                                   | [link](https://github.com/techmn/satmae_pp)                  |
| `prithvi`         | S2 6-band                | 30m                 | [arXiv 2023](https://arxiv.org/abs/2310.18660)                                  | [link](https://huggingface.co/ibm-nasa-geospatial)           |
| `scalemae`        | S2 RGB (+ scale)         | 10m                 | [ICCV 2023](https://arxiv.org/abs/2212.14532)                                   | [link](https://github.com/bair-climate-initiative/scale-mae) |
| `remoteclip`      | S2 RGB                   | 10m                 | [TGRS 2024](https://arxiv.org/abs/2306.11029)                                   | [link](https://github.com/ChenDelong1999/RemoteCLIP)         |
| `dofa`            | Multi-band + wavelengths | 10m                 | [arXiv 2024](https://arxiv.org/abs/2403.15356)                                  | [link](https://github.com/zhu-xlab/DOFA)                     |
| `satvision`       | TOA 14-channel           | 1000m               | [arXiv 2024](https://arxiv.org/abs/2411.17000)                                  | [link](https://github.com/nasa-nccs-hpda/pytorch-caney)      |
| `anysat`          | S2 time series (10-band) | 10m                 | [CVPR 2025](https://arxiv.org/abs/2412.14123)                                   | [link](https://github.com/gastruc/AnySat)                    |
| `galileo`         | S2 time series (10-band) | 10m                 | [ICML 2025](https://arxiv.org/abs/2502.09356)                                   | [link](https://github.com/nasaharvest/galileo)               |
| `wildsat`         | S2 RGB                   | 10m                 | [ICCV 2025](https://arxiv.org/abs/2412.14428)                                   | [link](https://github.com/mdchuc/HRSFM)                      |
| `fomo`            | S2 12-band               | 10m                 | [AAAI 2025](https://arxiv.org/abs/2312.10114)                                   | [link](https://github.com/RolnickLab/FoMo-Bench)             |
| `terramind`       | S2 12-band               | 10m                 | [ICCV 2025](https://arxiv.org/abs/2504.11171)                                   | [link](https://github.com/IBM/terramind)                     |
| `terrafm`         | S2 12-band / S1 VV-VH    | 10m                 | [ICLR 2026](https://arxiv.org/abs/2506.06281)                                   | [link](https://github.com/mbzuai-oryx/TerraFM)               |
| `thor`            | S2 10-band               | 10m                 | [arXiv 2026](https://arxiv.org/abs/2601.16011)                                  | [link](https://github.com/FM4CS/THOR)                        |
| `agrifm`          | S2 time series (10-band) | 10m                 | [RSE 2026](https://www.sciencedirect.com/science/article/pii/S0034425726000040) | [link](https://github.com/flyakon/AgriFM)                    |
| `olmoearth`       | S2 L2A 12-band / S1 VV-VH    | 10m                 | [arXiv 2025](https://arxiv.org/abs/2511.13655)                                  | [link](https://huggingface.co/collections/allenai/olmoearth) |

Resolution here means the default provider/source fetch resolution used by the adapter, not the final resized tensor shape seen by the model.

## Learn More

📚 [Full documentation](https://cybergis.github.io/rs-embed/)

🪄 [Get Started: Try `rs-embed` Now](https://github.com/cybergis/rs-embed/blob/main/examples/playground.ipynb)

🪀 [Use case: Maize yield mapping Illinois](https://github.com/cybergis/rs-embed/blob/main/examples/demo.ipynb)

📢 [Disscusion](https://github.com/cybergis/rs-embed/discussions)

🧾 [Release policy and versioning](https://cybergis.github.io/rs-embed/releases/)

📌 [Project changelog](https://github.com/cybergis/rs-embed/blob/main/CHANGELOG.md)

## Extending & Contributing

We welcome issues for new model integrations, extension ideas, bugs, and documentation gaps. If you have your own work, or a model or paper that you think would be valuable to include in `rs-embed`, please open an [Issue](https://github.com/cybergis/rs-embed/issues) and share the relevant links, context, and examples.

We also warmly welcome community contributions, including new model support, bug fixes, documentation improvements, and example notebooks. If you would like to contribute directly, please start with the [`extending`](https://cybergis.github.io/rs-embed/extending/) guide and the [contributing guide](https://cybergis.github.io/rs-embed/contributing/).

## 🎖 Acknowledgements

We would like to thank the following organizations and projects that make rs-embed possible: [Google Earth Engine](https://earthengine.google.com), [TorchGeo](https://github.com/torchgeo/torchgeo), [GeoTessera](https://github.com/ucam-eo/geotessera), [TerraTorch](https://github.com/terrastackai/terratorch), [rshf](https://github.com/mvrl/rshf), and the [Copernicus-Embed](https://huggingface.co/datasets/torchgeo/copernicus_embed).

This library also builds upon the incredible work of the Remote Sensing community!(Full list and citations available in our Documentation)

## Citation

```
@article{ye2026modelplacetimeremote,
      title={Any Model, Any Place, Any Time: Get Remote Sensing Foundation Model Embeddings On Demand},
      author={Dingqi Ye and Daniel Kiv and Wei Hu and Jimeng Shi and Shaowen Wang},
      year={2026},
      eprint={2602.23678},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.23678},
}
```

## License

This project is released under the [Apache-2.0](https://github.com/cybergis/rs-embed/blob/main/LICENSE)

## Contributors

<a href="https://github.com/cybergis/rs-embed/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=cybergis/rs-embed" />
</a>

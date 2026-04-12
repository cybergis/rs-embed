![rs-embed banner](assets/banner.png)

> One line of code to get embeddings from **any Remote Sensing Foundation Model (RSFM)** for **any place** and **any time**

---

## Motivation

![rs-embed background](assets/background.png)

The remote sensing community has seen an explosion of foundation models in recent years.
Yet, using them in practice remains surprisingly painful. Model interfaces vary widely between imagery models and precomputed embedding products, input semantics are often ambiguous, and temporal, spectral, and spatial assumptions differ enough that cross-model comparison becomes tedious. In practice, even simple benchmarking can turn into glue-code work.

RS-Embed aims to fix this.

!!! success "Goal"
    Provide a **minimal**, **unified**, and **stable API** that turns diverse RS foundation models into a simple `ROI → embedding service` — so researchers can focus on **downstream tasks**, **benchmarking**, and **analysis**, not glue code.

---

## Start Here

Use the documentation the same way you would approach a mature library: start with one successful run, choose a model only after you understand the core API shape, and drop into the reference when you need exact contracts.

| Start here if you want to...  | Page                                                                                                 |
| ----------------------------- | ---------------------------------------------------------------------------------------------------- |
| get something running quickly | [Quickstart](quickstart.md): install the package, run a first example, and learn the three core APIs |
| understand settings first     | [Before You Start](choosing_settings.md): learn which settings change embedding quality, semantics, and runtime before you tune anything |
| choose a model                | [Models](models.md): shortlist model IDs by task, input type, and temporal behavior                  |
| check exact signatures        | [API](api.md): exact signatures for specs, embedding, export, and inspection                         |
| understand internals          | [Architecture](architecture.md): module map, call flows, and registration mechanics                  |
| add support for a new model   | [Extending](extending.md): add a new model adapter or integrate with the registry/export flow        |

---

## Why rs-embed

rs-embed is designed to make embedding acquisition work the same way whether you are probing one location once or building a large benchmark dataset. The public surface stays small enough for quick use, but still exposes the controls you need for fetch policy, output shape, export configuration, and model-specific options.

Most workflows revolve around three functions: `get_embedding(...)` for one ROI, `get_embeddings_batch(...)` for repeated inference with one model, and `export_batch(...)` for dataset or benchmark generation. The rest of the documentation expands on those paths rather than introducing a second parallel abstraction layer.

---

## Common Tasks

| Goal                             | Page                          | Main API                      |
| -------------------------------- | ----------------------------- | ----------------------------- |
| Get one embedding for one ROI    | [Quickstart](quickstart.md)   | `get_embedding(...)`          |
| Compute embeddings for many ROIs | [Quickstart](quickstart.md)   | `get_embeddings_batch(...)`   |
| Build an export dataset          | [Quickstart](quickstart.md)   | `export_batch(...)`           |
| Compare model assumptions        | [Models](models.md)           | model tables + detail pages   |
| Inspect a raw provider patch     | [Inspect API](api_inspect.md) | `inspect_provider_patch(...)` |

---

## Advanced Reading

Once the basic flow is familiar, read [Before You Start](choosing_settings.md) before changing model defaults for speed or quality. Then use [Concepts](concepts.md) for the semantics behind `TemporalSpec`, `OutputSpec`, and backends, [Workflows](workflows.md) for task-oriented recipes, and [Advanced Model Reference](models_reference.md) for deeper preprocessing and temporal comparison details. [Limitations](limitations.md) collects current constraints and edge cases.

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

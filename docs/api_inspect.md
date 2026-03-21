# API: Inspect

This page documents raw input inspection utilities (patch checks) used before model inference.

Related pages:

- [API: Specs and Data Structures](api_specs.md)
- [API: Embedding](api_embedding.md)
- [API: Export](api_export.md)

---

## inspect_provider_patch (recommended) { #inspect_provider_patch }

```python
inspect_provider_patch(
    *,
    spatial: SpatialSpec,
    temporal: Optional[TemporalSpec] = None,
    sensor: SensorSpec,
    backend: str = "gee",
    name: str = "gee_patch",
    value_range: Optional[Tuple[float, float]] = None,
    return_array: bool = False,
) -> Dict[str, Any]
```

Provider-agnostic patch inspection utility (recommended entry point).
Use this when you want the same inspection flow but with a non-GEE provider backend.

---

---

"""Vendored THOR runtime adapted from FM4CS/thor_terratorch_ext.

Source:
- https://github.com/FM4CS/thor_terratorch_ext
- commit fdbc4c4345adfbdbc41ed5b497a05b56eeec96af

Local changes:
- removed TerraTorch registry integration
- trimmed to the direct THOR loading path used by rs-embed
- replaced upstream thor package imports with vendored local runtime imports
- only warn on input param override when the value actually differs from the
  default (was warning unconditionally whenever the key was present)
"""

from __future__ import annotations

import logging
import warnings
from copy import deepcopy
from typing import Any, Literal

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

logger = logging.getLogger(__name__)


_default_input_params = {
    "ground_covers": [2880],
    "aggr_type": "subsetmean",
    "use_superposition_encoding": False,
    "use_fuzzy_encoding": False,
    "encoder_pos_type": "alibi",
    "cls_token_type": "pooled",
    "use_flexivit": False,
    "flexivit_ref_patch_size": 4,
    "flexivit_patch_size_seqs": [8],
    "flexivit_ref_grid_size": 14,
    "select_patch_strategy": "min",
    "groups": [
        ["S2:Red", "S2:Green", "S2:Blue", "S2:NIR"],
        ["S2:RE1", "S2:RE2", "S2:RE3", "S2:RE4", "S2:SWIR1", "S2:SWIR2"],
        ["S2:CoastAerosal", "S2:WaterVapor"],
        ["S1:IW-VH", "S1:IW-VV", "S1:EW-VH", "S1:EW-VV"],
        ["S1:IW-HV", "S1:IW-HH", "S1:EW-HV", "S1:EW-HH"],
        [
            "S3:Oa01_reflectance",
            "S3:Oa02_reflectance",
            "S3:Oa03_reflectance",
            "S3:Oa04_reflectance",
            "S3:Oa05_reflectance",
            "S3:Oa06_reflectance",
            "S3:Oa07_reflectance",
        ],
        [
            "S3:Oa08_reflectance",
            "S3:Oa09_reflectance",
            "S3:Oa10_reflectance",
            "S3:Oa11_reflectance",
            "S3:Oa12_reflectance",
            "S3:Oa13_reflectance",
            "S3:Oa14_reflectance",
        ],
        [
            "S3:Oa15_reflectance",
            "S3:Oa16_reflectance",
            "S3:Oa17_reflectance",
            "S3:Oa18_reflectance",
            "S3:Oa19_reflectance",
            "S3:Oa20_reflectance",
            "S3:Oa21_reflectance",
        ],
        [
            "S3:S1_reflectance_an",
            "S3:S2_reflectance_an",
            "S3:S3_reflectance_an",
            "S3:S4_reflectance_an",
            "S3:S5_reflectance_an",
            "S3:S6_reflectance_an",
        ],
        ["S3:S7_BT_in", "S3:S8_BT_in", "S3:S9_BT_in"],
    ],
    "channels": {
        "S2:Red": {"GSD": 10, "patch_size": 16},
        "S2:Green": {"GSD": 10, "patch_size": 16},
        "S2:Blue": {"GSD": 10, "patch_size": 16},
        "S2:NIR": {"GSD": 10, "patch_size": 16},
        "S2:RE1": {"GSD": 20, "patch_size": 16},
        "S2:RE2": {"GSD": 20, "patch_size": 16},
        "S2:RE3": {"GSD": 20, "patch_size": 16},
        "S2:RE4": {"GSD": 20, "patch_size": 16},
        "S2:SWIR1": {"GSD": 20, "patch_size": 16},
        "S2:SWIR2": {"GSD": 20, "patch_size": 16},
        "S2:CoastAerosal": {"GSD": 60, "patch_size": 16},
        "S2:WaterVapor": {"GSD": 60, "patch_size": 16},
        "S1:IW-VV": {"GSD": 10, "patch_size": 16, "patch_embed_name": "S1:VV"},
        "S1:IW-VH": {"GSD": 10, "patch_size": 16, "patch_embed_name": "S1:VH"},
        "S1:IW-HV": {"GSD": 10, "patch_size": 16, "patch_embed_name": "S1:HV"},
        "S1:IW-HH": {"GSD": 10, "patch_size": 16, "patch_embed_name": "S1:HH"},
        "S1:EW-VV": {"GSD": 10, "patch_size": 16, "patch_embed_name": "S1:VV"},
        "S1:EW-VH": {"GSD": 10, "patch_size": 16, "patch_embed_name": "S1:VH"},
        "S1:EW-HV": {"GSD": 10, "patch_size": 16, "patch_embed_name": "S1:HV"},
        "S1:EW-HH": {"GSD": 10, "patch_size": 16, "patch_embed_name": "S1:HH"},
        "S3:Oa01_reflectance": {"GSD": 240, "patch_size": 16},
        "S3:Oa02_reflectance": {"GSD": 240, "patch_size": 16},
        "S3:Oa03_reflectance": {"GSD": 240, "patch_size": 16},
        "S3:Oa04_reflectance": {"GSD": 240, "patch_size": 16},
        "S3:Oa05_reflectance": {"GSD": 240, "patch_size": 16},
        "S3:Oa06_reflectance": {"GSD": 240, "patch_size": 16},
        "S3:Oa07_reflectance": {"GSD": 240, "patch_size": 16},
        "S3:Oa08_reflectance": {"GSD": 240, "patch_size": 16},
        "S3:Oa09_reflectance": {"GSD": 240, "patch_size": 16},
        "S3:Oa10_reflectance": {"GSD": 240, "patch_size": 16},
        "S3:Oa11_reflectance": {"GSD": 240, "patch_size": 16},
        "S3:Oa12_reflectance": {"GSD": 240, "patch_size": 16},
        "S3:Oa13_reflectance": {"GSD": 240, "patch_size": 16},
        "S3:Oa14_reflectance": {"GSD": 240, "patch_size": 16},
        "S3:Oa15_reflectance": {"GSD": 240, "patch_size": 16},
        "S3:Oa16_reflectance": {"GSD": 240, "patch_size": 16},
        "S3:Oa17_reflectance": {"GSD": 240, "patch_size": 16},
        "S3:Oa18_reflectance": {"GSD": 240, "patch_size": 16},
        "S3:Oa19_reflectance": {"GSD": 240, "patch_size": 16},
        "S3:Oa20_reflectance": {"GSD": 240, "patch_size": 16},
        "S3:Oa21_reflectance": {"GSD": 240, "patch_size": 16},
        "S3:S1_reflectance_an": {"GSD": 480, "patch_size": 16},
        "S3:S2_reflectance_an": {"GSD": 480, "patch_size": 16},
        "S3:S3_reflectance_an": {"GSD": 480, "patch_size": 16},
        "S3:S4_reflectance_an": {"GSD": 480, "patch_size": 16},
        "S3:S5_reflectance_an": {"GSD": 480, "patch_size": 16},
        "S3:S6_reflectance_an": {"GSD": 480, "patch_size": 16},
        "S3:S7_BT_in": {"GSD": 960, "patch_size": 16},
        "S3:S8_BT_in": {"GSD": 960, "patch_size": 16},
        "S3:S9_BT_in": {"GSD": 960, "patch_size": 16},
    },
}

_user_overridable_params = {
    "ground_covers": "List of ground cover sizes (m) that define the crop fed to the ViT",
    "flexivit_patch_size_seqs": "Per-group or shared FlexiViT patch sizes to use",
    "flexivit_ref_patch_size": "Reference patch size for FlexiViT patch embedding",
    "select_patch_strategy": "Strategy for choosing patch sizes across groups (min, max, equal-*)",
}


def _format_overridable_params() -> str:
    return ", ".join(
        f"{key}: {_user_overridable_params[key]}"
        for key in sorted(_user_overridable_params)
    )


def _ensure_allowed_input_params(keys: list[str] | tuple[str, ...] | set[str]) -> None:
    disallowed_keys = sorted(set(keys) - set(_user_overridable_params))
    if disallowed_keys:
        allowed_str = _format_overridable_params()
        msg = (
            "Cannot override the following THOR ViT input params via configuration: "
            f"{disallowed_keys}. Allowed keys: {allowed_str}"
        )
        raise ValueError(msg)


lookup_band = {
    "COASTAL_AEROSOL": "S2:CoastAerosal",
    "BLUE": "S2:Blue",
    "GREEN": "S2:Green",
    "RED": "S2:Red",
    "RED_EDGE_1": "S2:RE1",
    "RED_EDGE_2": "S2:RE2",
    "RED_EDGE_3": "S2:RE3",
    "NIR_BROAD": "S2:NIR",
    "NIR_NARROW": "S2:RE4",
    "SWIR_1": "S2:SWIR1",
    "SWIR_2": "S2:SWIR2",
    "WATER_VAPOR": "S2:WaterVapor",
    "VV": "S1:IW-VV",
    "VH": "S1:IW-VH",
    "ASC_VV": "S1:IW-VV",
    "ASC_VH": "S1:IW-VH",
    "DSC_VV": "S1:IW-VV",
    "DSC_VH": "S1:IW-VH",
    "IW_VV": "S1:IW-VV",
    "IW_VH": "S1:IW-VH",
    "IW_HV": "S1:IW-HV",
    "IW_HH": "S1:IW-HH",
    "EW_VV": "S1:EW-VV",
    "EW_VH": "S1:EW-VH",
    "EW_HV": "S1:EW-HV",
    "EW_HH": "S1:EW-HH",
    "OA01_REFLECTANCE": "S3:Oa01_reflectance",
    "OA02_REFLECTANCE": "S3:Oa02_reflectance",
    "OA03_REFLECTANCE": "S3:Oa03_reflectance",
    "OA04_REFLECTANCE": "S3:Oa04_reflectance",
    "OA05_REFLECTANCE": "S3:Oa05_reflectance",
    "OA06_REFLECTANCE": "S3:Oa06_reflectance",
    "OA07_REFLECTANCE": "S3:Oa07_reflectance",
    "OA08_REFLECTANCE": "S3:Oa08_reflectance",
    "OA09_REFLECTANCE": "S3:Oa09_reflectance",
    "OA10_REFLECTANCE": "S3:Oa10_reflectance",
    "OA11_REFLECTANCE": "S3:Oa11_reflectance",
    "OA12_REFLECTANCE": "S3:Oa12_reflectance",
    "OA13_REFLECTANCE": "S3:Oa13_reflectance",
    "OA14_REFLECTANCE": "S3:Oa14_reflectance",
    "OA15_REFLECTANCE": "S3:Oa15_reflectance",
    "OA16_REFLECTANCE": "S3:Oa16_reflectance",
    "OA17_REFLECTANCE": "S3:Oa17_reflectance",
    "OA18_REFLECTANCE": "S3:Oa18_reflectance",
    "OA19_REFLECTANCE": "S3:Oa19_reflectance",
    "OA20_REFLECTANCE": "S3:Oa20_reflectance",
    "OA21_REFLECTANCE": "S3:Oa21_reflectance",
    "S1_REFLECTANCE_AN": "S3:S1_reflectance_an",
    "S2_REFLECTANCE_AN": "S3:S2_reflectance_an",
    "S3_REFLECTANCE_AN": "S3:S3_reflectance_an",
    "S4_REFLECTANCE_AN": "S3:S4_reflectance_an",
    "S5_REFLECTANCE_AN": "S3:S5_reflectance_an",
    "S6_REFLECTANCE_AN": "S3:S6_reflectance_an",
    "S7_BT_IN": "S3:S7_BT_in",
    "S8_BT_IN": "S3:S8_BT_in",
    "S9_BT_IN": "S3:S9_BT_in",
}

AVAILABLE_GROUPS = {
    f"group{i}": group for i, group in enumerate(_default_input_params["groups"])
}

name_mapping = {
    "thor_vit_base_encoder_alibi_patch_size_embed_v1": "thor_v1_base",
    "thor_vit_small_encoder_alibi_patch_size_embed_v1": "thor_v1_small",
    "thor_vit_tiny_encoder_alibi_patch_size_embed_v1": "thor_v1_tiny",
    "thor_vit_large_encoder_alibi_patch_size_embed_v1": "thor_v1_large",
}

pretrained_weights = {
    "thor_v1_tiny": {
        "hf_hub_id": "FM4CS/THOR-1.0-tiny",
        "hf_hub_filename": "thor_v1_vit_tiny.pt",
    },
    "thor_v1_small": {
        "hf_hub_id": "FM4CS/THOR-1.0-small",
        "hf_hub_filename": "thor_v1_vit_small.pt",
    },
    "thor_v1_base": {
        "hf_hub_id": "FM4CS/THOR-1.0-base",
        "hf_hub_filename": "thor_v1_vit_base.pt",
    },
    "thor_v1_large": {
        "hf_hub_id": "FM4CS/THOR-1.0-large",
        "hf_hub_filename": "thor_v1_vit_large.pt",
    },
}


def _get_vendored_model_registry():
    from rs_embed.embedders._vendor.thor.core.model_registry import MODELS
    from rs_embed.embedders._vendor.thor.models import enable_alibi_for_timm

    enable_alibi_for_timm()
    from rs_embed.embedders._vendor.thor.models import thor_vit as _thor_vit  # noqa: F401

    return MODELS


def _coerce_band_name(band: Any) -> str:
    if isinstance(band, str):
        return band
    value = getattr(band, "value", None)
    if isinstance(value, str):
        return value
    return str(band)


def process_thor_bands(
    bands: list[Any],
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    thor_bands = []
    channel_params = deepcopy(_default_input_params["channels"])
    for band in bands:
        band_name = _coerce_band_name(band)
        try:
            if (gsd_str := band_name.split("_")[-1]).isdigit() and any(
                part in band_name.split("_") for part in ["VV", "VH", "HH", "HV"]
            ):
                base_band = "_".join(band_name.split("_")[:-1])
                thor_band = lookup_band[base_band]
                if int(gsd_str) != channel_params[thor_band]["GSD"]:
                    channel_params[thor_band]["GSD"] = int(gsd_str)
                    group_channel_members = None
                    for group in AVAILABLE_GROUPS.values():
                        if thor_band in group:
                            group_channel_members = group
                            break
                    if group_channel_members is None:
                        raise ValueError(f"Could not find group for band: {thor_band}")
                    for member in group_channel_members:
                        if channel_params[member]["GSD"] != int(gsd_str):
                            channel_params[member]["GSD"] = int(gsd_str)
                thor_bands.append(thor_band)
            else:
                thor_bands.append(lookup_band[band_name])
        except KeyError as exc:
            raise NotImplementedError(f"This band is not implemented in THOR: {band_name}") from exc

    if any(thor_bands.count(b) > 1 for b in set(thor_bands)):
        duplicates = [
            (b, bands[i]) for i, b in enumerate(thor_bands) if thor_bands.count(b) > 1
        ]
        msg = (
            "Duplicate bands are not allowed/implemented. "
            "Make sure to only use one sigma0 product. "
            f"Duplicates found: {duplicates}"
        )
        raise ValueError(msg)
    return thor_bands, channel_params


class THOREncoderWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        bands: list[str] | None = None,
        out_indices: list[int] | None = None,
        return_channel_params: bool = False,
        merge_method: Literal["concat", "sum", "mean"] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.return_channel_params = return_channel_params
        if merge_method not in ["concat", "sum", "mean", None]:
            msg = (
                f"Unknown merge_method={merge_method!r}. "
                "Expected one of 'concat', 'sum', 'mean', or None."
            )
            raise ValueError(msg)
        self.merge_method = merge_method
        self.channels = self.model.channels

        if bands is None:
            bands = list(self.model.channels.keys())
        self.bands = bands
        self.band_index = list(range(len(bands)))
        self.groups = self.model.get_available_groups(dict.fromkeys(self.bands, None))

        AVAILABLE_GROUPS.clear()
        AVAILABLE_GROUPS.update({
            f"group{i}": group
            for i, group in enumerate(_default_input_params["groups"])
        })
        for group_name in list(AVAILABLE_GROUPS.keys()):
            if group_name not in self.groups:
                del AVAILABLE_GROUPS[group_name]

        self.single_embedding_shape = self.model.state_dict()["norm.bias"].shape[0]

        lowest_gsd = 1_000_000
        for group_member in self.groups.values():
            product_band = group_member[0]
            lowest_gsd = min(lowest_gsd, self.channels[product_band]["GSD"])
        self.lowest_gsd = lowest_gsd

        if out_indices is None:
            num_blocks = len(self.model.blocks)
            out_indices = list(range(0, num_blocks, 1))
        self.out_indices = out_indices

        if not hasattr(self.model, "ground_covers") or len(self.model.ground_covers) != 1:
            raise ValueError("Multiple or missing ground covers found, please specify one.")
        self.ground_cover = self.model.ground_covers[0]
        self.input_size = self.ground_cover // self.lowest_gsd

    @property
    def out_channels(self) -> list[int]:
        if self.merge_method == "concat":
            return [self.single_embedding_shape * len(self.groups)] * len(self.out_indices)
        return [self.single_embedding_shape] * len(self.out_indices)

    def _preprocess_input(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            channel: F.interpolate(
                x[:, [band_index], :, :],
                (
                    int(self.ground_cover / self.channels[channel]["GSD"]),
                    int(self.ground_cover / self.channels[channel]["GSD"]),
                ),
                mode="bilinear",
            )
            for band_index, channel in zip(self.band_index, self.bands, strict=False)
        }

    def _merge_tokens_to_image_features(
        self,
        features: list[torch.Tensor],
        channel_params: dict[str, dict[str, Any]],
    ) -> list[torch.Tensor]:
        highest_num_patch = 0
        for params in channel_params.values():
            highest_num_patch = max(highest_num_patch, params["num_patch"])

        out_features = []
        for feature in features:
            start_idx = 0
            grouped = []
            for group_members in self.groups.values():
                member = next((m for m in group_members if m in channel_params), None)
                if member is None:
                    raise ValueError(
                        f"None of the group members {group_members} found in channel_params"
                    )

                num_patch = channel_params[member]["num_patch"]
                x_ = feature[:, start_idx : start_idx + num_patch**2, :].reshape(
                    -1, num_patch, num_patch, self.single_embedding_shape
                )
                x_ = x_.permute(0, 3, 1, 2)
                if num_patch != highest_num_patch:
                    x_ = F.interpolate(
                        x_,
                        size=(highest_num_patch, highest_num_patch),
                        mode="bilinear",
                    )

                grouped.append(x_)
                start_idx += num_patch**2

            if start_idx != feature.shape[1]:
                raise ValueError(
                    f"Number of patches used: {start_idx} does not match input shape {feature.shape[1]}"
                )

            if self.merge_method == "sum":
                out = torch.sum(torch.stack(grouped), dim=0)
            elif self.merge_method == "mean":
                out = torch.mean(torch.stack(grouped), dim=0)
            elif self.merge_method == "concat":
                out = torch.cat(grouped, dim=1)
            else:
                raise NotImplementedError(f"Unsupported merge_method: {self.merge_method}")

            out_features.append(out)

        return out_features

    def forward(
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> list[torch.Tensor] | tuple[list[torch.Tensor], dict[str, Any]]:
        x = self._preprocess_input(x)
        return_channel_params = self.return_channel_params or self.merge_method is not None
        features = self.model.forward_intermediates(
            x,
            indices=self.out_indices,
            intermediates_only=True,
            return_channel_params=return_channel_params,
        )

        if self.merge_method is None:
            return features

        token_features, channel_params = features
        merged_token_features = self._merge_tokens_to_image_features(
            token_features,
            channel_params,
        )
        if self.return_channel_params:
            return merged_token_features, channel_params
        return merged_token_features


def load_thor_model(
    model_name: str,
    model_bands: list[Any] | None = None,
    out_indices: list[int] | None = None,
    pretrained: bool = True,
    **kwargs: Any,
) -> THOREncoderWrapper:
    if model_name in name_mapping.values():
        model_name = next(key for key, value in name_mapping.items() if value == model_name)

    if model_bands is None:
        model_bands = [
            "COASTAL_AEROSOL",
            "BLUE",
            "GREEN",
            "RED",
            "NIR_BROAD",
            "RED_EDGE_1",
            "RED_EDGE_2",
            "RED_EDGE_3",
            "NIR_NARROW",
            "SWIR_1",
            "SWIR_2",
            "WATER_VAPOR",
            "VV",
            "VH",
        ]

    bands, channel_params_updated = process_thor_bands(model_bands)
    ckpt_path = kwargs.pop("ckpt", None)
    input_params = kwargs.pop("input_params", {})
    return_channel_params = kwargs.pop("return_channel_params", False)
    merge_method = kwargs.pop("merge_method", None)
    model_checkpoint_key = kwargs.pop("model_ckpt_type", "encoder")

    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected THOR loader kwargs: {unexpected}")

    default_input_params = deepcopy(_default_input_params)
    default_input_params["channels"].update(channel_params_updated)
    model_config = {
        "type": model_name,
        "strict": False,
        "ckpt": None,
        "ckpt_ignore": [
            "decoder_.*",
            "decode_.*",
            "ref_pos_embed.*",
            "pos_embed.*",
        ],
        "input_params": default_input_params,
    }

    if ckpt_path is not None:
        model_config["ckpt"] = ckpt_path

    if input_params:
        _ensure_allowed_input_params(input_params.keys())
        for key, value in input_params.items():
            if (
                key in model_config["input_params"]
                and model_config["input_params"][key] != value
            ):
                warnings.warn(
                    f"Overwriting input param {key} for model {model_checkpoint_key} "
                    f"from {model_config['input_params'][key]} to {value}",
                    stacklevel=2,
                )
            model_config["input_params"][key] = value

    if pretrained and model_config["ckpt"] is None:
        from huggingface_hub import hf_hub_download

        variant = name_mapping[model_name]
        model_config["ckpt"] = hf_hub_download(
            repo_id=pretrained_weights[variant]["hf_hub_id"],
            filename=pretrained_weights[variant]["hf_hub_filename"],
        )

    model = _get_vendored_model_registry().build(
        model_cfgs={model_checkpoint_key: model_config}
    )[model_checkpoint_key]
    return THOREncoderWrapper(
        model=model,
        bands=bands,
        out_indices=out_indices,
        return_channel_params=return_channel_params,
        merge_method=merge_method,
    )


__all__ = ["THOREncoderWrapper", "load_thor_model", "lookup_band", "process_thor_bands"]

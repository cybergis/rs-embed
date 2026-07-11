"""Vendored Clay foundation model (encoder inference path only).

Source: https://github.com/Clay-foundation/model @ f14e698 (Apache-2.0).
Weights: https://huggingface.co/made-with-clay/Clay (v1.5).
Only the modules needed to run the pretrained encoder are vendored:
``backbone`` (ViT Transformer), ``factory`` (wavelength-conditioned
DynamicEmbedding), ``utils`` (sin/cos position embeddings) and ``model``
(the Encoder). Decoder, teacher distillation and Lightning training code
are intentionally omitted. See LICENSE.clay.
"""

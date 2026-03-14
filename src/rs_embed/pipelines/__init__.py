"""
Workflow Orchestration and Execution.

This module contains the stateful "Verbs" of the system.  It wires together
Providers (data) and Embedders (models) to perform work.

Key Components
--------------
- ``InferenceEngine``  — manages model loading, threading, and batch inference.
- ``BatchExporter``    — orchestrates the end-to-end flow
  (Prefetch → Tile → Infer → Save).
- ``ParallelRunner``   — handles concurrency, retries, and error suppression.
- ``PrefetchManager``  — pre-downloads imagery ahead of inference.
- ``CheckpointManager`` — tracks progress so interrupted jobs can resume.

Use this module for high-level process control, performance optimisation, and
loop management.
"""

from .checkpoint import CheckpointManager
from .exporter import BatchExporter
from .inference import InferenceEngine
from .prefetch import PrefetchManager
from .runner import ParallelRunner, run_with_retry

__all__ = [
    "BatchExporter",
    "CheckpointManager",
    "InferenceEngine",
    "ParallelRunner",
    "PrefetchManager",
    "run_with_retry",
]

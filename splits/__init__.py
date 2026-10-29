"""Evaluation protocol split generators (P1, P2, P4) and split persistence."""
from splits.protocols import (
    SplitIndices,
    generate_p1_lofo_splits,
    generate_p2_temporal_splits,
    generate_p4_contamination_splits,
    save_splits,
    load_splits,
)

__all__ = [
    "SplitIndices",
    "generate_p1_lofo_splits",
    "generate_p2_temporal_splits",
    "generate_p4_contamination_splits",
    "save_splits",
    "load_splits",
]

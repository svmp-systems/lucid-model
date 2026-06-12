"""Training package — checkpoints, loop, modules, corpus, validation.

Layout
------
cli.py          ``lucid train`` entrypoint
checkpoint/     weight stores, save points, ``lucid checkpoint``
loop/           orchestrator + pipeline bridge + promotion hook
modules/        per-module trainers (registry)
corpus/         episode adapters + ``lucid-gen`` generator
learn/          DMF learning hooks + quantization measurement
validate/       gold scoring
tests/          pytest sources (committed)
tree/           local artifacts only — gitignored (checkpoints, audit, data)
"""

from lucid.training.learn.quant import (
    QuantizationMeasurementResult,
    RetrievalQualitySample,
    binary_signature,
    measure_candidate_quality,
    rank_by_popcount,
)

__all__ = [
    "QuantizationMeasurementResult",
    "RetrievalQualitySample",
    "binary_signature",
    "measure_candidate_quality",
    "rank_by_popcount",
]

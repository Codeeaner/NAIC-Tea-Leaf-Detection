"""Models package initialization."""

from .detection import DetectionSession, DetectionResult, DetectionBox
from .analytics import (
    AnalysisResult,
    BatchAnalysisResult,
    WastePreventionRecommendation,
    QualityMetric,
    ProcessingAction
)

__all__ = [
    "DetectionSession",
    "DetectionResult", 
    "DetectionBox",
    "AnalysisResult",
    "BatchAnalysisResult",
    "WastePreventionRecommendation",
    "QualityMetric",
    "ProcessingAction"
]
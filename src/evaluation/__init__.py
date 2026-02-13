"""Evaluation package for DERM-EQUITY"""

from .metrics import (
    comprehensive_evaluation,
    compute_auc_roc,
    expected_calibration_error,
    auc_gap_across_groups,
    print_evaluation_report,
)

__all__ = [
    'comprehensive_evaluation',
    'compute_auc_roc',
    'expected_calibration_error',
    'auc_gap_across_groups',
    'print_evaluation_report',
]

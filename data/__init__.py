# data/__init__.py
from .data_loader import load_california_housing, basic_eda
from .preprocessing import (
    get_features_and_target,
    split_data,
    build_preprocessor,
    validate_input,
    FEATURE_COLS,
    TARGET_COL,
)

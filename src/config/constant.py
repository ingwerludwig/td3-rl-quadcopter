from enum import Enum

NUM_EPISODE = 100
MAX_STEP_PER_EPISODE = 500
DATASET_FILENAME = "lqr_trajectories.json"
DATASET_VALIDATION_FILENAME = "val_lqr_trajectories.json"
LOG_FILENAME = "result.log"

class DatasetType(Enum):
    LINE_TYPE = "line"
    S_CURVE_TYPE = "s_curve"
    HOVER_TYPE = "hover"
    STEP_TYPE = "step"
    CIRCLE_TYPE = "circle"
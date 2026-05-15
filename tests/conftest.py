import sys
from unittest.mock import MagicMock

# Mock heavy ML dependencies before pipeline.py imports trigger them.
# These libraries require native extensions and model weights not present
# in the unit test environment. Tests that actually exercise model
# inference belong in a separate integration test suite.
for _mod in [
    "cv2",
    "insightface",
    "insightface.app",
    "insightface.model_zoo",
    "insightface.utils",
    "insightface.utils.face_align",
    "onnxruntime",
]:
    sys.modules.setdefault(_mod, MagicMock())

"""Debug: Check the web_collector gaze ratio computation using a saved frame from the server."""
import sys, os, math
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

# Import the actual function used by the running server
from web_collector import (
    compute_gaze_ratio,
    LEFT_IRIS, RIGHT_IRIS,
    LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR,
    LEFT_EYE_INNER, LEFT_EYE_OUTER,
    LEFT_EYE_TOP, LEFT_EYE_BOTTOM,
    RIGHT_EYE_INNER, RIGHT_EYE_OUTER,
    RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM,
)
import inspect

print("=== Verifying compute_gaze_ratio signature ===")
sig = inspect.signature(compute_gaze_ratio)
print(f"Parameters: {list(sig.parameters.keys())}")
print(f"\nSource code:")
print(inspect.getsource(compute_gaze_ratio))

print(f"\n=== Contour arrays ===")
print(f"LEFT_EYE_CONTOUR: {LEFT_EYE_CONTOUR}")
print(f"RIGHT_EYE_CONTOUR: {RIGHT_EYE_CONTOUR}")

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import pytest

from hdrmerge import writers

#: Skips a test that actually writes EXR. OpenEXR is an optional dependency --
#: it has no wheel for Python 3.14 -- so the suite has to stay green without it.
#: Only apply this to tests that reach the writer: anything that fails earlier,
#: in validation, must keep running so the degraded path stays covered.
requires_openexr = pytest.mark.skipif(
    not writers.available("exr"),
    reason="OpenEXR is not installed (optional dependency)",
)

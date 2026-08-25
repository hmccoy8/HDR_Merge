"""hdrmerge -- merge bracketed exposures into an HDR photograph.

    from hdrmerge import merge_bracket, MergeOptions, Adjustments

    result = merge_bracket(
        ["IMG_0001.CR2", "IMG_0002.CR2", "IMG_0003.CR2"],
        MergeOptions(formats=["jpg", "exr"], adjustments=Adjustments.auto()),
    )
    print(result.written)
"""

from .adjust import Adjustments
from .grouping import Bracket, group_frames
from .loaders import LoadError, Frame, load_bracket, load_frame
from .merge import MergeError, MergeStats, exposure_fusion, merge_radiance
from .metadata import FrameMeta, read_exif
from .pipeline import MergeFallback, MergeOptions, MergeResult, merge_bracket
from .writers import FORMATS, WriteError

__version__ = "0.1.0"

__all__ = [
    "Adjustments",
    "Bracket",
    "FORMATS",
    "Frame",
    "FrameMeta",
    "LoadError",
    "MergeError",
    "MergeFallback",
    "MergeOptions",
    "MergeResult",
    "MergeStats",
    "WriteError",
    "__version__",
    "exposure_fusion",
    "group_frames",
    "load_bracket",
    "load_frame",
    "merge_bracket",
    "merge_radiance",
    "read_exif",
]

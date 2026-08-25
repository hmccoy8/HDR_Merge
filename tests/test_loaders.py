"""Reading frames in, and the guards on what makes a valid bracket."""

import os

import numpy as np
import pytest
import synthetic
from PIL import Image

from hdrmerge.loaders import (
    LoadError,
    expand_inputs,
    is_raw_path,
    is_supported,
    load_bracket,
    load_frame,
)


@pytest.fixture
def bracket(tmp_path):
    return synthetic.write_jpeg_bracket(str(tmp_path))


def test_a_jpeg_loads_as_float_rgb_in_range(bracket):
    frame = load_frame(bracket[0])

    assert frame.data.dtype == np.float32
    assert frame.data.shape == (96, 128, 3)
    assert 0.0 <= frame.data.min() and frame.data.max() <= 1.0
    assert not frame.is_raw and not frame.is_linear


def test_sixteen_bit_input_keeps_its_precision(tmp_path):
    """An 8-bit read of a 16-bit file would quantise away the shadow detail."""
    import tifffile

    values = np.linspace(0, 65535, 96 * 128 * 3, dtype=np.uint16).reshape(96, 128, 3)
    path = str(tmp_path / "deep.tif")
    tifffile.imwrite(path, values, photometric="rgb")

    data = load_frame(path).data

    assert len(np.unique(data)) > 300, "8-bit input would give at most 256 levels"
    assert data.max() == pytest.approx(1.0, abs=1e-4)


def test_greyscale_input_becomes_three_channels(tmp_path):
    path = str(tmp_path / "grey.png")
    Image.fromarray(np.full((48, 64), 128, dtype=np.uint8), "L").save(path)

    data = load_frame(path).data

    assert data.shape == (48, 64, 3)
    assert np.allclose(data[..., 0], data[..., 1])


def test_an_alpha_channel_is_dropped_without_disturbing_the_colours(tmp_path):
    """Alpha has to come off before any channel reordering, not after."""
    path = str(tmp_path / "alpha.png")
    rgba = np.zeros((48, 64, 4), dtype=np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2], rgba[..., 3] = 240, 120, 30, 255
    Image.fromarray(rgba, "RGBA").save(path)

    data = load_frame(path).data

    assert data.shape == (48, 64, 3)
    assert data[0, 0, 0] == pytest.approx(240 / 255, abs=0.01)
    assert data[0, 0, 1] == pytest.approx(120 / 255, abs=0.01)
    assert data[0, 0, 2] == pytest.approx(30 / 255, abs=0.01)


def test_channel_order_survives_a_sixteen_bit_png(tmp_path):
    import cv2

    path = str(tmp_path / "deep.png")
    rgb = np.zeros((48, 64, 3), dtype=np.uint16)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 60000, 30000, 5000
    cv2.imwrite(path, rgb[..., ::-1])          # OpenCV writes BGR

    data = load_frame(path).data

    assert data[0, 0, 0] > data[0, 0, 1] > data[0, 0, 2]
    assert data[0, 0, 0] == pytest.approx(60000 / 65535, abs=0.01)


def test_preview_scale_downsamples_on_load(bracket):
    assert load_frame(bracket[0], preview_scale=0.25).data.shape == (24, 32, 3)


def test_preview_scale_must_be_a_fraction(bracket):
    with pytest.raises(LoadError, match="preview-scale"):
        load_frame(bracket[0], preview_scale=2.0)


def test_raw_extensions_are_recognised():
    assert is_raw_path("shot.CR2") and is_raw_path("shot.nef") and is_raw_path("x.dng")
    assert not is_raw_path("shot.jpg")
    assert is_supported("a.JPG") and is_supported("b.tiff")
    assert not is_supported("notes.txt")


def test_loading_a_bracket_returns_every_frame(bracket):
    frames = load_bracket(bracket)

    assert len(frames) == 3
    assert all(frame.shape == (96, 128) for frame in frames)
    assert [frame.meta.name for frame in frames] == [os.path.basename(p) for p in bracket]


def test_a_bracket_needs_two_frames(bracket):
    with pytest.raises(LoadError, match="at least 2 frames"):
        load_bracket(bracket[:1])


def test_max_frames_is_enforced(bracket):
    with pytest.raises(LoadError, match="max-frames"):
        load_bracket(bracket, max_frames=2)


def test_mismatched_sizes_are_refused(tmp_path, bracket):
    with Image.open(bracket[-1]) as img:
        img.resize((64, 48)).save(bracket[-1])

    with pytest.raises(LoadError, match="same dimensions"):
        load_bracket(bracket)


def test_mixing_raw_and_rendered_frames_is_refused(tmp_path, bracket, monkeypatch):
    """The two have incompatible linearity, so averaging them is meaningless."""
    import hdrmerge.loaders as loaders

    real = loaders.load_frame

    def pretend_raw(path, preview_scale=1.0):
        frame = real(path, preview_scale)
        if path.endswith("0003.jpg"):
            frame.is_raw = True
        return frame

    monkeypatch.setattr(loaders, "load_frame", pretend_raw)

    with pytest.raises(LoadError, match="Cannot mix RAW and rendered"):
        loaders.load_bracket(bracket)


def test_a_corrupt_file_names_itself_in_the_error(tmp_path):
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"not a jpeg at all")

    with pytest.raises(LoadError, match="broken.jpg"):
        load_frame(str(path))


def test_expand_inputs_collects_images_from_a_folder(tmp_path, bracket):
    (tmp_path / "readme.txt").write_text("ignore me")

    found = expand_inputs([str(tmp_path)])

    assert len(found) == 3
    assert all(path.endswith(".jpg") for path in found)


def test_expand_inputs_accepts_a_mix_of_files_and_folders(tmp_path, bracket):
    found = expand_inputs([bracket[0], str(tmp_path)])

    assert len(found) == 4       # the explicit file, plus the folder's three


def test_expand_inputs_rejects_a_missing_path(tmp_path):
    with pytest.raises(LoadError, match="No such file or directory"):
        expand_inputs([str(tmp_path / "ghost")])

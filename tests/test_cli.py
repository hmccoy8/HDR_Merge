"""The command-line interface, driven the way a user would drive it."""

import datetime as dt
import os

import pytest
import synthetic

from hdrmerge.cli import main


@pytest.fixture
def bracket(tmp_path):
    return synthetic.write_jpeg_bracket(str(tmp_path / "shoot"))


@pytest.fixture
def shoot(tmp_path):
    """Three brackets: two separated by a pause, two shot back-to-back."""
    start = dt.datetime(2026, 4, 12, 9, 30, 0)
    directory = str(tmp_path / "shoot")
    paths = []
    for index, offset in enumerate([0.0, 8.0, 9.2]):
        paths += synthetic.write_jpeg_bracket(
            directory, start=start + dt.timedelta(seconds=offset), index_from=1 + index * 3
        )
    return directory, paths


def test_merge_writes_an_output(tmp_path, bracket):
    out = str(tmp_path / "out")

    assert main(["merge", *bracket, "--out-dir", out]) == 0
    assert os.listdir(out) == ["IMG_0001_hdr.jpg"]


def test_repeated_format_flags_write_every_format(tmp_path, bracket):
    out = str(tmp_path / "out")

    code = main(["merge", *bracket, "-o", out, "-f", "jpg", "-f", "exr", "-f", "tif32"])

    assert code == 0
    assert sorted(os.listdir(out)) == [
        "IMG_0001_hdr.exr", "IMG_0001_hdr.jpg", "IMG_0001_hdr_linear.tif"
    ]


def test_a_folder_is_split_into_brackets_and_merged_separately(tmp_path, shoot):
    directory, paths = shoot
    out = str(tmp_path / "out")

    assert main(["merge", directory, "-o", out]) == 0
    assert sorted(os.listdir(out)) == [
        "IMG_0001_hdr.jpg", "IMG_0004_hdr.jpg", "IMG_0007_hdr.jpg"
    ]


def test_an_explicit_file_list_is_treated_as_one_bracket(tmp_path, shoot):
    """Naming files means "merge these", not "guess what I meant"."""
    directory, paths = shoot
    out = str(tmp_path / "out")

    assert main(["merge", *paths[:6], "-o", out]) == 0
    assert os.listdir(out) == ["IMG_0001_hdr.jpg"]


def test_bracket_size_overrides_detection(tmp_path, shoot):
    directory, paths = shoot
    out = str(tmp_path / "out")

    assert main(["merge", *paths, "-o", out, "--bracket-size", "3"]) == 0
    assert len(os.listdir(out)) == 3


def test_auto_produces_a_different_image_than_the_plain_merge(tmp_path, bracket):
    plain, auto = str(tmp_path / "plain"), str(tmp_path / "auto")

    assert main(["merge", *bracket, "-o", plain]) == 0
    assert main(["merge", *bracket, "-o", auto, "--auto"]) == 0

    with open(os.path.join(plain, "IMG_0001_hdr.jpg"), "rb") as handle:
        first = handle.read()
    with open(os.path.join(auto, "IMG_0001_hdr.jpg"), "rb") as handle:
        second = handle.read()
    assert first != second


def test_adjustment_flags_are_accepted(tmp_path, bracket):
    code = main([
        "merge", *bracket, "-o", str(tmp_path / "out"),
        "--exposure", "0.5", "--shadows", "30", "--highlights", "-20",
        "--contrast", "10", "--vibrance", "25", "--temp", "10", "--tint", "-5",
        "--whites", "5", "--blacks", "-5", "--saturation", "5",
        "--output-gamma", "1.1",
    ])

    assert code == 0


@pytest.mark.parametrize("operator", ["reinhard", "drago", "mantiuk", "linear", "fusion"])
def test_every_tonemap_choice_runs(tmp_path, bracket, operator):
    out = str(tmp_path / operator)

    assert main(["merge", *bracket, "-o", out, "-t", operator]) == 0
    assert len(os.listdir(out)) == 1


@pytest.mark.parametrize("method", ["mtb", "ecc", "none"])
def test_every_alignment_choice_runs(tmp_path, bracket, method):
    out = str(tmp_path / method)

    assert main(["merge", *bracket, "-o", out, "--align", method]) == 0
    assert len(os.listdir(out)) == 1


def test_deghosting_can_be_switched_off(tmp_path, bracket):
    out = str(tmp_path / "out")

    assert main(["merge", *bracket, "-o", out, "--deghost", "none"]) == 0


def test_preview_scale_produces_a_smaller_image(tmp_path, bracket):
    from PIL import Image

    out = str(tmp_path / "out")
    assert main(["merge", *bracket, "-o", out, "--preview-scale", "0.5"]) == 0

    with Image.open(os.path.join(out, "IMG_0001_hdr.jpg")) as img:
        assert img.size == (64, 48)


def test_fusion_with_a_radiance_format_fails_with_a_clear_message(tmp_path, bracket, capsys):
    code = main(["merge", *bracket, "-o", str(tmp_path / "out"), "-t", "fusion", "-f", "exr"])

    assert code == 1
    assert "never builds a radiance map" in capsys.readouterr().err


def test_a_missing_file_is_reported_not_crashed_on(tmp_path, capsys):
    code = main(["merge", str(tmp_path / "nope.jpg"), str(tmp_path / "nope2.jpg")])

    assert code == 1
    assert "No such file" in capsys.readouterr().err


def test_an_unsupported_file_type_lists_what_is_supported(tmp_path, capsys):
    path = tmp_path / "notes.txt"
    path.write_text("hello")

    code = main(["merge", str(path)])

    assert code == 1
    assert ".cr2" in capsys.readouterr().err


def test_inspect_reports_each_frames_exposure(bracket, capsys):
    assert main(["inspect", *bracket]) == 0

    out = capsys.readouterr().out
    assert "1/200s f/8 ISO100" in out
    assert "exif" in out
    assert "4.0 EV span" in out
    assert "steps: 2.00, 2.00" in out


def test_inspect_can_cross_check_exif_against_the_pixels(bracket, capsys):
    assert main(["inspect", *bracket, "--pixels"]) == 0

    assert "pixels:" in capsys.readouterr().out


def test_group_shows_the_detected_brackets_without_merging(tmp_path, shoot, capsys):
    directory, paths = shoot

    assert main(["group", directory]) == 0

    out = capsys.readouterr().out
    assert "3 bracket(s) detected" in out
    assert "IMG_0001.jpg" in out
    assert not (tmp_path / "out").exists()


def test_help_lists_the_output_formats(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])

    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert all(fmt in out for fmt in ("jpg", "png16", "tif16", "tif32", "exr", "hdr"))


def test_version_is_reported(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])

    assert "hdrmerge" in capsys.readouterr().out

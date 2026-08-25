"""GUI tests, run against Qt's offscreen platform.

These exercise the window the way a user drives it -- drop files, move
sliders, press Export -- rather than poking at internals, because the parts
worth protecting are the wiring: that a merge happens off the UI thread, that
grading reuses a cached scene instead of re-merging, and that every command-line
option is actually reachable.
"""

from __future__ import annotations

import os
import time

import pytest
import synthetic

pytest.importorskip("PySide6", reason="the GUI extra is not installed")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from hdrmerge.gui import theme  # noqa: E402
from hdrmerge.gui.controls import LabelledSlider  # noqa: E402
from hdrmerge.gui.thumbnails import load_thumbnail  # noqa: E402
from hdrmerge.gui.window import MainWindow  # noqa: E402


@pytest.fixture(scope="session")
def app():
    instance = QApplication.instance() or QApplication([])
    theme.apply(instance)
    return instance


@pytest.fixture
def window(app, tmp_path):
    win = MainWindow()
    win.resize(1200, 800)
    # Shown, so that isVisible() means what it says: Qt reports children of an
    # unshown window as invisible regardless of their own state, which would
    # make every visibility assertion here pass for the wrong reason.
    win.show()
    yield win
    win.close()


@pytest.fixture
def bracket(tmp_path):
    return synthetic.write_jpeg_bracket(str(tmp_path / "shoot"))


def settle(app, window, seconds: float = 20.0, want_image: bool = True) -> None:
    """Pump the event loop until background work is done.

    processEvents returns immediately when the queue is empty, so waiting on
    the thread pool is what actually blocks -- a bare processEvents loop spins
    and proves nothing. A merge is followed by a render, so waiting only for
    the scene would sample the preview before it has been drawn.
    """
    deadline = time.time() + seconds
    while time.time() < deadline:
        done = window.jobs.wait(50)
        app.processEvents(QEventLoop.AllEvents, 50)
        ready = window._scene is not None and (
            window.preview.canvas.has_image() or not want_image
        )
        if done and ready:
            app.processEvents(QEventLoop.AllEvents, 50)
            return
    raise AssertionError("background work did not finish in time")


def test_the_window_opens_on_the_drop_zone(window):
    assert window.preview.currentWidget() is window.preview.drop_zone
    assert not window.frames_panel.isVisible()
    assert not window.export_button.isEnabled()


def test_dropping_a_bracket_merges_and_previews_it(app, window, bracket):
    window.load_paths(bracket)
    settle(app, window)

    assert window._scene is not None
    assert window.preview.canvas.has_image()
    assert window.export_button.isEnabled()
    assert "3 frames" in window.status.text()
    assert "stops recovered" in window.status.text()


def test_the_filmstrip_lists_every_frame_darkest_first(app, window, bracket):
    window.load_paths(bracket)

    metas = window.filmstrip.metas
    assert [m.name for m in metas] == ["IMG_0001.jpg", "IMG_0002.jpg", "IMG_0003.jpg"]
    # Darkest first, matching the merge order and the reference-frame numbers.
    # A darker frame received less light, so its EV is *higher*.
    evs = [round(m.ev, 2) for m in metas]
    assert evs == sorted(evs, reverse=True)
    assert metas[0].relative_exposure < metas[-1].relative_exposure
    assert window.frames_panel.isVisible()


def test_grading_reuses_the_merge_instead_of_redoing_it(app, window, bracket):
    """The whole point of the two-stage split: sliders must not re-merge."""
    window.load_paths(bracket)
    settle(app, window)
    scene = window._scene

    window.tone_panel.sliders["shadows"].set_value(40)
    window.tone_panel.apply_auto()
    settle(app, window)

    assert window._scene is scene, "a grading change must not trigger a re-merge"


def test_changing_a_merge_setting_does_remerge(app, window, bracket):
    window.load_paths(bracket)
    settle(app, window)
    scene = window._scene

    window.merge_panel.align.setCurrentText("none")
    settle(app, window)

    assert window._scene is not scene, "alignment is structural and must re-merge"


def test_switching_to_fusion_re_merges_and_blocks_radiance_formats(app, window, bracket):
    """Fusion never builds a radiance map, so EXR must become unavailable."""
    window.load_paths(bracket)
    settle(app, window)

    window.tone_panel.tonemap.setCurrentText("fusion")
    settle(app, window)

    assert window._scene.is_fusion
    assert window._scene.radiance is None
    assert not window.output_panel.formats["exr"].isChecked()
    assert not window.output_panel.formats["tif32"].isEnabled()


def test_auto_matches_the_command_line_preset(app, window, bracket):
    from hdrmerge.adjust import Adjustments

    window.load_paths(bracket)
    window.tone_panel.apply_auto()

    adjustments = window.tone_panel.adjustments()
    for name, value in Adjustments.AUTO.items():
        assert getattr(adjustments, name) == value


def test_reset_returns_every_slider_to_zero(window, bracket):
    window.load_paths(bracket)
    window.tone_panel.apply_auto()
    window.tone_panel.reset_all()

    assert window.tone_panel.adjustments().is_identity()


def test_the_options_the_gui_builds_match_the_panels(app, window, bracket):
    window.load_paths(bracket)
    window.merge_panel.align.setCurrentText("ecc")
    window.merge_panel.crf.setCurrentText("srgb")
    window.merge_panel.deghost.setChecked(False)
    window.output_panel.formats["tif16"].setChecked(True)
    window.output_panel.suffix.setText("_merged")
    window.tone_panel.tonemap.setCurrentText("drago")

    options = window.options(for_export=True)

    assert options.align_method == "ecc"
    assert options.crf_method == "srgb"
    assert options.deghost_threshold == 0.0
    assert "tif16" in options.formats
    assert options.suffix == "_merged"
    assert options.tonemap == "drago"
    assert options.preview_scale == 1.0, "exports always run at full resolution"


def test_preview_runs_at_reduced_scale_but_export_does_not(window, bracket):
    window.load_paths(bracket)

    assert window.options().preview_scale < 1.0
    assert window.options(for_export=True).preview_scale == 1.0


def test_export_writes_the_selected_formats(app, window, bracket, tmp_path):
    out = tmp_path / "out"
    window.load_paths(bracket)
    settle(app, window)
    window.output_panel.out_dir.setText(str(out))
    window.output_panel.formats["tif16"].setChecked(True)

    written = []
    window._on_export_done = lambda token, paths: written.extend(paths)
    window.export()
    window.jobs.wait(60_000)
    app.processEvents(QEventLoop.AllEvents, 200)

    assert sorted(os.path.basename(p) for p in written) == [
        "IMG_0001_hdr.jpg", "IMG_0001_hdr.tif"
    ]
    assert all(os.path.getsize(p) > 0 for p in written)


def test_a_folder_is_split_into_brackets(app, window, tmp_path):
    import datetime as dt

    directory = str(tmp_path / "shoot")
    start = dt.datetime(2026, 4, 12, 9, 30)
    for index, offset in enumerate([0.0, 10.0, 20.0]):
        synthetic.write_jpeg_bracket(
            directory, start=start + dt.timedelta(seconds=offset), index_from=1 + index * 3
        )

    window.load_paths([directory])
    settle(app, window)

    assert len(window._brackets) == 3
    # The export hint carries the count persistently; the status line is free
    # to show merge progress over it.
    assert "3 brackets" in window.export_hint.text()
    assert "3 brackets queued" in window.status.text()


def test_a_single_image_is_refused_with_an_explanation(window, bracket, monkeypatch):
    seen = {}
    monkeypatch.setattr(window, "_error", lambda title, message: seen.update(t=title, m=message))

    window.load_paths(bracket[:1])

    assert "Not a bracket" in seen["t"]
    assert "two frames" in seen["m"]


def test_every_tonemap_operator_and_align_method_is_offered(window):
    from hdrmerge.align import METHODS as ALIGN_METHODS
    from hdrmerge.crf import METHODS as CRF_METHODS
    from hdrmerge.tonemap import OPERATORS
    from hdrmerge.writers import FORMATS

    offered = lambda box: [box.itemText(i) for i in range(box.count())]
    assert offered(window.tone_panel.tonemap) == list(OPERATORS)
    assert offered(window.merge_panel.align) == list(ALIGN_METHODS)
    assert offered(window.merge_panel.crf) == list(CRF_METHODS)
    assert set(window.output_panel.formats) == set(FORMATS)


def test_every_adjustment_slider_from_the_cli_is_present(window):
    from dataclasses import fields

    from hdrmerge.adjust import Adjustments

    # Every float field on Adjustments needs a control; gamma has its own
    # spin box rather than a slider.
    expected = {f.name for f in fields(Adjustments) if isinstance(f.default, float)}
    covered = set(window.tone_panel.sliders) | {"gamma"}
    assert expected == covered


def test_a_slider_reports_its_value_in_real_units():
    slider = LabelledSlider("Exposure", -50, 50, 0, scale=0.1, suffix=" EV")

    slider.set_value(1.5)

    assert slider.value() == pytest.approx(1.5)
    assert "+1.50" in slider.readout.text()
    slider.reset()
    assert slider.value() == 0 and slider.is_default()


def test_thumbnails_load_and_a_bad_file_returns_none(bracket, tmp_path):
    thumbnail = load_thumbnail(bracket[0], longest_edge=64)

    assert thumbnail is not None
    assert max(thumbnail.shape[:2]) <= 64
    assert load_thumbnail(str(tmp_path / "missing.jpg")) is None


def test_clearing_returns_to_the_drop_zone(app, window, bracket):
    window.load_paths(bracket)
    settle(app, window)

    window.clear()

    assert window.preview.currentWidget() is window.preview.drop_zone
    assert not window.frames_panel.isVisible()
    assert not window.export_button.isEnabled()
    assert window._scene is None

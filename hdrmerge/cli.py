"""Command-line interface.

    hdrmerge merge  IMG_*.CR2 --format jpg --format exr --auto
    hdrmerge inspect IMG_*.CR2
    hdrmerge group  ./shoot
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from typing import List, Optional, Sequence

from . import __version__
from .adjust import Adjustments
from .align import METHODS as ALIGN_METHODS
from .crf import METHODS as CRF_METHODS
from .grouping import Bracket, group_frames, output_base
from .loaders import LoadError, expand_inputs, load_frame
from .metadata import estimate_relative_exposures, read_exif, resolve_exposures
from .pipeline import MergeFallback, MergeOptions, merge_bracket
from .tonemap import OPERATORS as TONEMAP_OPERATORS
from .writers import FORMATS, WriteError, available

log = logging.getLogger("hdrmerge")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hdrmerge",
        description="Merge bracketed exposures into an HDR photograph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_epilog(),
    )
    parser.add_argument("--version", action="version", version=f"hdrmerge {__version__}")
    _add_verbosity(parser)

    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_merge(subparsers.add_parser(
        "merge", help="merge a bracket (or a folder of brackets) into HDR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    ))
    _add_inspect(subparsers.add_parser(
        "inspect", help="show the detected exposure of each frame",
    ))
    _add_group(subparsers.add_parser(
        "group", help="show how frames would be split into brackets, without merging",
    ))
    return parser


def _epilog() -> str:
    formats = "\n".join(
        f"    {key:<6} {spec[2]}" + ("" if available(key) else "  [unavailable here]")
        for key, spec in FORMATS.items()
    )
    return (
        "output formats:\n" + formats + "\n\n"
        "examples:\n"
        "    hdrmerge merge IMG_000{1,2,3}.CR2 --auto\n"
        "    hdrmerge merge bracket/*.jpg --format tif16 --tonemap fusion\n"
        "    hdrmerge merge ./shoot --format jpg --out-dir ./merged\n"
        "    hdrmerge inspect IMG_*.CR2\n"
    )


def _add_verbosity(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-v", "--verbose", action="store_true", help="show detailed progress")
    parser.add_argument("-q", "--quiet", action="store_true", help="only show errors")


def _add_merge(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("inputs", nargs="+", metavar="IMAGE",
                        help="bracket frames, or a folder to auto-group into brackets")

    output = parser.add_argument_group("output")
    output.add_argument("-f", "--format", dest="formats", action="append",
                        choices=sorted(FORMATS), metavar="FMT",
                        help=f"output format, repeatable ({', '.join(sorted(FORMATS))})")
    output.add_argument("-o", "--out-dir", help="write outputs here instead of alongside the inputs")
    output.add_argument("--suffix", default="_hdr", help="appended to the output filename")
    output.add_argument("--jpeg-quality", type=int, default=95, metavar="N",
                        help="JPEG quality, 1-100")

    merge = parser.add_argument_group("merge")
    merge.add_argument("--align", dest="align_method", default="mtb", choices=ALIGN_METHODS,
                       help="frame alignment: mtb (exposure-invariant), ecc (also fixes "
                            "slight rotation, slower), none (tripod)")
    merge.add_argument("--deghost-threshold", type=float, default=0.7, metavar="STOPS",
                       help="reject pixels disagreeing with the reference by more than "
                            "this many stops; 0 disables deghosting")
    merge.add_argument("--deghost", choices=("auto", "none"), default="auto",
                       help="shorthand: 'none' disables deghosting entirely")
    merge.add_argument("--crf", dest="crf_method", default="auto", choices=CRF_METHODS,
                       help="how to linearise rendered frames (RAW is already linear)")
    merge.add_argument("--reference", type=int, metavar="N",
                       help="index of the reference frame, 0-based, darkest first "
                            "(default: the middle exposure)")
    merge.add_argument("--max-frames", type=int, default=15, metavar="N",
                       help="refuse brackets larger than this, as a memory guard")
    merge.add_argument("--preview-scale", type=float, default=1.0, metavar="S",
                       help="downscale on load (e.g. 0.25) for fast iteration on settings")
    merge.add_argument("--bracket-size", type=int, metavar="N",
                       help="with a folder input, split into fixed-size brackets "
                            "instead of auto-detecting them")

    post = parser.add_argument_group("post-processing")
    post.add_argument("-t", "--tonemap", default="reinhard", choices=TONEMAP_OPERATORS,
                      help="tone-mapping operator; 'fusion' blends the frames directly "
                           "and cannot produce radiance formats")
    post.add_argument("--gamma", type=float, default=2.2, help="tone-mapping gamma")
    post.add_argument("--auto", action="store_true",
                      help="apply a tuned default grade (mild shadow lift, highlight "
                           "recovery, contrast and vibrance)")
    for name, help_text in (
        ("exposure", "exposure in stops, applied to linear radiance"),
        ("temp", "white balance, -100 (cooler) to 100 (warmer)"),
        ("tint", "green/magenta balance, -100 to 100"),
        ("highlights", "recover (-) or lift (+) highlights, -100 to 100"),
        ("shadows", "lift (+) or deepen (-) shadows, -100 to 100"),
        ("whites", "white point, -100 to 100"),
        ("blacks", "black point, -100 to 100"),
        ("contrast", "contrast, -100 to 100"),
        ("saturation", "saturation, -100 to 100"),
        ("vibrance", "saturation weighted towards muted colours, -100 to 100"),
    ):
        post.add_argument(f"--{name}", type=float, metavar="N", help=help_text)
    post.add_argument("--output-gamma", dest="out_gamma", type=float, metavar="G",
                      help="final gamma trim applied after every other adjustment")


def _add_inspect(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("inputs", nargs="+", metavar="IMAGE")
    parser.add_argument("--pixels", action="store_true",
                        help="also estimate exposures from pixel data, to cross-check EXIF")


def _add_group(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("inputs", nargs="+", metavar="IMAGE")
    parser.add_argument("--max-gap", type=float, default=2.0, metavar="SECONDS",
                        help="minimum pause that starts a new bracket")
    parser.add_argument("--bracket-size", type=int, metavar="N",
                        help="split into fixed-size brackets instead of auto-detecting")


def _configure_logging(args) -> None:
    level = logging.WARNING if args.quiet else (logging.DEBUG if args.verbose else logging.INFO)
    # force=True so -v and -q still work if something else configured logging
    # first; basicConfig is otherwise a no-op once the root logger has handlers.
    logging.basicConfig(level=level, format="%(message)s", stream=sys.stderr, force=True)


def _adjustments(args) -> Adjustments:
    """Build the adjustment set, with explicit flags overriding --auto."""
    explicit = {
        name: getattr(args, name)
        for name in ("exposure", "temp", "tint", "highlights", "shadows", "whites",
                     "blacks", "contrast", "saturation", "vibrance")
        if getattr(args, name, None) is not None
    }
    if getattr(args, "out_gamma", None) is not None:
        explicit["gamma"] = args.out_gamma

    return Adjustments.auto(**explicit) if args.auto else Adjustments(**explicit)


def _options(args) -> MergeOptions:
    return MergeOptions(
        preview_scale=args.preview_scale,
        max_frames=args.max_frames,
        reference=args.reference,
        crf_method=args.crf_method,
        align_method=args.align_method,
        deghost_threshold=0.0 if args.deghost == "none" else args.deghost_threshold,
        formats=args.formats or ["jpg"],
        out_dir=args.out_dir,
        suffix=args.suffix,
        jpeg_quality=args.jpeg_quality,
        tonemap=args.tonemap,
        gamma=args.gamma,
        adjustments=_adjustments(args),
    )


def _brackets(args, paths: Sequence[str]) -> List[Bracket]:
    """Decide whether the inputs are one bracket or several.

    An explicit list of files is taken at face value -- the user said what they
    wanted. A folder is auto-grouped, because that is the whole point of
    pointing the tool at a folder.
    """
    folder_input = any(os.path.isdir(entry) for entry in args.inputs)
    if not folder_input and not args.bracket_size:
        return [Bracket([read_exif(path) for path in paths])]

    return group_frames(
        paths, expected_size=args.bracket_size,
        max_gap=getattr(args, "max_gap", 2.0),
    )


def cmd_merge(args) -> int:
    options = _options(args)
    options.validate()

    paths = expand_inputs(args.inputs)
    brackets = _brackets(args, paths)
    if not brackets:
        log.error("No brackets found in the given inputs")
        return 1

    if len(brackets) > 1:
        log.info("Detected %d brackets", len(brackets))

    failures = 0
    for index, bracket in enumerate(brackets, start=1):
        if len(brackets) > 1:
            log.info("[%d/%d] %s", index, len(brackets), bracket.describe())
        try:
            result = merge_bracket(bracket.paths, options)
        except (LoadError, WriteError, MergeFallback, ValueError) as exc:
            log.error("Failed on %s: %s", os.path.basename(bracket.paths[0]), exc)
            failures += 1
            continue
        if not result.written:
            log.warning("Nothing written for %s", os.path.basename(bracket.paths[0]))

    if failures:
        log.error("%d of %d bracket(s) failed", failures, len(brackets))
        return 1
    return 0


def cmd_inspect(args) -> int:
    """Print what the tool thinks each frame's exposure is, and where it got it."""
    paths = expand_inputs(args.inputs)
    metas = [read_exif(path) for path in paths]
    have_exif = all(meta.relative_exposure for meta in metas)

    # Pixel estimation compares frames in capture order, so establish an order
    # first: by EXIF exposure where we have it, by filename where we do not.
    order = sorted(
        range(len(metas)),
        key=lambda i: (metas[i].relative_exposure or 0.0, metas[i].name),
    )

    cross_check = {}
    if not have_exif or args.pixels:
        loaded = [load_frame(paths[i], preview_scale=0.25) for i in order]
        images = [frame.data for frame in loaded]
        is_linear = loaded[0].is_linear
        if have_exif:
            cross_check = _cross_check(metas, images, order, is_linear)
        else:
            # No usable EXIF anywhere: pixel estimation is the only source, so
            # write the estimates in as the real values.
            estimated = resolve_exposures(
                [metas[i] for i in order], images=images, images_are_linear=is_linear
            )
            for position, index in enumerate(order):
                metas[index] = estimated[position]

    header = f"{'frame':<28} {'settings':<26} {'EV':>7}  {'rel.exp':>9}  source"
    print(header)
    print("-" * len(header))
    for meta in sorted(metas, key=lambda m: m.ev if m.ev is not None else 0.0):
        ev = f"{meta.ev:+.2f}" if meta.ev is not None else "?"
        rel = f"{meta.relative_exposure:.4g}" if meta.relative_exposure else "?"
        line = f"{meta.name:<28} {meta.describe():<26} {ev:>7}  {rel:>9}  {meta.source}"
        if meta.name in cross_check:
            line += f"  (pixels: {cross_check[meta.name]:+.2f} EV)"
        print(line)

    evs = sorted(meta.ev for meta in metas if meta.ev is not None)
    if len(evs) > 1:
        steps = [b - a for a, b in zip(evs[:-1], evs[1:])]
        print(
            f"\n{len(metas)} frames, {evs[-1] - evs[0]:.1f} EV span, "
            f"steps: {', '.join(f'{step:.2f}' for step in steps)}"
        )
        if any(step < 0.05 for step in steps):
            print("note: two or more frames share an exposure")
    else:
        print(f"\n{len(metas)} frames, no exposure information available")
    return 0


def _cross_check(metas, images, order, images_are_linear) -> dict:
    """Pixel-estimated EV per frame, anchored to EXIF so the two are comparable.

    Pixel estimation only recovers exposure *ratios*, so its absolute scale is
    arbitrary. Anchoring to the darkest frame's EXIF value puts both on the same
    axis, which makes any disagreement in the steps -- the thing worth spotting
    -- visible at a glance.
    """
    estimates = estimate_relative_exposures(images, linear=images_are_linear)
    anchor_ev = metas[order[0]].ev
    if anchor_ev is None or estimates[0] <= 0:
        return {}

    offset = anchor_ev + math.log2(estimates[0])
    return {
        metas[index].name: -math.log2(estimates[position]) + offset
        for position, index in enumerate(order)
        if estimates[position] > 0
    }


def cmd_group(args) -> int:
    paths = expand_inputs(args.inputs)
    brackets = group_frames(paths, max_gap=args.max_gap, expected_size=args.bracket_size)
    if not brackets:
        print("No brackets detected.")
        return 1

    print(f"{len(brackets)} bracket(s) detected:\n")
    for index, bracket in enumerate(brackets, start=1):
        print(f"  [{index}] {bracket.describe()}")
        print(f"      -> {os.path.basename(output_base(bracket))}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args)

    handler = {"merge": cmd_merge, "inspect": cmd_inspect, "group": cmd_group}[args.command]
    try:
        return handler(args)
    except (LoadError, WriteError, MergeFallback, ValueError) as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.error("Interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())

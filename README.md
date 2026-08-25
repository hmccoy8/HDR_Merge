# HDR_Merge

Merge bracketed exposures into an HDR photograph — an open-source alternative to
Lightroom's *Photo Merge → HDR*.

Point it at a set of bracketed frames and it recovers a single high-dynamic-range
image, then tone maps and grades it into something you can actually use. Unlike
Lightroom's version, you choose the merge algorithm, you can inspect what it
decided about your exposures, and you can batch a whole shoot from a shell script.

```bash
hdrmerge merge IMG_000{1,2,3}.CR2 --auto
```

## Install

```bash
pip install -r requirements.txt
```

Or, equivalently, `pip install -e .` — the requirements file is a thin wrapper
around the package's own dependency list in `pyproject.toml`, so there is only
one list to keep correct.

Requires Python 3.9+. Camera RAW support comes from LibRaw via `rawpy`, which
ships as a wheel on Linux, macOS and Windows — no system libraries to install.

### Python 3.14 and EXR output

OpenEXR publishes wheels only up to Python 3.13. On 3.14 it is skipped
automatically, so hdrmerge installs with no compiler — but `--format exr` is
then unavailable and says so, up front, naming what to use instead. Nothing else
is affected: `tif32` and `hdr` both hold the same 32-bit linear radiance.

To add EXR back — once upstream ships 3.14 wheels, or if you have CMake and a
C++ toolchain:

```bash
pip install "hdrmerge[exr]"
```

On Python 3.9–3.13 it installs automatically and all six formats work.

One other thing to watch: this installs `opencv-python-headless`. If your
environment already has `opencv-python`, keep only one of the two — both provide
the `cv2` module and whichever was installed last silently wins.

## Usage

```bash
# One bracket, defaults (JPEG out, alongside the inputs)
hdrmerge merge IMG_0001.CR2 IMG_0002.CR2 IMG_0003.CR2

# A polished result with no fiddling
hdrmerge merge bracket/*.CR2 --auto

# Several formats at once -- the merge runs once and is written out repeatedly
hdrmerge merge bracket/*.CR2 -f jpg -f exr -f tif16 -o merged/

# A whole shoot: brackets are detected and merged one by one
hdrmerge merge ./shoot -o ./merged --auto

# Check what it thinks your exposures are, before trusting a merge
hdrmerge inspect bracket/*.CR2

# Check how a folder will be split, before committing to a long batch
hdrmerge group ./shoot
```

Use it as a library just as easily:

```python
from hdrmerge import Adjustments, MergeOptions, merge_bracket

result = merge_bracket(
    ["IMG_0001.CR2", "IMG_0002.CR2", "IMG_0003.CR2"],
    MergeOptions(formats=["jpg", "exr"], adjustments=Adjustments.auto()),
)
print(result.written, result.stats.dynamic_range_stops)
```

## How many photos at a time

Two at minimum, no fixed maximum. Three, five, seven and nine-frame brackets are
all ordinary; the merge is a weighted average across frames and does not care how
many there are.

The real limit is memory, not the algorithm. A 24 MP frame costs about 300 MB as
float32, so `--max-frames` defaults to **15** and refuses larger sets with an
explanation rather than letting the process get killed. Raise it if you have the
RAM. For experimenting with settings, `--preview-scale 0.25` works at quarter
size and is dramatically faster.

Frames may be given in any order — they are sorted by exposure before anything
else happens.

## Exposure detection

Exposures are detected automatically. You never enter a shutter speed.

**From EXIF**, when it is available:

```
relative_exposure = ExposureTime × (ISO / 100) / FNumber²
```

Only the ratios between frames matter, so no absolute calibration is needed.

**From the pixels**, when EXIF is missing, stripped, or in a container that
cannot be parsed (notably Canon CR3). For each adjacent pair of frames the tool
takes the pixels well exposed in *both* and uses the median of their ratios as
the exposure step, then chains those steps into a full scale. On rendered images
the sRGB curve is undone first, since brightness ratios only track exposure
ratios in linear light.

The two agree to within about 0.01 EV on well-behaved files. Cross-check them
yourself:

```
$ hdrmerge inspect bracket/*.jpg --pixels
frame                        settings                        EV    rel.exp  source
----------------------------------------------------------------------------------
IMG_0003.jpg                 1/12s f/8 ISO100             +9.64    0.00125  exif  (pixels: +9.65 EV)
IMG_0002.jpg                 1/50s f/8 ISO100            +11.64  0.0003125  exif  (pixels: +11.65 EV)
IMG_0001.jpg                 1/200s f/8 ISO100           +13.64  7.813e-05  exif  (pixels: +13.64 EV)

3 frames, 4.0 EV span, steps: 2.00, 2.00
```

## Input formats

| Class | Formats | Reader |
|---|---|---|
| Camera RAW | CR2, CR3, NEF, ARW, DNG, RAF, ORF, RW2, PEF, SRW, and anything else LibRaw opens | `rawpy` |
| Rendered | JPEG, TIFF (8/16-bit), PNG (8/16-bit), WebP, BMP | `tifffile` / OpenCV / Pillow |

**RAW is the better input, and not by a small margin.** LibRaw hands back linear
sensor data, so a merge is a straight exposure-weighted average with nothing to
estimate. A JPEG has been through a camera tone curve nobody published, which has
to be recovered (`--crf debevec`, the default) or assumed (`--crf srgb`) before
the same maths applies. 16-bit TIFF and PNG are read at their real depth rather
than being silently downgraded to 8-bit.

RAW and rendered frames cannot be mixed in one bracket — they are linear and
non-linear respectively, so averaging them together is meaningless. The tool says
so rather than producing a quietly wrong result.

## Output formats

Choose with `-f/--format`, repeat it for several. **Radiance formats are never
tone mapped** — preserving the linear data is the entire point of them.

| Format | Contents | Use for |
|---|---|---|
| `jpg` | 8-bit, tone-mapped | Sharing. Full resolution, quality 95, 4:4:4 chroma, sRGB profile embedded |
| `png16` | 16-bit, tone-mapped | Lossless delivery |
| `tif16` | 16-bit, tone-mapped | Handing to Lightroom or Photoshop for further editing |
| `tif32` | 32-bit float, **linear** | Radiance map in a familiar container |
| `exr` | OpenEXR 32-bit float, **linear** | Compositing and grading elsewhere (needs the `exr` extra on Python 3.14 — see [Install](#install)) |
| `hdr` | Radiance RGBE, **linear** | Compact interchange with 3D tools |

Outputs are named after the bracket's first frame: `IMG_0001_hdr.jpg`. The
32-bit TIFF gets `_linear` appended, since it shares an extension with `tif16`.

## Post-processing

Enough to finish a photo, deliberately not a full editor.

### Tone mapping (`-t/--tonemap`)

| Operator | Character |
|---|---|
| `reinhard` *(default)* | Global, natural, predictable |
| `drago` | Gentle highlight rolloff |
| `mantiuk` | Stronger local contrast |
| `linear` | Gamma and clip, no curve — for grading elsewhere |
| `fusion` | Exposure fusion: blends the frames directly, skipping the radiance map entirely |

`fusion` (Mertens) needs neither a response curve nor exposure metadata, which
makes it both the most robust option and often the best looking. Because it never
builds a radiance map it cannot produce `exr`, `hdr` or `tif32`, and says so
rather than writing something misleading.

### Adjustments

All default to no-op, all run −100…100 except where noted:

`--exposure` (stops) · `--temp` · `--tint` · `--highlights` · `--shadows` ·
`--whites` · `--blacks` · `--contrast` · `--saturation` · `--vibrance` ·
`--output-gamma`

`--vibrance` weights its boost towards muted colours, so it lifts a flat sky
without turning skin orange the way `--saturation` does. Highlights and shadows
work on luminance and are re-applied as a ratio, so heavy recovery does not drift
the colours.

**`--auto`** applies a tuned grade — mild shadow lift, highlight recovery, a
little contrast and vibrance — so a no-flags merge already looks finished. Any
explicit flag overrides the corresponding part of the preset.

## Motion

Handheld brackets drift, and things in the scene move. Both are handled by
default.

**Alignment** (`--align`) uses median-threshold bitmaps, which compare each frame
against its own median and are therefore unaffected by the exposure differences
that defeat ordinary registration. Every proposed shift is verified to actually
improve agreement with the reference frame before it is applied, so a frame too
dark or too blown to align does not get moved somewhere worse. `--align ecc` adds
an intensity-based refinement that also catches slight rotation; `--align none`
skips it for tripod work and is meaningfully faster.

**Deghosting** (`--deghost-threshold`, in stops) rejects pixels that disagree
with the reference frame once exposure is accounted for, so a moving subject
resolves to its position in that one frame instead of smearing across all of
them. The cost is that moving regions get the dynamic range of a single frame,
which is why the reference defaults to the middle exposure. `--deghost none`
turns it off.

One inherent limit worth knowing: where the reference frame is black or blown it
cannot judge anything, so other frames are trusted there. A ghost in deep shadow
is indistinguishable from shadow detail the bracket was shot to recover.

## How the merge works

For each pixel:

```
R = Σᵢ w(Zᵢ) · gᵢ · (Lᵢ / tᵢ)  /  Σᵢ w(Zᵢ) · gᵢ
```

`Lᵢ` is the linearised pixel value, `tᵢ` the relative exposure, `gᵢ` the
deghosting weight, and `w` a hat function discounting values near black (noise)
and near saturation (clipped, so they say nothing about the real scene
brightness).

Two details matter more than the average itself:

- **The weight is per-pixel, not per-channel**, taken from the brightest channel.
  Weighting channels independently is what gives blown highlights a colour cast,
  because the unclipped channels keep contributing after the clipped one stops.
- **Pixels no frame can describe fall back to a single frame** rather than
  dividing by zero. This is the usual source of black speckle in blown skies.

The result is anchored to the reference frame's exposure, so well-exposed
midtones come back near the value they had in that frame while recovered
highlights run above 1.0 — still perfectly linear, just scaled somewhere
intuitive.

## Development

```bash
pip install -r requirements-dev.txt
pytest
```

Run both from the repository root — the requirements files install the package
itself with `-e .`, which pip resolves relative to the working directory rather
than to the file.

Dependencies carry minimum bounds rather than exact pins, so hdrmerge installs
alongside your own packages. Two of those bounds are load-bearing and verified
against the suite: `opencv-python-headless>=4.10.0.84` (earlier releases were
built against the numpy 1 ABI and cannot import under numpy 2) and
`OpenEXR>=3.3` (the `OpenEXR.File` API that `writers.py` uses does not exist in
the 3.2 series).

The test suite builds synthetic brackets from scenes whose true radiance is known,
so merge accuracy is checked against ground truth in stops rather than eyeballed.

## Licence

MIT — see [LICENSE](LICENSE).

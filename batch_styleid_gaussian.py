import argparse
import gc
import os
from pathlib import Path

os.environ["ATTN_BACKEND"] = "xformers"
os.environ["SPCONV_ALGO"] = "auto"

import imageio.v2 as imageio
import torch
from PIL import Image

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils


DEFAULT_INPUT_ROOT = Path("/home/ps/cmh/StyleID/styleID_image")
DEFAULT_OUTPUT_ROOT = Path("/home/ps/cmh/StyleID/styleID_image_3d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render StyleID images as TRELLIS 3D Gaussian MP4 videos."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default="microsoft/TRELLIS-image-large")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--num-frames", type=int, default=300)
    parser.add_argument("--limit", type=int, help="Process at most this many pending images.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def output_is_complete(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    source_paths = sorted(input_root.glob("*/*.png"))
    if not source_paths:
        raise FileNotFoundError(f"No PNG inputs found under {input_root}")

    pending = []
    skipped = 0
    for source_path in source_paths:
        output_path = output_root / source_path.relative_to(input_root).with_suffix(".mp4")
        if not args.overwrite and output_is_complete(output_path):
            skipped += 1
            continue
        pending.append((source_path, output_path))

    if args.limit is not None:
        pending = pending[: args.limit]

    print(
        f"Found {len(source_paths)} inputs; pending={len(pending)}, "
        f"skipped={skipped}, backend={os.environ['ATTN_BACKEND']}"
    )
    if not pending:
        return

    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()

    completed = 0
    failed = []
    for index, (source_path, output_path) in enumerate(pending, start=1):
        temp_path = output_path.with_name(f".{output_path.stem}.tmp.mp4")
        print(f"[{index}/{len(pending)}] {source_path} -> {output_path}", flush=True)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path.unlink(missing_ok=True)
            with Image.open(source_path) as source_image:
                image = source_image.convert("RGB")

            outputs = pipeline.run(
                image,
                image,
                seed=args.seed,
                formats=["gaussian"],
            )
            frames = render_utils.render_video(
                outputs["gaussian"][0],
                bg_color=(1, 1, 1),
                num_frames=args.num_frames,
            )["color"]
            imageio.mimsave(temp_path, frames, fps=args.fps)
            temp_path.replace(output_path)
            completed += 1
        except Exception as error:
            temp_path.unlink(missing_ok=True)
            failed.append((source_path, repr(error)))
            print(f"FAILED: {source_path}: {error!r}", flush=True)
        finally:
            if "outputs" in locals():
                del outputs
            if "frames" in locals():
                del frames
            gc.collect()
            torch.cuda.empty_cache()

    print(f"Completed={completed}, failed={len(failed)}, skipped={skipped}")
    if failed:
        failure_log = output_root / "failures.tsv"
        failure_log.parent.mkdir(parents=True, exist_ok=True)
        failure_log.write_text(
            "".join(f"{path}\t{error}\n" for path, error in failed),
            encoding="utf-8",
        )
        print(f"Failure details: {failure_log}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
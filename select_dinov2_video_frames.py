import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((518, 518), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the video frame most similar to its input using DINOv2."
    )
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frame-step", type=int, default=25)
    parser.add_argument("--model", default="dinov2_vitl14_reg")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def image_tensor(image: Image.Image) -> torch.Tensor:
    return IMAGE_TRANSFORM(image.convert("RGB"))


def encode(images: list[Image.Image], model, device: torch.device, batch_size: int) -> torch.Tensor:
    embeddings = []
    for start in range(0, len(images), batch_size):
        batch = torch.stack([image_tensor(image) for image in images[start:start + batch_size]])
        batch = batch.to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            features = model.forward_features(batch)["x_norm_clstoken"]
        embeddings.append(F.normalize(features.float(), dim=-1).cpu())
    return torch.cat(embeddings)


def read_candidates(video_path: Path, frame_step: int) -> tuple[list[int], list[np.ndarray]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = list(range(0, frame_count, frame_step))
    frames = []
    try:
        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            actual_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if not ok or actual_index != frame_index:
                raise RuntimeError(
                    f"Could not read frame {frame_index} from {video_path}; got {actual_index}"
                )
            frames.append(frame)
    finally:
        capture.release()
    return frame_indices, frames


def main() -> None:
    args = parse_args()
    if args.frame_step <= 0:
        raise ValueError("--frame-step must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    video_root = args.video_root.resolve()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    video_paths = sorted(video_root.glob("*/*.mp4"))
    if not video_paths:
        raise FileNotFoundError(f"No MP4 videos found under {video_root}")

    pairs = []
    for video_path in video_paths:
        relative_path = video_path.relative_to(video_root)
        input_path = input_root / relative_path.with_suffix(".png")
        output_path = output_root / relative_path.with_suffix(".png")
        if not input_path.is_file():
            raise FileNotFoundError(f"Missing input image for {video_path}: {input_path}")
        pairs.append((video_path, input_path, output_path))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = torch.hub.load("facebookresearch/dinov2", args.model, pretrained=True)
    model.eval().to(device)

    input_images = []
    for _, input_path, _ in pairs:
        with Image.open(input_path) as image:
            input_images.append(image.convert("RGB"))
    input_embeddings = encode(input_images, model, device, args.batch_size)
    del input_images

    rows = []
    for pair_index, (video_path, input_path, output_path) in enumerate(
        tqdm(pairs, desc="Selecting frames"),
    ):
        frame_indices, frames = read_candidates(video_path, args.frame_step)
        candidate_images = [
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames
        ]
        candidate_embeddings = encode(candidate_images, model, device, args.batch_size)
        similarities = candidate_embeddings @ input_embeddings[pair_index]
        best_position = int(similarities.argmax())
        best_frame_index = frame_indices[best_position]
        best_score = float(similarities[best_position])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
        if not cv2.imwrite(str(temporary_path), frames[best_position]):
            raise RuntimeError(f"Could not write selected frame: {temporary_path}")
        temporary_path.replace(output_path)

        rows.append(
            {
                "video": str(video_path),
                "input": str(input_path),
                "output": str(output_path),
                "best_frame": best_frame_index,
                "best_similarity": best_score,
                "candidate_frames": ";".join(str(index) for index in frame_indices),
                "candidate_similarities": ";".join(
                    f"{float(score):.8f}" for score in similarities
                ),
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "dinov2_frame_selection.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Selected {len(rows)} frames with {args.model}; results: {csv_path}")


if __name__ == "__main__":
    main()
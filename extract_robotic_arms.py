#!/usr/bin/env python3
"""
Extract robotic arms from video using SAM3Predictor.

This script processes a video to detect robotic arms, extracts them using masks,
and saves cropped images with pure color backgrounds.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from sam3_predictor import SAM3Predictor


def extract_arms_with_mask(
    image: np.ndarray, mask: np.ndarray, background_color: tuple = (255, 255, 255)
) -> np.ndarray:
    """
    Extract robotic arm from image using mask and apply pure color background.

    Args:
        image: Original RGB image (H, W, 3)
        mask: Binary mask (H, W) where True indicates robotic arm
        background_color: RGB color for background (default: white)

    Returns:
        Image with robotic arm and pure color background (H, W, 3)
    """
    # Ensure image is uint8
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    # Ensure mask is boolean
    if mask.dtype != bool:
        mask = mask > 0.5

    # Create output image with background color
    h, w = image.shape[:2]
    output = np.full((h, w, 3), background_color, dtype=np.uint8)

    # Apply mask: copy pixels where mask is True
    output[mask] = image[mask]

    return output


def crop_to_mask_bounds(
    image: np.ndarray, mask: np.ndarray, padding: int = 10
) -> tuple:
    """
    Get bounding box coordinates for cropping to mask region.

    Args:
        image: Image array (H, W, 3)
        mask: Binary mask (H, W)
        padding: Padding pixels around mask (default: 10)

    Returns:
        Tuple of (x1, y1, x2, y2) bounding box coordinates
    """
    # Find mask bounds
    mask_coords = np.where(mask)
    if len(mask_coords[0]) == 0:
        return None

    y_min, y_max = mask_coords[0].min(), mask_coords[0].max()
    x_min, x_max = mask_coords[1].min(), mask_coords[1].max()

    # Add padding
    h, w = image.shape[:2]
    x1 = max(0, x_min - padding)
    y1 = max(0, y_min - padding)
    x2 = min(w, x_max + padding + 1)
    y2 = min(h, y_max + padding + 1)

    return (x1, y1, x2, y2)


def process_video_for_robotic_arms(
    video_path: str,
    text_prompt: str = "robotic arm",
    output_dir: str = "output_robotic_arms",
    background_color: tuple = (255, 255, 255),
    crop_to_mask: bool = True,
    padding: int = 10,
):
    """
    Process video to extract robotic arms with pure color backgrounds.

    Args:
        video_path: Path to input video file
        text_prompt: Text prompt for SAM3 (default: "robotic arm")
        output_dir: Output directory for results
        background_color: RGB color for background (default: white)
        crop_to_mask: Whether to crop to mask bounds (default: True)
        padding: Padding around mask when cropping (default: 10)
    """
    print("=" * 60)
    print("Robotic Arm Extraction using SAM3")
    print("=" * 60)

    # Initialize predictor
    print("\nInitializing SAM3 video predictor...")
    predictor = SAM3Predictor(mode="video", confidence_threshold=0.2)

    # Process video
    print(f"\nProcessing video: {video_path}")
    print(f"Text prompt: '{text_prompt}'")
    results = predictor.predict_video(
        video_path=video_path,
        text_prompt=text_prompt,
        frame_index=0,
        return_visualization=False,  # We'll create our own visualizations
        propagate=True,
    )

    print(f"Processed {len(results['outputs_per_frame'])} frames")

    # Load original video frames
    print("\nLoading original video frames...")
    cap = cv2.VideoCapture(video_path)
    original_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        original_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    print(f"Loaded {len(original_frames)} frames")

    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    full_frame_dir = output_path / "full_frame"
    cropped_dir = output_path / "cropped"
    composite_dir = output_path / "composite"  # For video generation
    original_dir = output_path / "original_rgb"  # Original RGB frames
    full_frame_dir.mkdir(exist_ok=True)
    cropped_dir.mkdir(exist_ok=True)
    composite_dir.mkdir(exist_ok=True)
    original_dir.mkdir(exist_ok=True)

    print(f"\nExtracting robotic arms and saving to {output_path}")

    # Save original RGB frames
    print("\nSaving original RGB frames...")
    for frame_idx, frame_rgb in enumerate(original_frames):
        original_path = original_dir / f"frame_{frame_idx:05d}.jpg"
        Image.fromarray(frame_rgb).save(original_path)
    print(f"Saved {len(original_frames)} original RGB frames to {original_dir}")

    # Process each frame
    total_arms_extracted = 0
    frames_with_detections = 0

    for frame_idx in range(len(original_frames)):
        frame_rgb = original_frames[frame_idx]

        if frame_idx not in results["outputs_per_frame"]:
            # No detections for this frame - save empty frame with background
            composite_image = np.full(frame_rgb.shape, background_color, dtype=np.uint8)
            composite_path = composite_dir / f"frame_{frame_idx:05d}.jpg"
            Image.fromarray(composite_image).save(composite_path)
            continue

        outputs = results["outputs_per_frame"][frame_idx]

        # Get masks from outputs
        if "out_binary_masks" not in outputs:
            # No masks - save empty frame
            composite_image = np.full(frame_rgb.shape, background_color, dtype=np.uint8)
            composite_path = composite_dir / f"frame_{frame_idx:05d}.jpg"
            Image.fromarray(composite_image).save(composite_path)
            continue

        masks = outputs["out_binary_masks"]
        obj_ids = outputs["out_obj_ids"]

        if len(masks) == 0:
            # No masks - save empty frame
            composite_image = np.full(frame_rgb.shape, background_color, dtype=np.uint8)
            composite_path = composite_dir / f"frame_{frame_idx:05d}.jpg"
            Image.fromarray(composite_image).save(composite_path)
            continue

        # Create composite mask combining all detected arms
        h, w = frame_rgb.shape[:2]
        combined_mask = np.zeros((h, w), dtype=bool)

        # Process each detected robotic arm individually (for separate saving)
        for arm_idx, (mask, obj_id) in enumerate(zip(masks, obj_ids)):
            # Ensure mask matches frame dimensions
            if mask.shape != frame_rgb.shape[:2]:
                mask_resized = cv2.resize(
                    mask.astype(np.float32),
                    (frame_rgb.shape[1], frame_rgb.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            else:
                mask_resized = mask.astype(bool)

            # Add to combined mask
            combined_mask = combined_mask | mask_resized

            # Extract arm with pure color background (for individual saving)
            arm_image = extract_arms_with_mask(
                frame_rgb, mask_resized, background_color
            )

            # Save full frame version (individual arm)
            full_frame_path = full_frame_dir / f"frame_{frame_idx:05d}_arm_{obj_id}.jpg"
            Image.fromarray(arm_image).save(full_frame_path)

            # Crop to mask bounds if requested (individual arm)
            if crop_to_mask:
                bbox = crop_to_mask_bounds(arm_image, mask_resized, padding=padding)
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    cropped_arm = arm_image[y1:y2, x1:x2]
                    cropped_path = (
                        cropped_dir / f"frame_{frame_idx:05d}_arm_{obj_id}.jpg"
                    )
                    Image.fromarray(cropped_arm).save(cropped_path)

            total_arms_extracted += 1

        # Create composite image with all arms in the same frame
        composite_image = extract_arms_with_mask(
            frame_rgb, combined_mask, background_color
        )
        composite_path = composite_dir / f"frame_{frame_idx:05d}.jpg"
        Image.fromarray(composite_image).save(composite_path)
        frames_with_detections += 1

        # Progress update
        if (frame_idx + 1) % 10 == 0:
            print(f"Processed {frame_idx + 1}/{len(original_frames)} frames...")

    print("\n✓ Extraction complete!")
    print(f"  Total robotic arms extracted: {total_arms_extracted}")
    print(f"  Frames with detections: {frames_with_detections}")
    print(f"  Original RGB frames: {original_dir}")
    print(f"  Individual arm images (full frame): {full_frame_dir}")
    print(f"  Individual arm images (cropped): {cropped_dir}")
    print(f"  Composite frames (all arms per frame): {composite_dir}")

    # Also save a summary video from composite frames
    print("\nCreating summary video from composite frames...")
    try:
        create_summary_video(composite_dir, output_path / "summary_video.mp4", fps=30.0)
        print(f"  Summary video: {output_path / 'summary_video.mp4'}")
    except Exception as e:
        print(f"  Warning: Could not create summary video: {e}")


def create_summary_video(image_dir: Path, output_path: Path, fps: float = 30.0):
    """
    Create a video from composite frame images (one per frame).

    Args:
        image_dir: Directory containing composite frame images (frame_XXXXX.jpg)
        output_path: Output video path
        fps: Frames per second for output video
    """
    # Get all frame images sorted by frame number
    image_files = sorted(
        image_dir.glob("frame_*.jpg"), key=lambda p: int(p.stem.split("_")[1])
    )

    if len(image_files) == 0:
        print(f"  Warning: No images found in {image_dir}")
        return

    # Get dimensions from first image
    first_img = Image.open(image_files[0])
    w, h = first_img.size

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    print(f"  Writing {len(image_files)} frames to video...")
    for img_path in image_files:
        img = Image.open(img_path)
        img_array = np.array(img)
        # Convert RGB to BGR for OpenCV
        img_bgr = img_array[:, :, ::-1]
        out.write(img_bgr)

    out.release()
    print(f"  Video saved: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract robotic arms from video using SAM3"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="./assets/videos/agi001-10.mp4",
        help="Path to input video file",
    )
    parser.add_argument(
        "--text_prompt",
        type=str,
        default="robotic arm",
        help="Text prompt for SAM3 detection",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_robotic_arms",
        help="Output directory",
    )
    parser.add_argument(
        "--background_color",
        type=int,
        nargs=3,
        default=[255, 255, 255],
        help="Background color RGB (default: 255 255 255 for white)",
    )
    parser.add_argument(
        "--no_crop",
        action="store_true",
        help="Don't crop to mask bounds (save full frame)",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=10,
        help="Padding pixels around mask when cropping (default: 10)",
    )

    args = parser.parse_args()

    # Convert background color to tuple
    bg_color = tuple(args.background_color)

    # Process video
    process_video_for_robotic_arms(
        video_path=args.video,
        text_prompt=args.text_prompt,
        output_dir=args.output,
        background_color=bg_color,
        crop_to_mask=not args.no_crop,
        padding=args.padding,
    )

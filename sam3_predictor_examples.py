#!/usr/bin/env python3
"""
Examples demonstrating how to use SAM3Predictor.

This script shows various usage patterns for the SAM3Predictor class.
"""

from pathlib import Path

from sam3_predictor import SAM3Predictor
from PIL import Image
import numpy as np


def example_image_text_prompt():
    """Example: Process image with text prompt."""
    print("=" * 60)
    print("Example 1: Image with text prompt")
    print("=" * 60)

    # Initialize predictor
    predictor = SAM3Predictor(mode="image", confidence_threshold=0.5)

    # Load image
    image_path = Path(__file__).parent / "assets" / "images" / "test_image.jpg"
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        print("Please provide a valid image path")
        return

    # Process with text prompt
    results = predictor.predict_image(
        image=str(image_path),
        text_prompt="shoe",
        return_visualization=True,
    )

    print(f"Found {len(results['bboxes'])} objects")
    print(f"Bboxes shape: {results['bboxes'].shape}")
    print(f"Masks shape: {results['masks'].shape}")
    print(f"Scores: {results['scores']}")

    # Save results (including visualization)
    output_dir = Path(__file__).parent / "test_output" / "example1"
    predictor.save_results(results, str(output_dir), save_visualization=True)
    print(f"\nResults saved to {output_dir}")
    if results["visualization"] is not None:
        print(f"✓ Visualization image saved: {output_dir / 'visualization.jpg'}")


def example_image_box_prompt():
    """Example: Process image with box prompt."""
    print("\n" + "=" * 60)
    print("Example 2: Image with box prompt")
    print("=" * 60)

    predictor = SAM3Predictor(mode="image")

    image_path = Path(__file__).parent / "assets" / "images" / "test_image.jpg"
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return

    # Define box in [x, y, w, h] format (top-left corner, width, height)
    # Example: box at (480, 290) with size 110x360
    boxes = [[480.0, 290.0, 110.0, 360.0]]
    box_labels = [True]  # True = positive prompt

    results = predictor.predict_image(
        image=str(image_path),
        boxes=boxes,
        box_labels=box_labels,
        return_visualization=True,
    )

    print(f"Found {len(results['bboxes'])} objects")
    print(f"Bboxes:\n{results['bboxes']}")

    output_dir = Path(__file__).parent / "test_output" / "example2"
    predictor.save_results(results, str(output_dir), save_visualization=True)
    print(f"\nResults saved to {output_dir}")
    if results["visualization"] is not None:
        print(f"✓ Visualization image saved: {output_dir / 'visualization.jpg'}")


def example_image_multi_prompt():
    """Example: Process image with text + multiple boxes."""
    print("\n" + "=" * 60)
    print("Example 3: Image with text + multiple boxes")
    print("=" * 60)

    predictor = SAM3Predictor(mode="image")

    image_path = Path(__file__).parent / "assets" / "images" / "test_image.jpg"
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return

    # Text prompt + positive and negative boxes
    boxes = [
        [480.0, 290.0, 110.0, 360.0],  # Positive box
        [370.0, 280.0, 115.0, 375.0],  # Negative box
    ]
    box_labels = [True, False]  # First is positive, second is negative

    results = predictor.predict_image(
        image=str(image_path),
        text_prompt="person",
        boxes=boxes,
        box_labels=box_labels,
        return_visualization=True,
    )

    print(f"Found {len(results['bboxes'])} objects")
    print(f"Scores: {results['scores']}")

    output_dir = Path(__file__).parent / "test_output" / "example3"
    predictor.save_results(results, str(output_dir), save_visualization=True)
    print(f"\nResults saved to {output_dir}")
    if results["visualization"] is not None:
        print(f"✓ Visualization image saved: {output_dir / 'visualization.jpg'}")


def example_video_text_prompt():
    """Example: Process video with text prompt."""
    print("\n" + "=" * 60)
    print("Example 4: Video with text prompt")
    print("=" * 60)

    predictor = SAM3Predictor(mode="video")

    video_path = Path(__file__).parent / "assets" / "videos" / "segment_037.mp4"
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        print("Please provide a valid video path")
        return

    # Process video with text prompt
    results = predictor.predict_video(
        video_path=str(video_path),
        text_prompt="human hand",
        frame_index=0,
        return_visualization=True,
        propagate=True,
    )

    print(f"Session ID: {results['session_id']}")
    print(f"Processed {len(results['outputs_per_frame'])} frames")

    # Count objects per frame
    for frame_idx, outputs in list(results["outputs_per_frame"].items())[:5]:
        num_objects = len(outputs)
        print(f"Frame {frame_idx}: {num_objects} objects")

    output_dir = Path(__file__).parent / "test_output" / "example4"
    predictor.save_results(results, str(output_dir))
    print(f"\nResults saved to {output_dir}")


def example_video_point_prompt():
    """Example: Process video with point prompt (for refining existing tracked objects)."""
    print("\n" + "=" * 60)
    print("Example 5: Video with point prompt (refining existing object)")
    print("=" * 60)

    predictor = SAM3Predictor(mode="video")

    video_path = Path(__file__).parent / "assets" / "videos" / "bedroom.mp4"
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return

    # Step 1: First detect objects with text prompt
    print("Step 1: Detecting objects with text prompt...")
    results = predictor.predict_video(
        video_path=str(video_path),
        text_prompt="person",
        frame_index=0,
        return_visualization=False,  # Skip visualization for first pass
        propagate=False,  # Don't propagate yet
    )

    # Step 2: Refine first detected object with point prompt
    outputs_frame0 = results["outputs_per_frame"].get(0)
    if outputs_frame0 is not None and "out_obj_ids" in outputs_frame0:
        obj_ids = outputs_frame0["out_obj_ids"]
        if len(obj_ids) > 0:
            # Get first object ID
            first_obj_id = int(obj_ids[0])
            print(f"Step 2: Refining object {first_obj_id} with point prompt...")

            # Point prompt: click at (x, y) coordinates (normalized 0-1)
            # Note: Points must be in normalized coordinates [0-1]
            # Assuming image size ~1008x1008, adjust based on your video
            img_width, img_height = 1008, 1008  # Adjust based on your video
            points = [[500 / img_width, 300 / img_height]]  # Normalized coordinates

            # Refine with point prompt (requires same session)
            # Note: This creates a new session, so we'll just demonstrate the concept
            print(
                "Note: Point prompts require obj_id and work within the same session. "
                "This example demonstrates the API usage."
            )
            print(f"To refine object {first_obj_id}, use:")
            print(
                f"  predictor.predict_video(..., points={points}, "
                f"point_labels=[1], obj_id={first_obj_id})"
            )

            # Save initial results
            output_dir = Path(__file__).parent / "test_output" / "example5"
            predictor.save_results(results, str(output_dir))
            print(f"\nInitial detection results saved to {output_dir}")
        else:
            print("No objects detected in first frame.")
    else:
        print("No outputs available for point refinement example.")


def example_from_path():
    """Example: Process from file path (auto-detect image/video)."""
    print("\n" + "=" * 60)
    print("Example 6: Process from path (auto-detect)")
    print("=" * 60)

    # Image mode
    predictor_image = SAM3Predictor(mode="image")
    image_path = Path(__file__).parent / "assets" / "images" / "test_image.jpg"
    if image_path.exists():
        results = predictor_image.predict_from_path(
            str(image_path), text_prompt="person", return_visualization=True
        )
        print(f"Image: Found {len(results['bboxes'])} objects")

    # # Video mode
    # predictor_video = SAM3Predictor(mode="video")
    # video_path = Path(__file__).parent / "assets" / "videos" / "bedroom.mp4"
    # if video_path.exists():
    #     results = predictor_video.predict_from_path(
    #         str(video_path), text_prompt="person", return_visualization=True
    #     )
    #     print(f"Video: Processed {len(results['outputs_per_frame'])} frames")


def example_numpy_array():
    """Example: Process numpy array directly."""
    print("\n" + "=" * 60)
    print("Example 7: Process numpy array")
    print("=" * 60)

    predictor = SAM3Predictor(mode="image")

    # Create dummy image (or load from file)
    image_path = Path(__file__).parent / "assets" / "images" / "test_image.jpg"
    if image_path.exists():
        img = Image.open(image_path)
        img_array = np.array(img)  # Convert to numpy array

        # Process numpy array
        results = predictor.predict_image(
            image=img_array,
            text_prompt="person",
            return_visualization=True,
        )

        print(f"Found {len(results['bboxes'])} objects")
        print(f"Input was numpy array with shape {img_array.shape}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAM3Predictor Examples")
    parser.add_argument(
        "--example",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7],
        help="Example number to run (1-7)",
    )
    args = parser.parse_args()

    examples = {
        1: example_image_text_prompt,
        2: example_image_box_prompt,
        3: example_image_multi_prompt,
        4: example_video_text_prompt,
        5: example_video_point_prompt,
        6: example_from_path,
        7: example_numpy_array,
    }

    if args.example:
        examples[args.example]()
    else:
        print("Running all examples...")
        for example_func in examples.values():
            try:
                example_func()
            except Exception as e:
                print(f"Error in example: {e}")
                import traceback

                traceback.print_exc()

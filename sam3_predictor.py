# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
SAM3Predictor - A reusable, thread-safe predictor for SAM 3.

This class provides a clean interface for processing images/videos with SAM 3,
supporting text prompts, geometric prompts (boxes, points), and both image
and video processing modes.

Features:
- Single images or image sequences (videos)
- Text prompts for open-vocabulary segmentation
- Box prompts (positive/negative)
- Point prompts (positive/negative)
- Returns: bboxes, masks, scores, visualizations
- Thread-safe for concurrent use
"""

import threading
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import cv2
import torch
from PIL import Image

# Import SAM3 components
import sam3
from sam3 import build_sam3_image_model, build_sam3_video_predictor
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.visualization_utils import plot_results, normalize_bbox


class SAM3Predictor:
    """
    A reusable, thread-safe predictor for SAM 3 segmentation and tracking.

    This class can process single images or sequences of images (videos) and returns
    both the raw prediction results (bboxes, masks, scores) and rendered visualizations.

    Example:
        >>> # Image mode
        >>> predictor = SAM3Predictor(mode="image")
        >>> img = Image.open("image.jpg")
        >>> results = predictor.predict_image(
        ...     image=img,
        ...     text_prompt="person",
        ...     return_visualization=True
        ... )
        >>>
        >>> # Video mode
        >>> predictor = SAM3Predictor(mode="video")
        >>> results = predictor.predict_video(
        ...     video_path="video.mp4",
        ...     text_prompt="person",
        ...     return_visualization=True
        ... )
    """

    def __init__(
        self,
        mode: str = "image",
        checkpoint_path: Optional[str] = None,
        bpe_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        device: Optional[torch.device] = None,
        gpus_to_use: Optional[List[int]] = None,
    ):
        """
        Initialize the SAM3Predictor.

        Args:
            mode: Processing mode - "image" or "video" (default: "image")
            checkpoint_path: Optional path to SAM3 checkpoint file
            bpe_path: Optional path to BPE tokenizer vocabulary file
            confidence_threshold: Confidence threshold for detections (default: 0.5)
            device: Torch device (default: auto-detect CUDA)
            gpus_to_use: List of GPU indices for video mode (default: all available)
        """
        if mode not in ["image", "video"]:
            raise ValueError(f"mode must be 'image' or 'video', got '{mode}'")

        self.mode = mode
        self.confidence_threshold = confidence_threshold

        # Thread safety lock
        self._lock = threading.Lock()

        # Device setup
        if device is None:
            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            self.device = device

        # Setup BPE path
        if bpe_path is None:
            sam3_root = Path(sam3.__file__).parent.parent
            bpe_path = str(sam3_root / "assets" / "bpe_simple_vocab_16e6.txt.gz")

        print(f"Loading SAM3 {mode} model on {self.device}...")

        if mode == "image":
            # Build image model
            self.model = build_sam3_image_model(
                bpe_path=bpe_path,
                device=str(self.device),
                checkpoint_path=checkpoint_path,
            )
            # Create processor
            self.processor = Sam3Processor(
                model=self.model,
                device=str(self.device),
                confidence_threshold=confidence_threshold,
            )
            self.video_predictor = None
        else:
            # Build video predictor
            if gpus_to_use is None:
                gpus_to_use = list(range(torch.cuda.device_count()))
            self.video_predictor = build_sam3_video_predictor(
                checkpoint_path=checkpoint_path,
                bpe_path=bpe_path,
                gpus_to_use=gpus_to_use,
            )
            self.model = None
            self.processor = None

        print("Initialization complete.")

    def predict_image(
        self,
        image: Union[Image.Image, np.ndarray, str],
        text_prompt: Optional[str] = None,
        boxes: Optional[List[List[float]]] = None,
        box_labels: Optional[List[bool]] = None,
        return_visualization: bool = True,
    ) -> Dict:
        """
        Process a single image and return predictions.

        Args:
            image: PIL Image, numpy array (H, W, 3) RGB, or path to image file
            text_prompt: Optional text prompt (e.g., "person", "shoe")
            boxes: Optional list of boxes in [x, y, w, h] format (top-left, width, height)
            box_labels: Optional list of bool labels for boxes (True=positive, False=negative)
            return_visualization: Whether to generate visualization (default: True)

        Returns:
            Dictionary containing:
                - 'bboxes': List of bounding boxes [x0, y0, x1, y1] (N, 4)
                - 'masks': Binary masks (N, H, W) as numpy arrays
                - 'scores': Confidence scores (N,)
                - 'visualization': PIL Image with overlays (if return_visualization=True)
        """
        if self.mode != "image":
            raise RuntimeError("predict_image() requires mode='image'")

        # Thread-safe execution
        with self._lock:
            return self._predict_image_impl(
                image, text_prompt, boxes, box_labels, return_visualization
            )

    def _predict_image_impl(
        self,
        image: Union[Image.Image, np.ndarray, str],
        text_prompt: Optional[str],
        boxes: Optional[List[List[float]]],
        box_labels: Optional[List[bool]],
        return_visualization: bool,
    ) -> Dict:
        """Internal implementation (called within lock)."""
        # Load image
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)

        if not isinstance(image, Image.Image):
            raise ValueError("image must be PIL Image, numpy array, or file path")

        width, height = image.size

        # Set image in processor
        inference_state = self.processor.set_image(image)

        # Reset prompts
        self.processor.reset_all_prompts(inference_state)

        # Add text prompt if provided
        if text_prompt:
            inference_state = self.processor.set_text_prompt(
                prompt=text_prompt, state=inference_state
            )

        # Add box prompts if provided
        if boxes:
            if box_labels is None:
                box_labels = [True] * len(boxes)

            for box, label in zip(boxes, box_labels):
                # Convert [x, y, w, h] to [cx, cy, w, h] and normalize
                box_cxcywh = box_xywh_to_cxcywh(torch.tensor(box).view(-1, 4))
                norm_box = normalize_bbox(box_cxcywh, width, height).flatten().tolist()
                inference_state = self.processor.add_geometric_prompt(
                    state=inference_state, box=norm_box, label=label
                )

        # Extract results
        if "boxes" not in inference_state:
            # No detections
            return {
                "bboxes": np.array([]).reshape(0, 4),
                "masks": np.array([]).reshape(0, height, width),
                "scores": np.array([]),
                "visualization": image.copy() if return_visualization else None,
            }

        # Convert tensors to numpy, handling BFloat16
        boxes_np = inference_state["boxes"].cpu().float().numpy()
        masks_np = inference_state["masks"].cpu().numpy()
        scores_tensor = inference_state["scores"]
        # Handle BFloat16 by converting to float32
        if scores_tensor.dtype == torch.bfloat16:
            scores_np = scores_tensor.cpu().float().numpy()
        else:
            scores_np = scores_tensor.cpu().numpy()

        result = {
            "bboxes": boxes_np,
            "masks": masks_np,
            "scores": scores_np,
            "visualization": None,
        }

        # Generate visualization if requested
        if return_visualization:
            vis_image = plot_results(image, inference_state)
            result["visualization"] = vis_image

        return result

    def predict_video(
        self,
        video_path: Union[str, Path],
        text_prompt: Optional[str] = None,
        points: Optional[List[List[float]]] = None,
        point_labels: Optional[List[int]] = None,
        bounding_boxes: Optional[List[List[float]]] = None,
        bounding_box_labels: Optional[List[int]] = None,
        obj_id: Optional[int] = None,
        frame_index: int = 0,
        return_visualization: bool = True,
        propagate: bool = True,
    ) -> Dict:
        """
        Process a video and return predictions.

        Args:
            video_path: Path to video file (.mp4) or directory with JPEG frames
            text_prompt: Optional text prompt (e.g., "person", "shoe")
            points: Optional list of points [[x, y], ...] in absolute coordinates.
                   Note: Point-only prompts require obj_id and cannot be combined with text/box.
            point_labels: Optional list of point labels (1=positive, 0=negative)
            bounding_boxes: Optional list of boxes [[x, y, w, h], ...] in absolute coordinates
            bounding_box_labels: Optional list of box labels (1=positive, 0=negative)
            obj_id: Optional object ID for point prompts (required if only points provided)
            frame_index: Frame index to add prompt on (default: 0)
            return_visualization: Whether to generate visualization (default: True)
            propagate: Whether to propagate masks through video (default: True)

        Returns:
            Dictionary containing:
                - 'session_id': Session identifier
                - 'outputs_per_frame': Dict mapping frame_index to outputs
                    Each output contains:
                    - 'masks': List of mask dictionaries with 'mask', 'obj_id', 'score'
                    - 'bboxes': List of bounding boxes
                    - 'scores': List of confidence scores
                - 'visualizations': List of PIL Images (if return_visualization=True)
        """
        if self.mode != "video":
            raise RuntimeError("predict_video() requires mode='video'")

        # Thread-safe execution
        with self._lock:
            return self._predict_video_impl(
                video_path,
                text_prompt,
                points,
                point_labels,
                bounding_boxes,
                bounding_box_labels,
                obj_id,
                frame_index,
                return_visualization,
                propagate,
            )

    def _predict_video_impl(
        self,
        video_path: Union[str, Path],
        text_prompt: Optional[str],
        points: Optional[List[List[float]]],
        point_labels: Optional[List[int]],
        bounding_boxes: Optional[List[List[float]]],
        bounding_box_labels: Optional[List[int]],
        obj_id: Optional[int],
        frame_index: int,
        return_visualization: bool,
        propagate: bool,
    ) -> Dict:
        """Internal implementation (called within lock)."""
        video_path = str(video_path)

        # Start session
        response = self.video_predictor.handle_request(
            request=dict(type="start_session", resource_path=video_path)
        )
        session_id = response["session_id"]

        # Add prompt
        # Note: Point-only prompts require obj_id and cannot be combined with text/box.
        # For new object detection, use text or box prompts.
        # For refining existing objects, use points with obj_id.
        if points and (text_prompt or bounding_boxes):
            raise ValueError(
                "Point prompts cannot be combined with text or box prompts. "
                "Points are for refining existing tracked objects (require obj_id). "
                "Use text_prompt or bounding_boxes for new object detection."
            )

        prompt_request = dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=frame_index,
        )
        if text_prompt:
            prompt_request["text"] = text_prompt
        if points:
            prompt_request["points"] = points
            prompt_request["point_labels"] = point_labels or [1] * len(points)
            if obj_id is not None:
                prompt_request["obj_id"] = obj_id
        if bounding_boxes:
            prompt_request["bounding_boxes"] = bounding_boxes
            prompt_request["bounding_box_labels"] = bounding_box_labels or [1] * len(
                bounding_boxes
            )

        response = self.video_predictor.handle_request(request=prompt_request)
        outputs_per_frame = {frame_index: response["outputs"]}

        # Propagate if requested
        if propagate:
            for response in self.video_predictor.handle_stream_request(
                request=dict(
                    type="propagate_in_video",
                    session_id=session_id,
                )
            ):
                outputs_per_frame[response["frame_index"]] = response["outputs"]

        result = {
            "session_id": session_id,
            "outputs_per_frame": outputs_per_frame,
            "visualizations": None,
        }

        # Generate visualizations if requested
        if return_visualization:
            # Load frames for visualization
            import glob
            import os

            if video_path.endswith(".mp4"):
                cap = cv2.VideoCapture(video_path)
                frames = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap.release()
            else:
                frame_files = sorted(
                    glob.glob(os.path.join(video_path, "*.jpg")),
                    key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
                )
                frames = [cv2.imread(f)[:, :, ::-1] for f in frame_files]

            # Convert outputs to format expected by render_masklet_frame
            # outputs_per_frame contains dicts with keys: out_boxes_xywh, out_probs, out_obj_ids, out_binary_masks
            from sam3.visualization_utils import render_masklet_frame

            visualizations = []
            for frame_idx, frame_rgb in enumerate(frames):
                if frame_idx in outputs_per_frame:
                    outputs = outputs_per_frame[frame_idx]
                    # Use render_masklet_frame for simple visualization
                    vis_frame = render_masklet_frame(
                        frame_rgb, outputs, frame_idx=frame_idx, alpha=0.5
                    )
                    visualizations.append(Image.fromarray(vis_frame))
                else:
                    visualizations.append(Image.fromarray(frame_rgb))

            result["visualizations"] = visualizations

        return result

    def predict_from_path(
        self,
        input_path: str,
        text_prompt: Optional[str] = None,
        boxes: Optional[List[List[float]]] = None,
        box_labels: Optional[List[bool]] = None,
        return_visualization: bool = True,
    ) -> Dict:
        """
        Process input from file path (image or video).

        Args:
            input_path: Path to image file (.jpg, .png, etc.) or video file (.mp4, etc.)
            text_prompt: Optional text prompt
            boxes: Optional list of boxes [x, y, w, h]
            box_labels: Optional list of box labels
            return_visualization: Whether to return visualization

        Returns:
            Results dictionary (format depends on image vs video)
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

        if input_path.suffix.lower() in video_extensions:
            if self.mode != "video":
                raise ValueError(
                    "Video file provided but predictor is in 'image' mode. "
                    "Initialize with mode='video' for video processing."
                )
            return self.predict_video(
                video_path=str(input_path),
                text_prompt=text_prompt,
                bounding_boxes=boxes,
                bounding_box_labels=box_labels,
                return_visualization=return_visualization,
            )
        elif input_path.suffix.lower() in image_extensions:
            if self.mode != "image":
                raise ValueError(
                    "Image file provided but predictor is in 'video' mode. "
                    "Initialize with mode='image' for image processing."
                )
            return self.predict_image(
                image=str(input_path),
                text_prompt=text_prompt,
                boxes=boxes,
                box_labels=box_labels,
                return_visualization=return_visualization,
            )
        else:
            raise ValueError(
                f"Unsupported file format: {input_path.suffix}. "
                f"Supported: {video_extensions | image_extensions}"
            )

    def save_results(
        self,
        results: Dict,
        output_path: str,
        save_visualization: bool = True,
        save_masks: bool = True,
        save_bboxes: bool = True,
    ):
        """
        Save prediction results to disk.

        Args:
            results: Results dictionary from predict_image() or predict_video()
            output_path: Output directory or file path
            save_visualization: Whether to save visualization images
            save_masks: Whether to save mask arrays
            save_bboxes: Whether to save bounding box coordinates
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.mode == "image":
            # Save single image results
            if save_visualization and results["visualization"] is not None:
                vis_path = output_path / "visualization.jpg"
                results["visualization"].save(vis_path)
                print(f"Saved visualization to {vis_path}")

            if save_masks and len(results["masks"]) > 0:
                masks_path = output_path / "masks.npy"
                np.save(str(masks_path), results["masks"])
                print(f"Saved masks to {masks_path}")

            if save_bboxes and len(results["bboxes"]) > 0:
                bboxes_path = output_path / "bboxes.npy"
                np.save(str(bboxes_path), results["bboxes"])
                print(f"Saved bboxes to {bboxes_path}")

            # Save scores
            if len(results["scores"]) > 0:
                scores_path = output_path / "scores.npy"
                np.save(str(scores_path), results["scores"])
                print(f"Saved scores to {scores_path}")

        else:
            # Save video results
            outputs_per_frame = results["outputs_per_frame"]

            if save_visualization and results.get("visualizations"):
                vis_dir = output_path / "visualizations"
                vis_dir.mkdir(exist_ok=True)
                for frame_idx, vis_img in enumerate(results["visualizations"]):
                    vis_path = vis_dir / f"frame_{frame_idx:05d}.jpg"
                    vis_img.save(vis_path)
                print(
                    f"Saved {len(results['visualizations'])} visualization frames to {vis_dir}"
                )

            # Also save a video if visualizations exist
            if save_visualization and results.get("visualizations"):
                try:
                    import cv2

                    video_path = output_path / "output_video.mp4"
                    if len(results["visualizations"]) > 0:
                        # Get dimensions from first frame
                        first_frame = np.array(results["visualizations"][0])
                        h, w = first_frame.shape[:2]
                        # Default FPS (adjust based on your video)
                        fps = 30.0
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        out = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
                        for vis_img in results["visualizations"]:
                            vis_array = np.array(vis_img)
                            # Convert RGB to BGR for OpenCV
                            if vis_array.shape[2] == 3:
                                vis_bgr = vis_array[:, :, ::-1]
                            else:
                                vis_bgr = vis_array
                            out.write(vis_bgr)
                        out.release()
                        print(f"Saved output video to {video_path}")
                except Exception as e:
                    print(f"Warning: Could not save video: {e}")

            if save_masks or save_bboxes:
                import pickle

                data_path = output_path / "results.pkl"
                save_data = {
                    "outputs_per_frame": outputs_per_frame,
                    "session_id": results.get("session_id"),
                }
                with open(data_path, "wb") as f:
                    pickle.dump(save_data, f)
                print(f"Saved raw results to {data_path}")


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAM3Predictor Demo")
    parser.add_argument("--input", type=str, required=True, help="Input image or video")
    parser.add_argument("--text_prompt", type=str, help="Text prompt (e.g., 'person')")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument(
        "--mode", type=str, choices=["image", "video"], default="image", help="Mode"
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="Confidence threshold",
    )
    args = parser.parse_args()

    # Initialize predictor
    predictor = SAM3Predictor(
        mode=args.mode, confidence_threshold=args.confidence_threshold
    )

    # Process input
    results = predictor.predict_from_path(
        args.input, text_prompt=args.text_prompt, return_visualization=True
    )

    # Save results
    predictor.save_results(results, args.output)

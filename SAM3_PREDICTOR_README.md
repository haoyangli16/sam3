# SAM3Predictor Usage Guide

A clean, professional interface for SAM 3 segmentation and tracking.

## Features

- **Image Processing**: Single image segmentation with text/box/point prompts
- **Video Processing**: Video segmentation and tracking with temporal propagation
- **Multiple Prompt Types**: Text prompts, bounding boxes, points (positive/negative)
- **Structured Outputs**: Returns bboxes, masks, scores, and visualizations
- **Thread-Safe**: Safe for concurrent use
- **Easy Saving**: Built-in methods to save results and visualizations

## Quick Start

### Image Mode

```python
from sam3_predictor import SAM3Predictor
from PIL import Image

# Initialize predictor
predictor = SAM3Predictor(mode="image", confidence_threshold=0.5)

# Process image with text prompt
results = predictor.predict_image(
    image="path/to/image.jpg",
    text_prompt="person",
    return_visualization=True
)

# Access results
print(f"Found {len(results['bboxes'])} objects")
print(f"Bboxes: {results['bboxes']}")  # Shape: (N, 4) [x0, y0, x1, y1]
print(f"Masks: {results['masks']}")    # Shape: (N, H, W) binary masks
print(f"Scores: {results['scores']}")  # Shape: (N,) confidence scores

# Save results
predictor.save_results(results, "output_dir")
```

### Video Mode

```python
# Initialize video predictor
predictor = SAM3Predictor(mode="video")

# Process video with text prompt
results = predictor.predict_video(
    video_path="path/to/video.mp4",
    text_prompt="person",
    frame_index=0,
    return_visualization=True,
    propagate=True  # Propagate masks through video
)

# Access results
print(f"Session ID: {results['session_id']}")
print(f"Frames processed: {len(results['outputs_per_frame'])}")

# Each frame's outputs contain masks, obj_ids, scores
for frame_idx, outputs in results['outputs_per_frame'].items():
    print(f"Frame {frame_idx}: {len(outputs)} objects")
    for obj in outputs:
        print(f"  Object ID: {obj['obj_id']}, Score: {obj['score']}")

# Save results
predictor.save_results(results, "output_dir")
```

## API Reference

### Initialization

```python
SAM3Predictor(
    mode: str = "image",                    # "image" or "video"
    checkpoint_path: Optional[str] = None,  # Path to checkpoint (auto-downloads if None)
    bpe_path: Optional[str] = None,         # Path to BPE tokenizer (auto-detects if None)
    confidence_threshold: float = 0.5,      # Detection confidence threshold
    device: Optional[torch.device] = None,  # Device (auto-detects CUDA if None)
    gpus_to_use: Optional[List[int]] = None # GPU list for video mode (all if None)
)
```

### Image Processing

#### `predict_image()`

Process a single image.

**Parameters:**

- `image`: PIL Image, numpy array (H, W, 3) RGB, or path to image file
- `text_prompt`: Optional text prompt (e.g., "person", "shoe")
- `boxes`: Optional list of boxes `[[x, y, w, h], ...]` (top-left, width, height)
- `box_labels`: Optional list of bool labels (True=positive, False=negative)
- `return_visualization`: Whether to generate visualization (default: True)

**Returns:**

```python
{
    "bboxes": np.ndarray,      # (N, 4) [x0, y0, x1, y1]
    "masks": np.ndarray,       # (N, H, W) binary masks
    "scores": np.ndarray,      # (N,) confidence scores
    "visualization": PIL.Image # Visualization with overlays (if requested)
}
```

#### Examples

**Text prompt:**

```python
results = predictor.predict_image(
    image="image.jpg",
    text_prompt="shoe"
)
```

**Box prompt:**

```python
boxes = [[480, 290, 110, 360]]  # [x, y, w, h]
box_labels = [True]  # Positive prompt
results = predictor.predict_image(
    image="image.jpg",
    boxes=boxes,
    box_labels=box_labels
)
```

**Text + multiple boxes (positive/negative):**

```python
boxes = [
    [480, 290, 110, 360],  # Positive
    [370, 280, 115, 375],  # Negative
]
box_labels = [True, False]
results = predictor.predict_image(
    image="image.jpg",
    text_prompt="person",
    boxes=boxes,
    box_labels=box_labels
)
```

### Video Processing

#### `predict_video()`

Process a video with tracking.

**Parameters:**

- `video_path`: Path to video file (.mp4) or directory with JPEG frames
- `text_prompt`: Optional text prompt
- `points`: Optional list of points `[[x, y], ...]` in absolute coordinates
- `point_labels`: Optional list of labels (1=positive, 0=negative)
- `bounding_boxes`: Optional list of boxes `[[x, y, w, h], ...]` in absolute coordinates
- `bounding_box_labels`: Optional list of labels (1=positive, 0=negative)
- `frame_index`: Frame index to add prompt on (default: 0)
- `return_visualization`: Whether to generate visualization (default: True)
- `propagate`: Whether to propagate masks through video (default: True)

**Returns:**

```python
{
    "session_id": str,                    # Session identifier
    "outputs_per_frame": Dict[int, List], # Frame index -> list of object outputs
    "visualizations": List[PIL.Image]     # Visualization frames (if requested)
}
```

Each object output contains:

- `mask`: Binary mask array
- `obj_id`: Object tracking ID
- `score`: Confidence score
- `bbox`: Bounding box (if available)

#### Examples

**Text prompt:**

```python
results = predictor.predict_video(
    video_path="video.mp4",
    text_prompt="person",
    propagate=True
)
```

**Point prompt:**

```python
points = [[500, 300]]  # Click at (x, y)
point_labels = [1]     # Positive point
results = predictor.predict_video(
    video_path="video.mp4",
    points=points,
    point_labels=point_labels,
    propagate=True
)
```

**Box prompt:**

```python
boxes = [[100, 100, 200, 300]]  # [x, y, w, h]
box_labels = [1]  # Positive
results = predictor.predict_video(
    video_path="video.mp4",
    bounding_boxes=boxes,
    bounding_box_labels=box_labels,
    propagate=True
)
```

### Utility Methods

#### `predict_from_path()`

Auto-detect image/video and process accordingly.

```python
results = predictor.predict_from_path(
    input_path="path/to/file.jpg",  # or .mp4
    text_prompt="person",
    return_visualization=True
)
```

#### `save_results()`

Save prediction results to disk.

```python
predictor.save_results(
    results=results,
    output_path="output_dir",
    save_visualization=True,  # Save visualization images
    save_masks=True,          # Save mask arrays
    save_bboxes=True         # Save bounding boxes
)
```

**For images:**

- `visualization.jpg`: Visualization image
- `masks.npy`: Mask arrays
- `bboxes.npy`: Bounding boxes
- `scores.npy`: Confidence scores

**For videos:**

- `frame_XXXXX.jpg`: Visualization frames
- `results.pkl`: Raw results (pickle format)

## Complete Examples

See `examples/sam3_predictor_examples.py` for comprehensive examples:

```bash
# Run all examples
python examples/sam3_predictor_examples.py

# Run specific example
python examples/sam3_predictor_examples.py --example 1
```

## Notes

- **Image mode** uses `Sam3Processor` for efficient single-image processing
- **Video mode** uses `Sam3VideoPredictor` for temporal tracking
- Boxes are in `[x, y, w, h]` format (top-left corner, width, height)
- Coordinates are in absolute pixel values (not normalized)
- The predictor is thread-safe and can be reused for multiple predictions
- Checkpoints are automatically downloaded from HuggingFace if not provided

## Thread Safety

The predictor uses internal locks to ensure thread-safe operation. You can safely use the same predictor instance from multiple threads:

```python
import threading

predictor = SAM3Predictor(mode="image")

def process_image(image_path):
    results = predictor.predict_image(image=image_path, text_prompt="person")
    return results

# Safe to use from multiple threads
threads = []
for img_path in image_paths:
    t = threading.Thread(target=process_image, args=(img_path,))
    threads.append(t)
    t.start()
```

import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import random

def visualize_ground_truth_vs_prediction(model, processor, image_id, image_folder, annotation_file, id2label, 
                                         device, score_threshold=0.7):
    """
    Visualize ground truth vs. DETR predictions side by side for a single image
    
    Args:
        model: DETR model
        processor: DETR image processor
        image_id (int): ID of the image in the COCO annotations
        image_folder (str): Path to image directory
        annotation_file (str): Path to COCO annotation file
        id2label (dict): Mapping from category_id to class name
        device: Device to run inference on
        score_threshold (float): Minimum score threshold to show predictions
    """
    # Import COCO
    try:
        from pycocotools.coco import COCO
    except ImportError:
        raise ImportError("Please install pycocotools: pip install pycocotools")
    
    # Load COCO annotation
    coco = COCO(annotation_file)
    ann_ids = coco.getAnnIds(imgIds=image_id)
    anns = coco.loadAnns(ann_ids)
    img_info = coco.loadImgs(image_id)[0]
    image_path = os.path.join(image_folder, img_info["file_name"])

    # Load and process image
    image = Image.open(image_path).convert("RGB")
    encoding = processor(images=image, return_tensors="pt").to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(**encoding)

    processed = processor.post_process_object_detection(outputs, target_sizes=[image.size[::-1]])[0]
    pred_boxes = processed["boxes"]
    pred_labels = processed["labels"]
    pred_scores = processed["scores"]

    # Setup plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(image)
    axes[0].set_title("Ground Truth")
    axes[1].imshow(image)
    axes[1].set_title("Prediction")

    # Draw ground truth
    for ann in anns:
        bbox = ann["bbox"]  # [x, y, w, h]
        x, y, w, h = bbox
        category_id = ann["category_id"]
        label = id2label[category_id] if category_id in id2label else str(category_id)
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="g", facecolor="none")
        axes[0].add_patch(rect)
        axes[0].text(x, y - 5, label, color="white", backgroundcolor="green", fontsize=10)

    # Draw predictions
    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        if score < score_threshold:
            continue
        x1, y1, x2, y2 = box.tolist()
        w, h = x2 - x1, y2 - y1
        class_name = id2label[label.item()] if label.item() in id2label else str(label.item())
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor="r", facecolor="none")
        axes[1].add_patch(rect)
        axes[1].text(x1, y1 - 5, f"{class_name}: {score:.2f}", color="white", backgroundcolor="red", fontsize=10)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def visualize_predictions(model, processor, image_path, id2label, device, score_threshold=0.7, save_path=None):
    """
    Visualize predictions on a single image
    
    Args:
        model: DETR model
        processor: DETR image processor
        image_path (str): Path to the image file
        id2label (dict): Mapping from category_id to class name
        device: Device to run inference on
        score_threshold (float): Score threshold to filter predictions
        save_path (str, optional): Path to save visualization to
    """
    model.eval()
    image = Image.open(image_path).convert("RGB")
    encoding = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**encoding)

    results = processor.post_process_object_detection(
        outputs, 
        target_sizes=[image.size[::-1]], 
        threshold=score_threshold
    )[0]
    
    boxes = results["boxes"]
    labels = results["labels"]
    scores = results["scores"]

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)

    for box, label, score in zip(boxes, labels, scores):
        if score < score_threshold:
            continue

        x1, y1, x2, y2 = box.tolist()
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        class_name = id2label[label.item()] if label.item() in id2label else f"class {label.item()}"
        ax.text(x1, y1 - 5, f"{class_name}: {score:.2f}", color="white", fontsize=12,
                bbox=dict(facecolor="red", alpha=0.5))

    plt.axis('off')
    plt.title("Predictions")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()

def visualize_random_sample(dataset, id2label):
    """
    Visualize a random sample from the dataset with annotations
    
    Args:
        dataset: COCO dataset
        id2label (dict): Mapping from category_id to class name
    """
    try:
        import cv2
        import supervision as sv
    except ImportError:
        raise ImportError("Please install OpenCV and supervision: pip install opencv-python supervision")
    
    # Select random image
    image_ids = dataset.coco.getImgIds()
    image_id = random.choice(image_ids)
    print(f'Image #{image_id}')

    # Load image and annotations
    image = dataset.coco.loadImgs(image_id)[0]
    annotations = dataset.coco.imgToAnns[image_id]
    image_path = os.path.join(dataset.root, image['file_name'])
    image = cv2.imread(image_path)

    # Annotate
    detections = sv.Detections.from_coco_annotations(coco_annotation=annotations)
    labels = [f"{id2label[class_id]}" for _, _, class_id, _ in detections]

    box_annotator = sv.BoxAnnotator()
    frame = box_annotator.annotate(scene=image, detections=detections, labels=labels)

    # Convert from BGR to RGB for matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(16, 16))
    plt.imshow(frame_rgb)
    plt.axis('off')
    plt.title(f'Sample Image with Annotations (ID: {image_id})')
    plt.show()

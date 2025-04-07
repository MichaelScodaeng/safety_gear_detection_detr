import os
import torch
import numpy as np
from tqdm import tqdm
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

def convert_to_xywh(boxes):
    """Convert bounding boxes from [x1, y1, x2, y2] to [x, y, w, h] format"""
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def prepare_for_coco_detection(predictions):
    """Prepare predictions for COCO evaluation"""
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results

def train_model(model, config):
    """
    Train the DETR model using PyTorch Lightning
    
    Args:
        model: DETR model
        config (dict): Configuration dictionary
        
    Returns:
        Trainer: PyTorch Lightning trainer
    """
    # Extract training configuration
    max_epochs = config['training']['max_epochs']
    gradient_clip_val = config['training']['gradient_clip_val']
    accumulate_grad_batches = config['training']['accumulate_grad_batches']
    log_every_n_steps = config['training']['log_every_n_steps']
    checkpoint_dir = config['training']['checkpoint_dir']
    
    # Create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='detr-{epoch:02d}-{validation/loss:.2f}',
        save_top_k=3,
        monitor='validation/loss',
        mode='min'
    )
    
    # Create trainer
    trainer = Trainer(
        devices=1 if torch.cuda.is_available() else None,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        max_epochs=max_epochs,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=log_every_n_steps,
        callbacks=[checkpoint_callback]
    )
    
    # Train model
    trainer.fit(model)
    
    return trainer

def evaluate_model(model, test_dataloader, test_dataset, processor, config, device):
    """
    Evaluate the DETR model on the test set
    
    Args:
        model: DETR model
        test_dataloader: Test data loader
        test_dataset: Test dataset
        processor: DETR image processor
        config (dict): Configuration dictionary
        device: Device to run evaluation on
        
    Returns:
        dict: Evaluation results
    """
    # Import here to avoid dependency issues
    try:
        from coco_eval import CocoEvaluator
    except ImportError:
        raise ImportError("Please install coco_eval: pip install coco_eval")
    
    print("Running evaluation...")
    
    # Create evaluator
    evaluator = CocoEvaluator(coco_gt=test_dataset.coco, iou_types=["bbox"])
    
    # Evaluate model
    model.to(device)
    model.eval()
    
    for idx, batch in enumerate(tqdm(test_dataloader)):
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
        results = processor.post_process_object_detection(outputs, target_sizes=orig_target_sizes)

        predictions = {target['image_id'].item(): output for target, output in zip(labels, results)}
        predictions = prepare_for_coco_detection(predictions)
        evaluator.update(predictions)

    # Aggregate results
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    results = evaluator.summarize()
    
    return results

def save_model(model, path):
    """
    Save the DETR model
    
    Args:
        model: DETR model
        path (str): Path to save the model to
    """
    # Create directory if it doesn't exist
    os.makedirs(path, exist_ok=True)
    
    # Save model
    model.model.save_pretrained(path)
    print(f"Model saved to {path}")

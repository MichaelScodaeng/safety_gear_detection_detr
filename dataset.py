import os
import torch
import torchvision
from torch.utils.data import DataLoader
from PIL import Image

class CocoDataset(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, processor):
        """
        COCO dataset for DETR model
        
        Args:
            img_folder (str): Path to images directory
            ann_file (str): Path to COCO annotation file
            processor (DetrImageProcessor): DETR image processor
        """
        super().__init__(img_folder, ann_file)
        self.processor = processor

    def __getitem__(self, idx):
        image, targets = super().__getitem__(idx)
        
        # Extract image_id
        image_id = self.ids[idx]
        
        # Wrap properly for DetrImageProcessor
        annotation_dict = {
            "image_id": image_id,
            "annotations": targets
        }

        try:
            encoding = self.processor(images=image, annotations=annotation_dict, return_tensors="pt")
            pixel_values = encoding["pixel_values"].squeeze(0)
            target = encoding["labels"][0]
            return pixel_values, target
        except Exception as e:
            print(f"[ERROR during processing item {idx}]")
            print(e)
            raise

def collate_fn(batch):
    """
    Collate function for DataLoader that properly handles DETR's variable image sizes
    
    Args:
        batch: Batch of data points
        
    Returns:
        dict: Batch with pixel_values, pixel_mask, and labels
    """
    pixel_values = [item[0] for item in batch]
    # Get the first processor from the first item's attribute
    processor = batch[0][1].get_processor() if hasattr(batch[0][1], 'get_processor') else None
    
    if processor:
        encoding = processor.pad(pixel_values, return_tensors="pt")
    else:
        # Fallback if processor not available
        from transformers import DetrImageProcessor
        processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')
        encoding = processor.pad(pixel_values, return_tensors="pt")
        
    labels = [item[1] for item in batch]
    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }

def create_dataloaders(config, processor):
    """
    Create DataLoaders for training, validation, and testing
    
    Args:
        config (dict): Configuration dictionary
        processor (DetrImageProcessor): DETR image processor
        
    Returns:
        tuple: train_dataloader, val_dataloader, test_dataloader, id2label
    """
    train_dir = config['dataset']['train_dir']
    val_dir = config['dataset']['val_dir']
    test_dir = config['dataset']['test_dir']
    ann_file = config['dataset']['annotation_file']
    batch_size = config['dataset']['batch_size']
    num_workers = config['dataset']['num_workers']
    
    # Create annotation file paths
    train_ann = os.path.join(train_dir, ann_file)
    val_ann = os.path.join(val_dir, ann_file)
    test_ann = os.path.join(test_dir, ann_file)
    
    # Create datasets
    train_dataset = CocoDataset(
        img_folder=train_dir,
        ann_file=train_ann,
        processor=processor
    )
    
    val_dataset = CocoDataset(
        img_folder=val_dir,
        ann_file=val_ann,
        processor=processor
    )
    
    test_dataset = CocoDataset(
        img_folder=test_dir,
        ann_file=test_ann,
        processor=processor
    )
    
    print(f"Number of training examples: {len(train_dataset)}")
    print(f"Number of validation examples: {len(val_dataset)}")
    print(f"Number of test examples: {len(test_dataset)}")
    
    # Create dataloaders
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        collate_fn=collate_fn, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers
    )
    
    val_dataloader = DataLoader(
        dataset=val_dataset, 
        collate_fn=collate_fn, 
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    test_dataloader = DataLoader(
        dataset=test_dataset, 
        collate_fn=collate_fn, 
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Create id2label mapping
    categories = train_dataset.coco.cats
    id2label = {k: v['name'] for k, v in categories.items()}
    
    return train_dataloader, val_dataloader, test_dataloader, id2label, train_dataset, val_dataset, test_dataset

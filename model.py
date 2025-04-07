import torch
import pytorch_lightning as pl
from transformers import DetrForObjectDetection, DetrImageProcessor

class DETR(pl.LightningModule):
    """
    PyTorch Lightning implementation of DETR (DEtection TRansformer)
    """
    def __init__(self, config, num_labels, train_dataloader=None, val_dataloader=None):
        """
        Initialize DETR model
        
        Args:
            config (dict): Configuration dictionary
            num_labels (int): Number of class labels
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
        """
        super().__init__()
        self.save_hyperparameters(ignore=['train_dataloader', 'val_dataloader'])
        
        # Load model configuration
        self.model_name = config['model']['name']
        self.lr = config['model']['lr']
        self.lr_backbone = config['model']['lr_backbone']
        self.weight_decay = config['model']['weight_decay']
        
        # Create model
        self.model = DetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        
        # Store dataloaders
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
    
    def forward(self, pixel_values, pixel_mask):
        """Forward pass through the model"""
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
    
    def common_step(self, batch, batch_idx):
        """Common step for training and validation"""
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        loss, loss_dict = self.common_step(batch, batch_idx)
        # Log metrics for each training_step, and the average across the epoch
        self.log("training_loss", loss)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation/loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())

        return loss
    
    def configure_optimizers(self):
        """Configure optimizers for training"""
        # DETR authors decided to use different learning rate for backbone
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]
            },
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        return torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)
    
    def train_dataloader(self):
        """Return training dataloader"""
        return self._train_dataloader
    
    def val_dataloader(self):
        """Return validation dataloader"""
        return self._val_dataloader

def create_model(config, num_labels, train_dataloader=None, val_dataloader=None):
    """
    Create DETR model
    
    Args:
        config (dict): Configuration dictionary
        num_labels (int): Number of class labels
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        
    Returns:
        DETR: DETR model
    """
    model = DETR(
        config=config, 
        num_labels=num_labels,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader
    )
    return model

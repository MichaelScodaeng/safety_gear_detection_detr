import os
import yaml
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection

def load_config(config_path):
    """
    Load configuration from YAML file and convert numeric strings to numbers
    
    Args:
        config_path (str): Path to YAML configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert string numbers to actual numbers
    config['model']['lr'] = float(config['model']['lr'])
    config['model']['lr_backbone'] = float(config['model']['lr_backbone'])
    config['model']['weight_decay'] = float(config['model']['weight_decay'])
    
    return config

def setup_device():
    """
    Set up device for training and inference
    
    Returns:
        device: PyTorch device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("GPU not available, using CPU")
    
    return device

def load_processor(model_name):
    """
    Load DETR image processor
    
    Args:
        model_name (str): Model name or path
        
    Returns:
        DetrImageProcessor: DETR image processor
    """
    processor = DetrImageProcessor.from_pretrained(model_name)
    return processor

def load_pretrained_model(model_path, device):
    """
    Load pretrained DETR model
    
    Args:
        model_path (str): Path to pretrained model
        device: Device to load model to
        
    Returns:
        DetrForObjectDetection: Pretrained DETR model
    """
    model = DetrForObjectDetection.from_pretrained(model_path)
    model.to(device)
    return model

def create_directories(config):
    """
    Create necessary directories for training and inference
    
    Args:
        config (dict): Configuration dictionary
    """
    # Create checkpoint directory
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    
    # Create visualization directory
    if config['inference']['save_visualizations']:
        os.makedirs(config['inference']['visualization_dir'], exist_ok=True)

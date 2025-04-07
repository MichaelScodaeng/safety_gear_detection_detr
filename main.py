import argparse
import os
import random
import torch

# Import modules
from utils import load_config, setup_device, load_processor, create_directories
from dataset import create_dataloaders
from model import create_model
from trainer import train_model, evaluate_model, save_model
from visualization import visualize_ground_truth_vs_prediction, visualize_predictions, visualize_random_sample

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='DETR Training and Inference')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'visualize'], default='train', 
                        help='Operation mode')
    parser.add_argument('--checkpoint', type=str, default=None, 
                        help='Path to checkpoint for evaluation or visualization')
    parser.add_argument('--image_path', type=str, default=None, 
                        help='Path to image for visualization')
    return parser.parse_args()
def display_cuda_info():
    """Display CUDA information"""
    if torch.cuda.is_available():
        print("CUDA is available")
        print("Current device:", torch.cuda.current_device())
        print("CUDA device name:", torch.cuda.get_device_name(0))
        print("CUDA device count:", torch.cuda.device_count())
    else:
        print("CUDA is not available, using CPU")
def set_random_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed} for reproducibility")
def main():
    
    """Main function for DETR training, evaluation, and visualization"""
    # Set random seed for reproducibility
    set_random_seed(42)
    # Display CUDA information
    display_cuda_info()
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up device
    device = setup_device()
    
    # Create directories
    create_directories(config)
    
    # Load processor
    processor = load_processor(config['model']['name'])
    
    # Create dataloaders
    train_dataloader, val_dataloader, test_dataloader, id2label, train_dataset, val_dataset, test_dataset = create_dataloaders(config, processor)
    
    print(f"Loaded {len(id2label)} classes: {id2label}")
    
    if args.mode == 'train':
        print("Starting training...")
        # Create model
        model = create_model(
            config=config, 
            num_labels=len(id2label),
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader
        )
        
        # Train model
        trainer = train_model(model, config)
        
        # Evaluate model
        print("Evaluating trained model...")
        evaluate_model(model, test_dataloader, test_dataset, processor, config, device)
        
        # Save model
        model_path = os.path.join(config['training']['checkpoint_dir'], 'final-model')
        save_model(model, model_path)
        
        # Visualize a random sample with predictions after training
        print("Visualizing a random sample with predictions...")
        image_ids = test_dataset.coco.getImgIds()
        random_image_id = random.choice(image_ids)
        visualize_ground_truth_vs_prediction(
            model=model,
            processor=processor,
            image_id=random_image_id,
            image_folder=config['dataset']['test_dir'],
            annotation_file=os.path.join(config['dataset']['test_dir'], config['dataset']['annotation_file']),
            id2label=id2label,
            device=device,
            score_threshold=config['inference']['confidence_threshold']
        )
        
    elif args.mode == 'evaluate':
        # Load model from checkpoint
        if args.checkpoint is None:
            raise ValueError("Please provide a checkpoint path for evaluation")
        
        from transformers import DetrForObjectDetection
        model = DetrForObjectDetection.from_pretrained(args.checkpoint, num_labels=len(id2label))
        
        # Evaluate model
        evaluate_model(model, test_dataloader, test_dataset, processor, config, device)
        
    elif args.mode == 'visualize':
        # Load model from checkpoint
        if args.checkpoint is None:
            raise ValueError("Please provide a checkpoint path for visualization")
        
        from transformers import DetrForObjectDetection
        model = DetrForObjectDetection.from_pretrained(args.checkpoint, num_labels=len(id2label))
        model.to(device)
        
        if args.image_path:
            # Visualize predictions on a specific image
            print(f"Visualizing predictions on {args.image_path}...")
            save_path = None
            if config['inference']['save_visualizations']:
                image_name = os.path.basename(args.image_path)
                save_path = os.path.join(config['inference']['visualization_dir'], f"pred_{image_name}")
            
            visualize_predictions(
                model=model,
                processor=processor,
                image_path=args.image_path,
                id2label=id2label,
                device=device,
                score_threshold=config['inference']['confidence_threshold'],
                save_path=save_path
            )
        else:
            # Visualize random samples from the test set
            print("Visualizing random samples from the test set...")
            
            # Visualize a random sample with ground truth annotations
            visualize_random_sample(test_dataset, id2label)
            
            # Visualize ground truth vs. predictions
            image_ids = test_dataset.coco.getImgIds()
            random_image_id = random.choice(image_ids)
            visualize_ground_truth_vs_prediction(
                model=model,
                processor=processor,
                image_id=random_image_id,
                image_folder=config['dataset']['test_dir'],
                annotation_file=os.path.join(config['dataset']['test_dir'], config['dataset']['annotation_file']),
                id2label=id2label,
                device=device,
                score_threshold=config['inference']['confidence_threshold']
            )
if __name__ == "__main__":
    main()
'''
# For training
python main.py --config config.yaml --mode train

# For evaluation
python main.py --config config.yaml --mode evaluate --checkpoint path/to/model

# For visualization
python main.py --config config.yaml --mode visualize --checkpoint path/to/model --image_path path/to/image.jpg'''
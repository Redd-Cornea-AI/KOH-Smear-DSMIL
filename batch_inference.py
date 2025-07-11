import os
import sys
import glob
import argparse
import numpy as np
import torch
from inference import WSIInference

def evaluate_dataset(pipeline, test_dir, output_dir=None):
    """Evaluate model performance on a dataset of WSI slides.
    
    Args:
        pipeline: Initialized WSIInference pipeline
        test_dir: Directory containing Fungal_Positive and Fungal_Negative folders
        output_dir: Optional directory to save attention heatmaps
    """
    results = []
    confusion_matrix = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    
    # Process negative cases
    neg_dir = os.path.join(test_dir, "Fungal_Negative")
    for slide_file in sorted(glob.glob(os.path.join(neg_dir, "*.svs"))):
        try:
            prediction, probability, _ = pipeline.process_slide(slide_file, output_dir)
            results.append({
                "slide": os.path.basename(slide_file),
                "true_label": "Negative",
                "predicted": "Positive" if prediction else "Negative",
                "probability": probability
            })
            if not prediction:  # True Negative
                confusion_matrix["TN"] += 1
            else:  # False Positive
                confusion_matrix["FP"] += 1
            print(f"Processed negative slide: {os.path.basename(slide_file)} - Prediction: {'Positive' if prediction else 'Negative'} ({probability:.4f})")
        except Exception as e:
            print(f"Error processing slide {slide_file}: {e}")
    
    # Process positive cases
    pos_dir = os.path.join(test_dir, "Fungal_Positive")
    for slide_file in sorted(glob.glob(os.path.join(pos_dir, "*.svs"))):
        try:
            prediction, probability, _ = pipeline.process_slide(slide_file, output_dir)
            results.append({
                "slide": os.path.basename(slide_file),
                "true_label": "Positive",
                "predicted": "Positive" if prediction else "Negative",
                "probability": probability
            })
            if prediction:  # True Positive
                confusion_matrix["TP"] += 1
            else:  # False Negative
                confusion_matrix["FN"] += 1
            print(f"Processed positive slide: {os.path.basename(slide_file)} - Prediction: {'Positive' if prediction else 'Negative'} ({probability:.4f})")
        except Exception as e:
            print(f"Error processing slide {slide_file}: {e}")
    
    return results, confusion_matrix

def print_metrics(results, confusion_matrix):
    """Print evaluation metrics."""
    total = len(results)
    correct = confusion_matrix["TP"] + confusion_matrix["TN"]
    accuracy = correct / total if total > 0 else 0
    sensitivity = confusion_matrix["TP"] / (confusion_matrix["TP"] + confusion_matrix["FN"]) if (confusion_matrix["TP"] + confusion_matrix["FN"]) > 0 else 0
    specificity = confusion_matrix["TN"] / (confusion_matrix["TN"] + confusion_matrix["FP"]) if (confusion_matrix["TN"] + confusion_matrix["FP"]) > 0 else 0
    
    print("\nTest Results:")
    print("-" * 50)
    print(f"Total slides processed: {total}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Sensitivity: {sensitivity:.2%}")
    print(f"Specificity: {specificity:.2%}")
    print("\nConfusion Matrix:")
    print(f"True Positives: {confusion_matrix['TP']}")
    print(f"True Negatives: {confusion_matrix['TN']}")
    print(f"False Positives: {confusion_matrix['FP']}")
    print(f"False Negatives: {confusion_matrix['FN']}")
    
    print("\nDetailed Results:")
    print("-" * 50)
    for result in results:
        print(f"Slide: {result['slide']}")
        print(f"True Label: {result['true_label']}")
        print(f"Predicted: {result['predicted']}")
        print(f"Probability: {result['probability']:.4f}")
        print("-" * 30)

def main():
    parser = argparse.ArgumentParser(description='Batch WSI Inference for Fungal Keratitis Detection')
    parser.add_argument('--embedder_low', required=True, help='Path to low magnification embedder weights')
    parser.add_argument('--embedder_high', required=True, help='Path to high magnification embedder weights')
    parser.add_argument('--aggregator', required=True, help='Path to aggregator model weights')
    parser.add_argument('--output_dir', help='Directory to save attention heatmaps')
    parser.add_argument('--tile_size', type=int, default=224, help='Tile size for patch extraction')
    parser.add_argument('--background_threshold', type=int, default=7, help='Background threshold')
    parser.add_argument('--base_mag', type=float, default=20, help='Base magnification')
    parser.add_argument('--detection_threshold', type=float, default=0.5, 
                      help='Threshold for positive detection (default: 0.5)')
    parser.add_argument('--test_dir', default='WSI/KOH_Dataset_test_lambda',
                      help='Directory containing Fungal_Positive and Fungal_Negative folders')
    
    args = parser.parse_args()
    
    # Initialize inference pipeline
    pipeline = WSIInference(
        embedder_low_path=args.embedder_low,
        embedder_high_path=args.embedder_high,
        aggregator_path=args.aggregator,
        tile_size=args.tile_size,
        background_threshold=args.background_threshold,
        base_mag=args.base_mag,
        detection_threshold=args.detection_threshold
    )
    
    # Run evaluation
    results, confusion_matrix = evaluate_dataset(pipeline, args.test_dir, args.output_dir)
    
    # Print metrics
    print_metrics(results, confusion_matrix)

if __name__ == '__main__':
    main() 
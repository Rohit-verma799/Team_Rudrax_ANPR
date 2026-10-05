"""
CLI Tool to run raw OCR on a specified license plate image crop.
"""

import os
import argparse
import sys
import cv2

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from plate_ocr import PlateOCR


def parse_args():
    parser = argparse.ArgumentParser(description="Run Raw OCR on a License Plate Image")
    parser.add_argument(
        "--image", 
        required=True, 
        help="Path to the license plate crop image file"
    )
    parser.add_argument(
        "--gpu", 
        type=lambda x: x.lower() == 'true', 
        default=False, 
        help="Use GPU acceleration (true/false)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image file not found at '{args.image}'")
        sys.exit(1)
        
    print(f"Initializing PlateOCR engine (GPU: {args.gpu})...")
    ocr = PlateOCR(use_gpu=args.gpu)
    
    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image from '{args.image}'")
        sys.exit(1)
        
    # Preprocess image
    processed = ocr.preprocess_plate(image)
    
    print(f"Running Raw OCR on '{args.image}'...")
    raw_results = ocr._run_ocr(ocr.ocr_en, processed)
    
    # Run the cleaned OCR for comparison
    cleaned_results = ocr.extract_text(image)
    
    print("\n" + "=" * 50)
    print("                    OCR RESULTS")
    print("=" * 50)
    print(f"Image Path: {args.image}")
    print("-" * 50)
    
    if not raw_results:
        print("Raw OCR Output: (No text detected)")
    else:
        print("Raw OCR Detections:")
        for idx, (text, conf) in enumerate(raw_results):
            print(f"  [{idx + 1}] Text: '{text}' (Confidence: {conf:.4f})")
            
    print("-" * 50)
    print(f"Cleaned Indian Plate Format: '{cleaned_results.get('text', '')}'")
    print(f"Average Confidence:           {cleaned_results.get('confidence', 0.0):.4f}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()

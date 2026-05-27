import argparse
import os
import cv2
from model import identify_faces

def main():
    parser = argparse.ArgumentParser(description="Predict / Identify faces in a given image")
    parser.add_argument("image", help="Path to the image file to run prediction on")
    parser.add_argument("--model", default="face_model.pkl", help="Path to the trained model (.pkl)")
    parser.add_argument("--threshold", type=float, default=0.53, help="Confidence threshold (default: 0.53)")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Query image not found at {args.image}")
        return

    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}. Please run train.py first.")
        return

    print(f"Loading image and running face identification...")
    try:
        results = identify_faces(image_path=args.image, model_path=args.model, threshold=args.threshold)
        
        if not results:
            print("No faces detected in the image.")
            return

        print("\n=== Identification Results ===")
        for idx, res in enumerate(results):
            loc = res['location']
            print(f"Face {idx + 1}:")
            print(f"  Name: {res['name']}")
            print(f"  Confidence Distance: {res['distance']}")
            print(f"  Location Box: {loc} (Top: {loc[0]}, Right: {loc[1]}, Bottom: {loc[2]}, Left: {loc[3]})")
            print("-" * 30)
            
    except Exception as e:
        print(f"Prediction failed with error: {e}")

if __name__ == "__main__":
    main()

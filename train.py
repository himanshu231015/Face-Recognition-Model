import sys
import argparse
from model import train_model

def main():
    parser = argparse.ArgumentParser(description="Train Face Recognition Model (KNN)")
    parser.add_argument("--dataset", default="dataset", help="Path to the dataset folder containing subfolders for each person (e.g., 'John_Doe')")
    parser.add_argument("--model", default="face_model.pkl", help="Output file path for the trained model classifier")
    parser.add_argument("--cache", default="encodings_cache.pkl", help="File path to save the face encodings cache")
    args = parser.parse_args()

    print(f"Starting training with dataset: {args.dataset}")
    success, msg = train_model(dataset_dir=args.dataset, model_path=args.model, cache_path=args.cache)
    print(msg)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()

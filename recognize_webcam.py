import argparse
import os
import cv2
import sys
import time
from model import identify_faces

def main():
    parser = argparse.ArgumentParser(description="Real-time Face Recognition from Webcam")
    parser.add_argument("--model", default="face_model.pkl", help="Path to the trained model (.pkl)")
    parser.add_argument("--threshold", type=float, default=0.53, help="Confidence threshold (default: 0.53)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[ERROR] Trained model file not found at '{args.model}'.")
        print("Please register a face (python register_face.py) and train the model first.")
        sys.exit(1)

    print("Loading face recognition model...")
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not access webcam. Please check connection.")
        sys.exit(1)

    print("\nWebcam started successfully! Press 'Q' to quit.")

    # Variables for calculating FPS
    prev_frame_time = 0
    new_frame_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from webcam.")
            break

        # Flip horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert BGR (OpenCV default) to RGB (face_recognition requirement)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run face identification
        try:
            results = identify_faces(image_content=rgb_frame, model_path=args.model, threshold=args.threshold)
        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")
            break

        # Draw bounding boxes and names
        for res in results:
            name = res['name']
            dist = res['distance']
            top, right, bottom, left = res['location']

            # Choose color: Neon Cyan for known, Red for unknown
            if name != "Unknown":
                # Convert distance to a confidence percentage (rough estimation for UI display)
                # Lower distance = Higher confidence. At threshold 0.53, let's map it:
                # 0.0 distance -> 100% confidence
                # 0.53 distance -> ~60% confidence
                # >0.60 distance -> 0%
                conf_pct = int(max(0, min(100, (1.0 - (dist / 1.3)) * 100)))
                display_text = f"{name} ({conf_pct}%)"
                box_color = (0, 255, 0)  # Green
            else:
                display_text = "Unknown"
                box_color = (0, 0, 255)  # Red

            # Draw elegant rounded bounding box corners
            # Main box
            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)

            # Draw text label background
            lbl_w, lbl_h = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (left, top - 30), (left + lbl_w + 10, top), box_color, -1)
            
            # Draw text label
            cv2.putText(frame, display_text, (left + 5, top - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Calculate & Display FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps_text = f"FPS: {int(fps)}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Show frame
        cv2.imshow("Face Matcher (Webcam)", frame)

        # Quit check
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")

if __name__ == "__main__":
    main()

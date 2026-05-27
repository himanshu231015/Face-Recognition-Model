import os
import cv2
import sys
import time
import face_recognition
from model import train_model

def get_valid_name():
    while True:
        name = input("\nEnter the name of the person to register: ").strip()
        if not name:
            print("Name cannot be empty. Please try again.")
            continue
        # Replace spaces and special characters to make a safe directory name
        safe_name = name.replace(" ", "_")
        safe_name = "".join([c for c in safe_name if c.isalnum() or c == "_"])
        if not safe_name:
            print("Invalid name. Please use alphanumeric characters and spaces.")
            continue
        return name, safe_name

def main():
    print("==================================================")
    print("          FACE REGISTRATION MODULE                ")
    print("==================================================")
    
    name, folder_name = get_valid_name()
    dataset_dir = "dataset"
    save_dir = os.path.join(dataset_dir, folder_name)
    
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("\n[ERROR] Could not access the webcam. Please check your camera connection.")
        sys.exit(1)
        
    print("\nWebcam started successfully!")
    print("Instructions:")
    print("1. Align your face inside the screen.")
    print("2. Press 'SPACE' to capture a photo (Try different expressions/angles).")
    print("3. You need to capture exactly 5 photos.")
    print("4. Press 'Q' at any time to cancel and exit.")
    
    captured_count = 0
    required_captures = 5
    flash_timer = 0
    error_msg = ""
    error_timer = 0
    
    while captured_count < required_captures:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame from webcam.")
            break
            
        # Flip frame horizontally for natural mirror view
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()
        h, w, _ = frame.shape
        
        # Draw UI overlay
        # Header banner
        cv2.rectangle(display_frame, (0, 0), (w, 50), (45, 45, 45), -1)
        cv2.putText(display_frame, f"Registering: {name}", (20, 32), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Footer banner
        cv2.rectangle(display_frame, (0, h - 60), (w, h), (45, 45, 45), -1)
        cv2.putText(display_frame, "SPACE: Capture  |  Q: Quit", (20, h - 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(display_frame, f"Captured: {captured_count}/{required_captures}", (w - 180, h - 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if captured_count > 0 else (255, 255, 255), 2)
        
        # Display capture flash effect
        if time.time() - flash_timer < 0.15:
            # White flash overlay
            flash_overlay = display_frame.copy()
            flash_overlay[:] = (255, 255, 255)
            cv2.addWeighted(flash_overlay, 0.5, display_frame, 0.5, 0, display_frame)
            
        # Display temporary error messages (e.g. no face detected)
        if error_msg and time.time() - error_timer < 2.0:
            cv2.rectangle(display_frame, (w // 2 - 180, h // 2 - 25), (w // 2 + 180, h // 2 + 25), (0, 0, 150), -1)
            cv2.putText(display_frame, error_msg, (w // 2 - 160, h // 2 + 6), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        else:
            error_msg = ""
            
        # Draw a guidelines box in the center
        box_w, box_h = 240, 280
        top_left = (w // 2 - box_w // 2, h // 2 - box_h // 2)
        bottom_right = (w // 2 + box_w // 2, h // 2 + box_h // 2)
        cv2.rectangle(display_frame, top_left, bottom_right, (255, 180, 0), 2)
        
        cv2.imshow("Face Registration", display_frame)
        key = cv2.waitKey(1) & 0xFF
        
        # Quit key
        if key == ord('q') or key == ord('Q'):
            print("\nRegistration cancelled by user.")
            break
            
        # Capture key
        if key == 32:  # SPACE bar
            # Show a processing message
            cv2.putText(display_frame, "Processing...", (w // 2 - 60, h // 2 + box_h // 2 + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Face Registration", display_frame)
            cv2.waitKey(10)
            
            # Detect face in the original frame (flip back to match original frame orientation if saving)
            # Actually saving the flipped frame is completely fine, but let's make sure it contains a face.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            
            if len(face_locations) == 0:
                error_msg = "No face detected! Please center your face."
                error_timer = time.time()
                print("[WARNING] Capture failed: No face detected in frame.")
            elif len(face_locations) > 1:
                error_msg = "Multiple faces detected! Only 1 person allowed."
                error_timer = time.time()
                print("[WARNING] Capture failed: Multiple faces detected.")
            else:
                captured_count += 1
                # Save the frame
                img_filename = f"face_{captured_count:02d}.jpg"
                img_path = os.path.join(save_dir, img_filename)
                
                # Write to disk
                cv2.imwrite(img_path, frame)
                flash_timer = time.time()
                print(f"[SUCCESS] Captured photo {captured_count}/{required_captures} saved to {img_path}")
                time.sleep(0.3)  # Brief pause to let user change expressions
                
    cap.release()
    cv2.destroyAllWindows()
    
    if captured_count == required_captures:
        print("\n==================================================")
        print(f"🎉 Face registration completed successfully for '{name}'!")
        print(f"Saved {required_captures} images to: {save_dir}")
        print("==================================================")
        
        # Prompt for immediate training
        ans = input("\nDo you want to train the model right now? (y/n): ").strip().lower()
        if ans == 'y' or ans == 'yes':
            print("\nTraining the model, please wait...")
            success, msg = train_model(dataset_dir=dataset_dir)
            print(msg)
            if success:
                print("\n👍 System is ready to match/recognize your face!")
            else:
                print("\n❌ Training failed. Please check the dataset and try again.")
        else:
            print("\nRemember to train the model using: python train.py")
    else:
        print("\n❌ Registration incomplete. Please try again.")

if __name__ == "__main__":
    main()

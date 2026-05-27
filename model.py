import os
import pickle
import re
import cv2
import face_recognition
from sklearn import neighbors

def train_model(dataset_dir="dataset", model_path="face_model.pkl", cache_path="encodings_cache.pkl"):
    """
    Trains a KNN classifier for face recognition.
    
    Args:
        dataset_dir: Directory containing subdirectories of person faces named like 'Person_Name' or 'Name'
        model_path: Path where the trained classifier data will be saved
        cache_path: Path where extracted face encodings cache will be saved
    """
    X = []
    y = []
    
    encodings_cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                encodings_cache = pickle.load(f)
            print(f"Loaded {len(encodings_cache)} cached encodings.")
        except Exception as e:
            print(f"Error loading cache: {e}")
            encodings_cache = {}

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        print(f"Created empty dataset directory at: {dataset_dir}")
        return False, "Dataset directory was empty. Please add images and train again."
        
    active_files = set()
    new_encodings_count = 0
    
    # Traverse dataset directory
    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
            
        # Check for image files
        has_images = False
        for f in os.listdir(person_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                has_images = True
                break
        
        if not has_images:
            continue
        
        # Display name (replacing underscores with space)
        display_name = person_name.replace('_', ' ')
        print(f"Processing folder: {person_name} (Name: {display_name})")
        
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            rel_path = os.path.relpath(image_path, dataset_dir)
            
            if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            active_files.add(rel_path)
                
            if rel_path in encodings_cache:
                X.append(encodings_cache[rel_path])
                y.append(person_name)
            else:
                try:
                    image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(image)
                    
                    if len(face_encodings) > 0:
                        encoding = face_encodings[0]
                        X.append(encoding)
                        y.append(person_name)
                        encodings_cache[rel_path] = encoding
                        new_encodings_count += 1
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
    
    if not X:
        return False, "No face encodings could be extracted. Make sure images contain clear frontal faces."

    # Cache Cleanup: Remove deleted files from cache
    clean_cache = {k: v for k, v in encodings_cache.items() if k in active_files}
    
    # Save cache
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(clean_cache, f)
        print(f"Cache updated. New: {new_encodings_count}, Total Cached: {len(clean_cache)}")
    except Exception as e:
        print(f"Error saving cache: {e}")
        
    # Train KNN (1-NN is used to map to the exact closest matching profile)
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree', weights='distance')
    knn_clf.fit(X, y)
    
    # Save self-contained model
    model_data = {
        'classifier': knn_clf
    }
    
    os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
        
    return True, f"Model trained successfully! Encodings: {len(X)}. Saved to: {model_path}"

def identify_faces(image_path=None, image_content=None, model_path="face_model.pkl", threshold=0.53):
    """
    Identifies faces in a given image using the trained KNN model.
    
    Args:
        image_path: Path to the query image file
        image_content: Optional RGB numpy image array (alternative to image_path)
        model_path: Path to the trained model pickle file
        threshold: Strictness of match (lower = stricter; 0.53 is balanced)
        
    Returns:
        List of dictionaries with name, roll_number, location, and distance values
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Please train the model first.")
        
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
        
    knn_clf = model_data['classifier']
    names_dict = model_data.get('names', {})
        
    if image_content is not None:
        image = image_content
    else:
        image = face_recognition.load_image_file(image_path)
    
    # 1. Resize image for faster face detection (2x downscaling)
    scale_factor = 2
    height, width = image.shape[:2]
    small_image = cv2.resize(image, (width // scale_factor, height // scale_factor))
    
    # Detect face locations with HOG on downscaled image
    hog_face_locations_small = face_recognition.face_locations(small_image)
    hog_face_locations = [(t * scale_factor, r * scale_factor, b * scale_factor, l * scale_factor) 
                           for (t, r, b, l) in hog_face_locations_small]

    # 2. Haar Cascade Detection on downscaled image (as backup for tilted/profile faces)
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        gray_small = cv2.cvtColor(small_image, cv2.COLOR_RGB2GRAY)
        haar_faces_rects = face_cascade.detectMultiScale(gray_small, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        haar_face_locations = []
        for (x, y, w, h) in haar_faces_rects:
            haar_face_locations.append((y * scale_factor, (x + w) * scale_factor, (y + h) * scale_factor, x * scale_factor))
            
        haar_profile_rects = profile_cascade.detectMultiScale(gray_small, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in haar_profile_rects:
            haar_face_locations.append((y * scale_factor, (x + w) * scale_factor, (y + h) * scale_factor, x * scale_factor))
    except Exception as e:
        print(f"Haar Cascade error: {e}")
        haar_face_locations = []

    # 3. Merge Detections using Intersection Over Union (IOU)
    final_face_locations = list(hog_face_locations)
    
    def calculate_iou(boxA, boxB):
        tA, rA, bA, lA = boxA
        tB, rB, bB, lB = boxB
        xA = max(lA, lB)
        yA = max(tA, tB)
        xB = min(rA, rB)
        yB = min(bA, bB)
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (rA - lA) * (bA - tA)
        boxBArea = (rB - lB) * (bB - tB)
        return interArea / float(boxAArea + boxBArea - interArea)

    for h_loc in haar_face_locations:
        is_duplicate = False
        for existing_loc in final_face_locations:
            if calculate_iou(h_loc, existing_loc) > 0.3:
                is_duplicate = True
                break
        if not is_duplicate:
            final_face_locations.append(h_loc)
            
    if len(final_face_locations) == 0:
        return []
        
    # Get high-quality encodings from original full-resolution image
    faces_encodings = face_recognition.face_encodings(image, known_face_locations=final_face_locations)
    
    if len(faces_encodings) == 0:
         return []

    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    predictions = []
    
    for i, (pred_label, loc, dist) in enumerate(zip(knn_clf.predict(faces_encodings), final_face_locations, closest_distances[0])):
        distance_val = round(dist[0], 2)
        is_match = distance_val <= threshold
        
        if is_match:
            if pred_label in names_dict:
                name = names_dict[pred_label]
            else:
                name = pred_label.replace('_', ' ')
        else:
            name = "Unknown"
            
        predictions.append({
            'name': name,
            'location': loc,
            'distance': distance_val
        })
        
    return predictions

def detect_and_crop_face(image_path, save_dir, filename_prefix="face"):
    """
    Detects faces in an image, crops the largest face, and saves it.
    
    Returns:
        True if a face was found and saved, False otherwise.
    """
    image = cv2.imread(image_path)
    if image is None:
        return False
        
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    
    if not face_locations:
        return False
        
    # Get largest face by area
    largest_face = max(face_locations, key=lambda f: (f[2] - f[0]) * (f[1] - f[3]))
    top, right, bottom, left = largest_face
    
    # Add padding around the cropped face
    height, width, _ = image.shape
    padding = 20
    top = max(0, top - padding)
    bottom = min(height, bottom + padding)
    left = max(0, left - padding)
    right = min(width, right + padding)
    
    face_image = image[top:bottom, left:right]
    
    os.makedirs(save_dir, exist_ok=True)
    existing_files = len(os.listdir(save_dir))
    save_path = os.path.join(save_dir, f"{filename_prefix}_{existing_files + 1:02d}.jpg")
    
    cv2.imwrite(save_path, face_image)
    return True

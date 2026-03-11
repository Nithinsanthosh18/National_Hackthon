import cv2
import os
import numpy as np

# Use an absolute path if necessary, but assuming relative works based on main execution context
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_DIR = os.path.join(BASE_DIR, "User")

# Pre-trained Haar Cascade classifier for frontal face detection
# OpenCV usually stores this internally, or we can use the cv2 data path
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def train_recognizer(base_dir=USER_DIR):
    """
    Scans the given base_dir for user folders, loads images, detects faces,
    and returns a trained LBPHFaceRecognizer and a label_map.
    """
    if not os.path.exists(base_dir):
        print(f"Warning: User directory '{base_dir}' not found.")
        return None, {}, {}

    faces = []
    labels = []
    label_map = {}
    thumbnail_map = {}
    current_label = 0

    print(f"Scanning target directory: {base_dir}")
    
    for user_name in os.listdir(base_dir):
        user_path = os.path.join(base_dir, user_name)
        if not os.path.isdir(user_path):
            continue
            
        label_map[current_label] = user_name
        print(f"Parsing faces for user: {user_name} (ID: {current_label})")
        
        # Keep track of the first valid image to use as a thumbnail
        ref_image_path = None
        
        for image_name in os.listdir(user_path):
            image_path = os.path.join(user_path, image_name)
            
            try:
                # Read image in grayscale explicitly
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                
                if ref_image_path is None:
                    ref_image_path = image_path
                    
                # Detect faces in the image - Use more lenient parameters for training
                detected_faces = face_cascade.detectMultiScale(
                    img, 
                    scaleFactor=1.05, 
                    minNeighbors=3, 
                    minSize=(20, 20)
                )
                
                # If a face is found, crop and add to our training data
                if len(detected_faces) > 0:
                    for (x, y, w, h) in detected_faces:
                        face_roi = img[y:y+h, x:x+w]
                        # Normalize lighting
                        face_roi = cv2.equalizeHist(face_roi)
                        faces.append(face_roi)
                        labels.append(current_label)
                        break 
                else:
                    # FALLBACK: If no face found via cascade, but image is smallish 
                    # (e.g. user uploaded a crop), just use the whole image
                    h, w = img.shape
                    if 50 < w < 500 and 50 < h < 500:
                        print(f"DEBUG: No face detected in {image_name}, using whole image as fallback")
                        face_roi = cv2.equalizeHist(img)
                        faces.append(face_roi)
                        labels.append(current_label)
                    else:
                        print(f"DEBUG: Skipping {image_name} - no face detected and doesn't look like a crop")
                    
            except Exception as e:
                print(f"Error parsing image {image_path}: {e}")
        
        if ref_image_path:
            thumbnail_map[current_label] = ref_image_path
            samples_count = labels.count(current_label)
            print(f"User {user_name} parsed. Found {samples_count} valid face samples.")
                
        current_label += 1

    if len(faces) > 0:
        print(f"Training Face Recognizer on {len(faces)} samples...")
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(labels))
        print("Training complete!")
        return recognizer, label_map, thumbnail_map
    else:
        print("No valid faces found to train on.")
        return None, {}, {}

if __name__ == "__main__":
    # Test execution
    r, m, t = train_recognizer()
    print(f"Label Map: {m}")
    print(f"Thumbnail Map: {t}")

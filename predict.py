import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

# Load model
model_path = os.path.join(os.path.dirname(__file__), "models", "age_gender_model.h5")

if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    exit(1)
model = load_model(model_path)

age_labels = ["Child","Teen","Adult","Middle Age","Senior"]
gender_labels = ["Male","Female"]

# Load test image
test_image_path = r"C:\Users\DHARSHINI\OneDrive\Desktop\Age_Gender_Prediction\test_images\test.jpg"

img = cv2.imread(test_image_path)
if img is None:
    print(f"Error: Could not load image from {test_image_path}")
    exit(1)

# Resize image if too large for better processing
height, width = img.shape[:2]
if height > 1280 or width > 1280:
    scale = min(1280/height, 1280/width)
    img = cv2.resize(img, (int(width*scale), int(height*scale)))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Use multiple cascade classifiers for better detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))

if len(faces) == 0:
    print("No faces detected in the image")
else:
    print(f"Detected {len(faces)} face(s)")
    for idx, (x,y,w,h) in enumerate(faces):
        # Add padding to face region for better context
        padding = int(0.1 * w)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + w + padding)
        y2 = min(img.shape[0], y + h + padding)
        
        face = img[y1:y2, x1:x2]
        face_resized = cv2.resize(face, (64, 64))
        
        # Normalize to float32
        face_normalized = face_resized.astype(np.float32) / 255.0
        face_input = np.expand_dims(face_normalized, axis=0)
        
        try:
            predictions = model.predict(face_input, verbose=0)
            
            # Handle both single and dual output models
            if isinstance(predictions, (list, tuple)) and len(predictions) == 2:
                age_pred, gender_pred = predictions
            else:
                age_pred, gender_pred = predictions, predictions
            
            age_idx = np.argmax(age_pred[0])
            gender_idx = np.argmax(gender_pred[0])
            age_conf = np.max(age_pred[0])
            gender_conf = np.max(gender_pred[0])
            
            age = age_labels[age_idx]
            gender = gender_labels[gender_idx]
            
            # Draw rectangle with thicker border
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
            # Display predictions with confidence scores
            label = f"{gender} ({gender_conf:.2f}), {age} ({age_conf:.2f})"
            cv2.putText(img, label, (x, y-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            print(f"Face {idx+1}: {gender} ({gender_conf:.2f}), {age} ({age_conf:.2f})")
            
        except Exception as e:
            print(f"Error during prediction for face {idx+1}: {e}")
            continue

cv2.imshow("Age & Gender Prediction", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Resize for display (increase output image size)
display_scale = 1.5  # set >1.0 to enlarge, <1.0 to shrink
display_w = int(img.shape[1] * display_scale)
display_h = int(img.shape[0] * display_scale)
display_img = cv2.resize(img, (display_w, display_h), interpolation=cv2.INTER_LINEAR)

cv2.namedWindow("Age & Gender Prediction", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Age & Gender Prediction", display_w, display_h)
cv2.imshow("Age & Gender Prediction", display_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Prediction complete!")

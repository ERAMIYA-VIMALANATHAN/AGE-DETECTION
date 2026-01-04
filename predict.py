import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

IMG_SIZE = 128


model_path = os.path.join(os.path.dirname(__file__), "models", "age_gender_model.h5")
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    exit(1)

model = load_model(model_path)

age_labels = ["Child","Teen","Adult","Middle Age","Senior"]
gender_labels = ["Male","Female"]

test_image_path = os.path.join(os.path.dirname(__file__), "test_images", "test.jpg")
if not os.path.exists(test_image_path):
    print(f"Error: Could not load image from {test_image_path}")
    exit(1)

img = cv2.imread(test_image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

if len(faces) == 0:
    print("No faces detected in the image")
else:
    for (x,y,w,h) in faces:
        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))/255.0
        face = np.expand_dims(face, axis=0)

        try:
            age_pred, gender_pred = model.predict(face, verbose=0)
            age = age_labels[np.argmax(age_pred)]
            gender = gender_labels[np.argmax(gender_pred)]

            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(img,f"{gender}, {age}",(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
        except Exception as e:
            print(f"Error during prediction: {e}")
            continue

cv2.imshow("Prediction", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Prediction complete!")

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

# Settings
IMG_SIZE = 64
dataset_path = "dataset/UTKFace"

# Load dataset
images, ages, genders = [], [], []
for file in os.listdir(dataset_path):
    try:
        age, gender = file.split("_")[:2]
        img_path = os.path.join(dataset_path, file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(img)
        ages.append(int(age))
        genders.append(int(gender))
    except:
        pass

images = np.array(images)/255.0
genders = to_categorical(genders, 2)

# Age groups
def age_group(age):
    if age <= 12: return 0
    elif age <= 19: return 1
    elif age <= 40: return 2
    elif age <= 60: return 3
    else: return 4

age_labels = np.array([age_group(a) for a in ages])
age_labels = to_categorical(age_labels, 5)

# Split dataset
X_train, X_test, y_age_train, y_age_test, y_gender_train, y_gender_test = train_test_split(
    images, age_labels, genders, test_size=0.2, random_state=42
)

# Build CNN
input_layer = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = Conv2D(32, (3,3), activation='relu')(input_layer)
x = MaxPooling2D()(x)
x = Conv2D(64, (3,3), activation='relu')(x)
x = MaxPooling2D()(x)
x = Flatten()(x)
age_output = Dense(5, activation='softmax', name='age')(x)
gender_output = Dense(2, activation='softmax', name='gender')(x)
model = Model(inputs=input_layer, outputs=[age_output, gender_output])

model.compile(
    optimizer='adam',
    loss={'age':'categorical_crossentropy','gender':'categorical_crossentropy'},
    metrics={'age':'accuracy', 'gender':'accuracy'}
)

# Train model
model.fit(
    X_train,
    {'age': y_age_train, 'gender': y_gender_train},
    validation_data=(X_test, {'age': y_age_test, 'gender': y_gender_test}),
    epochs=15,
    batch_size=32
)

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/age_gender_model.h5")
print("Model trained and saved successfully!")

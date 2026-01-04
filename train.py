import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

IMG_SIZE = 128
DATASET_PATH = "dataset/UTKFace"  # Put your UTKFace dataset here
EPOCHS = 50
BATCH_SIZE = 32


images, ages, genders = [], [], []

for file in os.listdir(DATASET_PATH):
    try:
        age, gender = file.split("_")[:2]
        img_path = os.path.join(DATASET_PATH, file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(img)
        ages.append(int(age))
        genders.append(int(gender))
    except:
        continue

images = np.array(images, dtype='float32') / 255.0
genders = to_categorical(genders, 2)


def age_group(age):
    if age <= 12: return 0
    elif age <= 19: return 1
    elif age <= 40: return 2
    elif age <= 60: return 3
    else: return 4

age_labels = np.array([age_group(a) for a in ages])
age_labels = to_categorical(age_labels, 5)


X_train, X_test, y_age_train, y_age_test, y_gender_train, y_gender_test = train_test_split(
    images, age_labels, genders, test_size=0.2, random_state=42
)


def augment(image, labels):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    return image, labels

train_dataset = tf.data.Dataset.from_tensor_slices(
    (X_train, {'age': y_age_train, 'gender': y_gender_train})
).shuffle(buffer_size=20000).map(augment).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices(
    (X_test, {'age': y_age_test, 'gender': y_gender_test})
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


input_layer = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = Conv2D(32, (3,3), activation='relu', padding='same')(input_layer)
x = MaxPooling2D()(x)

x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D()(x)

x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D()(x)

x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)

age_output = Dense(5, activation='softmax', name='age')(x)
gender_output = Dense(2, activation='softmax', name='gender')(x)

model = Model(inputs=input_layer, outputs=[age_output, gender_output])

model.compile(
    optimizer='adam',
    loss={'age':'categorical_crossentropy', 'gender':'categorical_crossentropy'},
    metrics={'age':'accuracy', 'gender':'accuracy'}
)

model.summary()

es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[es]
)
    

os.makedirs("models", exist_ok=True)
model.save("models/age_gender_model.h5")
print("Model trained and saved successfully!")


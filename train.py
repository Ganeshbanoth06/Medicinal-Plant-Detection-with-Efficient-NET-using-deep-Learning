import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import json

IMG_SIZE = 224
BATCH_SIZE = 32

train_dir = "dataset/train"
val_dir = "dataset/val"

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(train_dir,
                                         target_size=(IMG_SIZE, IMG_SIZE),
                                         batch_size=BATCH_SIZE,
                                         class_mode='categorical')

val_data = datagen.flow_from_directory(val_dir,
                                       target_size=(IMG_SIZE, IMG_SIZE),
                                       batch_size=BATCH_SIZE,
                                       class_mode='categorical')

# Save labels
labels = train_data.class_indices
with open("labels.json", "w") as f:
    json.dump(labels, f)

# Model
base_model = EfficientNetB0(weights='imagenet',
                            include_top=False,
                            input_shape=(224,224,3))

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(labels), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data,
          validation_data=val_data,
          epochs=20)

model.save("model/model.h5")
print("✅ Model trained and saved!")

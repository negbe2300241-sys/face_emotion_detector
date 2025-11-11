import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# -------------------------------
# 1. Data Preparation
# -------------------------------

# Make sure your dataset has this structure:
# dataset/
# ├── train/
# │   ├── angry/
# │   ├── happy/
# │   ├── sad/
# │   └── ... (7 emotion folders total)
# ├── val/
# └── test/

data_dir = 'dataset'
img_size = (48, 48)
batch_size = 32

datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    os.path.join(data_dir, 'train'),
    target_size=img_size,
    color_mode="grayscale",
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True
)

val_data = datagen.flow_from_directory(
    os.path.join(data_dir, 'val'),
    target_size=img_size,
    color_mode="grayscale",
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=False
)

test_data = datagen.flow_from_directory(
    os.path.join(data_dir, 'test'),
    target_size=img_size,
    color_mode="grayscale",
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=False
)

# -------------------------------
# 2. Build CNN Model
# -------------------------------

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 emotion categories
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------------------
# 3. Callbacks (optional)
# -------------------------------

checkpoint = ModelCheckpoint(
    'best_emotion_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# -------------------------------
# 4. Train Model
# -------------------------------

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=40,
    callbacks=[checkpoint, early_stop],
    verbose=1
)

# -------------------------------
# 5. Save Final Model
# -------------------------------

model.save('emotion_model_final.h5')
print("✅ Model training complete and saved as 'emotion_model_final.h5'")

# -------------------------------
# 6. Evaluate on Test Set
# -------------------------------

test_loss, test_acc = model.evaluate(test_data)
print(f"🧪 Test Accuracy: {test_acc:.4f}")
print(f"🧪 Test Loss: {test_loss:.4f}")

# -------------------------------
# 7. Plot Training History
# -------------------------------

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

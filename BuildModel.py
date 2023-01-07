import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import skimage.io as io
import skimage.transform as trans
import random

def build_model(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Activation('sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def load_and_preprocess_image(image_path):
    # Load image and convert to RGB
    image = io.imread(image_path)  # Load image
    image = trans.resize(image, (64, 64))  # Resize image to (128, 128)
    image = image / 255.0  # Normalize image to [0, 1] range

    #Use data augmentation
    num_rotations = random.randint(0, 3)  # Choose a random number of rotations (0, 1, 2, or 3)
    image = tf.image.rot90(image, k=num_rotations)  # Apply rotations
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)

    return image


def load_and_preprocess_data(data_dir):
    # Load all image paths
    image_paths = []
    for label in ['cats', 'dogs']:
        label_dir = os.path.join(data_dir, label)
        for image_filename in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_filename)
            image_paths.append(image_path)

    # Load and preprocess images
    images = [load_and_preprocess_image(p) for p in image_paths]
    images = np.array(images)  # Convert list to numpy array

    # Create labels
    labels = [1 if 'cat' in p else 0 for p in image_paths]
    labels = np.array(labels)  # Convert list to numpy array

    return images, labels

def main():
    # Load and preprocess data
    data_dir = '.'
    training_data_dir = os.path.join(data_dir, 'training_data')
    testing_data_dir = os.path.join(data_dir, 'testing_data')
    X_train, y_train = load_and_preprocess_data(training_data_dir)
    X_test, y_test = load_and_preprocess_data(testing_data_dir)

    # Build model
    input_shape = (64, 64, 3)
    model = build_model(input_shape)

    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=500,
        batch_size= 250,
        validation_data=(X_test, y_test),
        verbose=2,
    )

    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f'Test loss: {test_loss:.4f}')
    print(f'Test accuracy: {test_acc:.4f}')

    # Plot training and validation loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.show()


if __name__ == "__main__":
    main()

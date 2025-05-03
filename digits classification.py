!pip install pillow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from google.colab import files

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy (CNN): {test_acc * 100:.2f}%")

model.save('mnist_cnn_model.h5')

def predict_random_samples():
    idx = np.random.randint(0, len(X_test), 5)
    samples = X_test[idx]
    labels = np.argmax(y_test[idx], axis=1)

    predictions = np.argmax(model.predict(samples), axis=1)

    for i in range(5):
        plt.imshow(samples[i].reshape(28,28), cmap='gray')
        plt.title(f"True: {labels[i]}, Predicted: {predictions[i]}")
        plt.axis('off')
        plt.show()

def predict_uploaded_image():
    uploaded = files.upload()

    if uploaded:
        filename = list(uploaded.keys())[0]

        img = Image.open(filename).convert('L') 
        img = ImageOps.invert(img) 
        img = img.resize((28, 28))
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)
        prediction = np.argmax(model.predict(img_array), axis=1)

        plt.imshow(img_array.reshape(28, 28), cmap='gray')
        plt.title(f"Predicted Number: {prediction[0]}")
        plt.axis('off')
        plt.show()

print("\nPredicting random samples from test set...")
predict_random_samples()

print("\nNow upload an image for prediction...")
predict_uploaded_image()

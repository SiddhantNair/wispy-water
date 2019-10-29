import tensorflow as tf
from tensorflow.keras import models, layers
import cv2
import numpy as np
import json

yellow_images = []
homer_details = []
test_images = []
test_details = []

def load_data():

    global yellow_images, homer_details, test_images, test_details

    with open('assets/simpsons-faces/numbered/homer_details.json') as json_file:
        data = json.load(json_file)
        homer_details = data['list']
        test_details = data['train']

    homer_details = np.asarray(homer_details)
    test_details  = np.asarray(test_details)

    for i in range(1, 101):
        img = cv2.imread(f'assets/simpsons-faces/cropped/{i}.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        yellow_images.append(img)
    
    for i in range(131, 151):
        img = cv2.imread(f'assets/simpsons-faces/cropped/{i}.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        test_images.append(img)
    
    yellow_images = np.asarray(yellow_images)
    test_images = np.asarray(test_images)

def train_nn_model():

    global yellow_images, homer_details, test_images, test_details
    load_data()

    print('Training Neural Network')
    model = models.Sequential([
        layers.Conv2D(16, 3, padding='same', activation='relu', 
            input_shape=(200, 200 ,3)),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(yellow_images, homer_details, epochs=10, validation_data=(test_images, test_details))
    model.save("model.h5")

def evaluate_stuff():

    model = models.load_model('model.h5')
    results = []

    for i in range(261, 281):
        img = cv2.imread(f'assets/simpsons-faces/cropped/{i}.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        img = np.reshape(img, [1, 200, 200, 3])
        result = model.predict(img)
        results.append(round(result.item(0), 3))
    
    print(results)
    
    for i in range(len(results)):
        if results[i] >= 0.5:
            print(f'{i+261}. Homer')
from keras.layers import Input, Conv2D, Lambda, Dense, Flatten, Dropout
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
import numpy as np
import cv2

def load_image(image_path):
    # Load image and preprocess it
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (200, 100))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    return img

def create_siamese_network(input_shape):
    # Define inputs
    input_1 = Input(input_shape)
    input_2 = Input(input_shape)
    
    # Define shared convolutional layers
    conv_layers = [
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        Dropout(0.2),
        Conv2D(64, (3, 3), activation='relu'),
        Dropout(0.2),
        Conv2D(128, (3, 3), activation='relu'),
        Flatten(),
        Dense(512, activation='sigmoid')
    ]
    
    # Define function to share convolutional layers between inputs
    def shared_layers(input_layer):
        output = input_layer
        for layer in conv_layers:
            output = layer(output)
        return output
    
    # Pass both inputs through shared convolutional layers
    output_1 = shared_layers(input_1)
    output_2 = shared_layers(input_2)
    
    # Define distance metric for comparison
    distance = Lambda(lambda x: abs(x[0] - x[1]))
    distance_output = distance([output_1, output_2])
    
    # Define output layer
    output_layer = Dense(1, activation='sigmoid')(distance_output)
    
    # Define model
    model = Model(inputs=[input_1, input_2], outputs=output_layer)
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=0.0001))
    model.save("sign.h5")

def verify_signature(signature_path_1, signature_path_2, model_path):
    # Load signatures and model
    signature_1 = load_image(signature_path_1)
    signature_2 = load_image(signature_path_2)
    model = load_model(model_path)
    
    # Predict similarity
    similarity = model.predict([signature_1, signature_2])[0][0]
    
    # Define threshold for similarity
    threshold = 0.5
    
    # Compare similarity to threshold
    if similarity > threshold:
        return True
    else:
        return False
    

# verify_signature()
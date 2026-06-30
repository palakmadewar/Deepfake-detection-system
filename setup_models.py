import os
import sys

def build_mock_models():
    print("Checking for TensorFlow to generate mock models...")
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
        
        # 1. Create mock Image model
        image_path = 'image_cnn.h5'
        if not os.path.exists(image_path):
            print("Generating mock image_cnn.h5...")
            model = Sequential([
                Input(shape=(224, 224, 3)),
                Conv2D(8, (3, 3), activation='relu'),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dense(8, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.save(image_path)
            print(f"Mock image model saved to {os.path.abspath(image_path)}")
        else:
            print("image_cnn.h5 already exists.")

        # 2. Create mock Audio model
        audio_path = 'audio_model.h5'
        if not os.path.exists(audio_path):
            print("Generating mock audio_model.h5...")
            model = Sequential([
                Input(shape=(40,)),
                Dense(16, activation='relu'),
                Dense(8, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.save(audio_path)
            print(f"Mock audio model saved to {os.path.abspath(audio_path)}")
        else:
            print("audio_model.h5 already exists.")
            
    except ImportError:
        print("\n[WARNING] TensorFlow is not installed in the current environment.")
        print("Truth Shield will run in 'Simulation Mode' using high-fidelity numerical heuristics.")
        print("To run with actual models, install TensorFlow using: pip install tensorflow")
        print("And place your 'image_cnn.h5' and 'audio_model.h5' in this directory.\n")

if __name__ == '__main__':
    build_mock_models()

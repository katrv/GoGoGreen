import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('garbage_classifier_model.keras')
import cv2
import numpy as np

class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Start video capture from the camera (0 for default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Resize the frame to match the model's input size (adjust based on your model's input size)
    resized_frame = cv2.resize(frame, (224, 224))  # Example size, adjust accordingly

    # Preprocess the frame (normalize, expand dimensions for batch input)
    input_frame = np.expand_dims(resized_frame, axis=0)  # Add batch dimension
    input_frame = input_frame / 255.0  # Normalize if the model was trained with normalized data

    # Classify the frame
    predictions = model.predict(input_frame)

    # Get the predicted class (adjust based on your model's output layer)
    predicted_index = np.argmax(predictions, axis=-1)[0]
    predicted_label = class_names[predicted_index]

    # Display the predicted class on the frame
    cv2.putText(frame, f"Predicted Class: {predicted_label}", (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame with the predicted class
    cv2.imshow('Live Garbage Classification', frame)

    # Exit loop when the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()





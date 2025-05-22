import os
import serial
import json
import threading
import time
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
from pathlib import Path

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict allowed origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directory for captured images
os.makedirs("captured_images", exist_ok=True)

# Mount the captured_images directory to serve static files
app.mount("/captured_images", StaticFiles(directory="captured_images"), name="captured_images")

# Store the latest sensor readings
sensor_data = {
    "soil_moisture": 0,
    "temperature": None,
    "humidity": None,
    "timestamp": None
}

# Flag to control verbose output
VERBOSE_OUTPUT = False  # Set to False to reduce terminal output

def read_serial_data():
    """Read sensor data from Arduino via serial port"""
    try:
        # Adjust port and baudrate to match your Arduino
        try:
            ser = serial.Serial('/dev/ttyUSB0', 9600)  # Linux/Raspberry Pi
            print("Connected to Arduino on Linux port", ser.name)
        except:
            try:
                ser = serial.Serial('COM3', 9600)  # Windows
                print("Connected to Arduino on Windows port", ser.name)
            except:
                print("Failed to connect to Arduino on common ports. Retrying...")
                time.sleep(10)
                read_serial_data()  # Recursive retry
                return
        
        while True:
            try:
                # Read line from serial port
                line = ser.readline().decode('utf-8').strip()
                if VERBOSE_OUTPUT:
                    print(f"Received data: {line}")
                
                # Try to parse JSON data
                try:
                    data = json.loads(line)
                    # Update sensor data dictionary with all available values
                    if "soil_moisture" in data:
                        sensor_data["soil_moisture"] = data["soil_moisture"]
                    if "temperature" in data:
                        sensor_data["temperature"] = data["temperature"]
                    if "humidity" in data:
                        sensor_data["humidity"] = data["humidity"]
                    sensor_data["timestamp"] = time.time()
                    
                    # Print update notification only occasionally (every 10 seconds)
                    if VERBOSE_OUTPUT or int(time.time()) % 10 == 0:
                        print(f"Sensor data updated at {time.strftime('%H:%M:%S')}")
                        
                except json.JSONDecodeError as e:
                    if VERBOSE_OUTPUT:
                        print(f"JSON parsing error: {e} - Raw data: {line}")
                    
            except Exception as e:
                print(f"Error reading serial data: {str(e)}")
                
    except Exception as e:
        print(f"Failed to connect to serial port: {str(e)}")
        time.sleep(10)  # Wait before trying to reconnect

# Start the serial reading thread
serial_thread = threading.Thread(target=read_serial_data, daemon=True)
serial_thread.start()

@app.get("/capture-image")
async def capture_image():
    """Capture an image from the connected webcam"""
    try:
        # Try different camera indices if the webcam isn't detected
        for camera_index in [0, 1, 2]:
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                print(f"Connected to webcam at index {camera_index}")
                break
        
        if not cap.isOpened():
            raise HTTPException(
                status_code=500,
                detail="Could not connect to webcam. Check connection and try again."
            )
        
        # Allow camera to adjust to lighting
        for _ in range(5):
            ret = cap.grab()  # Grab a few frames to adjust
        
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            raise HTTPException(
                status_code=500,
                detail="Failed to capture image from webcam."
            )
        
        # Save the captured image
        timestamp = int(time.time())
        image_filename = f"capture_{timestamp}.jpg"
        image_path = f"captured_images/{image_filename}"
        cv2.imwrite(image_path, frame)
        
        # Release the webcam
        cap.release()
        
        return {
            "status": "success",
            "message": "Image captured successfully",
            "image_path": f"/captured_images/{image_filename}"
        }
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        print(f"Error capturing image: {str(e)}")
        print(f"Traceback: {traceback_str}")
        return {
            "status": "error",
            "message": f"Error capturing image: {str(e)}",
            "traceback": traceback_str
        }

@app.get("/sensor-data")
async def get_sensor_data():
    """Return the latest sensor readings"""
    if sensor_data["timestamp"] is None:
        return {
            "status": "waiting",
            "message": "No sensor data received yet"
        }
    
    # Calculate soil moisture percentage (assuming higher value means drier soil)
    soil_moisture_pct = 100 - ((sensor_data["soil_moisture"] / 1023) * 100)
    
    # Interpret the values
    moisture_status = "Wet" if soil_moisture_pct > 70 else "Moderate" if soil_moisture_pct > 30 else "Dry"
    
    response_data = {
        "status": "success",
        "data": {
            "soil_moisture": {
                "raw": sensor_data["soil_moisture"],
                "percentage": soil_moisture_pct,
                "status": moisture_status
            },
            "timestamp": sensor_data["timestamp"],
            "last_update": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(sensor_data["timestamp"]))
        }
    }
    
    # Add temperature and humidity data if available
    if sensor_data["temperature"] is not None:
        response_data["data"]["temperature"] = sensor_data["temperature"]
    
    if sensor_data["humidity"] is not None:
        response_data["data"]["humidity"] = sensor_data["humidity"]
    
    return response_data

def create_model(input_shape=(256, 256, 3), n_classes=4):
    """Create the model with the same architecture as training"""
    inputs = layers.Input(shape=input_shape)
    
    # Preprocessing
    x = layers.Rescaling(1.0 / 255)(inputs)
    
    # Convolutional blocks
    conv_blocks = [
        (32, (3, 3)),
        (64, (3, 3)),
        (128, (3, 3)),
        (128, (3, 3)),
        (128, (3, 3))
    ]
    
    for filters, kernel_size in conv_blocks:
        x = layers.Conv2D(filters, kernel_size, activation='relu')(x)
        x = layers.MaxPooling2D(2, 2)(x)
    
    # Dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    if VERBOSE_OUTPUT:
        print("Model architecture created:")
        model.summary()
    else:
        print("Model architecture created successfully")
    
    return model

def load_model_with_fallback():
    """Try multiple approaches to load the model"""
    base_dir = Path(__file__).parent
    
    # Try loading full model first
    full_model_path = base_dir / 'model_full.h5'
    weights_path = base_dir / 'model.weights.h5'
    
    print("Attempting to load model...")
    
    try:
        if full_model_path.exists():
            print("Found full model file, loading...")
            model = tf.keras.models.load_model(str(full_model_path))
            print("Full model loaded successfully!")
            return model
    except Exception as e:
        print(f"Error loading full model: {e}")
    
    try:
        print("Attempting to load weights into fresh model...")
        model = create_model()
        if weights_path.exists():
            model.load_weights(str(weights_path))
            print("Weights loaded successfully!")
            return model
    except Exception as e:
        print(f"Error loading weights: {e}")
    
    print("All loading attempts failed. Please ensure model files are present.")
    return None

# Initialize the model
print("Initializing model...")
model = load_model_with_fallback()

def preprocess_image(image_data):
    """Preprocess image for prediction"""
    img = tf.image.decode_image(image_data, channels=3)
    img = tf.image.resize(img, [256, 256])
    # No need to normalize here as the model has a Rescaling layer
    return tf.expand_dims(img, 0)

def is_valid_image(image_data, min_size_kb=10):
    """
    Check if image data is valid and meets minimum size requirements
    """
    # Check minimum file size (to avoid empty or corrupted files)
    if len(image_data) < min_size_kb * 1024:
        return False
    
    try:
        # Try to decode the image to verify it's a valid image file
        _ = tf.image.decode_image(image_data)
        return True
    except Exception:
        return False

@app.get("/")
async def root():
    model_status = "loaded" if model is not None else "not loaded"
    return {
        "status": "online",
        "model_status": model_status,
        "sensor_status": "connected" if sensor_data["timestamp"] is not None else "disconnected",
        "sensor_thread_active": serial_thread.is_alive(),
        "tensorflow_version": tf.__version__,
        "webcam_support": "enabled"
    }

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    disease_mapping = {
        0: {"Name": "Blight", "Solution": "Use resistant hybrids and practice crop rotation."},
        1: {"Name": "Common_Rust", "Solution": "Use resistant hybrids and apply fungicides if necessary."},
        2: {"Name": "Gray_Leaf_Spot", "Solution": "Apply fungicides and maintain proper field drainage."},
        3: {"Name": "Healthy", "Solution": "No action required. Your plant is healthy!"}
    }
    
    # Check if model is loaded
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Please check model files and server logs."
        )
    
    # Validate file type
    valid_extensions = ['.jpg', '.jpeg', '.png']
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in valid_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Please upload an image with one of these extensions: {', '.join(valid_extensions)}"
        )
    
    try:
        # Read image data
        image_data = await file.read()
        
        # Validate image data
        if not is_valid_image(image_data):
            raise HTTPException(
                status_code=400,
                detail="Invalid or corrupted image. Please upload a valid image file."
            )
        
        # Process the image
        processed_image = preprocess_image(image_data)
        
        # Get prediction
        prediction = model.predict(processed_image, verbose=0)  # Reduce verbosity
        predicted_class = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][predicted_class])
        
        # Set confidence threshold - LOWERED to detect more instances
        confidence_threshold = 0.5
        
        # Get raw prediction values for debugging
        raw_predictions = prediction[0].tolist()
        class_confidences = {
            disease_mapping[i]["Name"]: float(prediction[0][i]) 
            for i in range(len(prediction[0]))
        }
        
        # Print minimal debug information
        if VERBOSE_OUTPUT:
            print(f"Predicted class: {predicted_class} ({disease_mapping[predicted_class]['Name']}), Confidence: {confidence}")
        
        # Include current sensor data in response
        current_sensor_data = None
        if sensor_data["timestamp"] is not None:
            soil_moisture_pct = 100 - ((sensor_data["soil_moisture"] / 1023) * 100)
            current_sensor_data = {
                "soil_moisture": {
                    "raw": sensor_data["soil_moisture"],
                    "percentage": soil_moisture_pct
                },
                "temperature": sensor_data["temperature"],
                "humidity": sensor_data["humidity"],
                "timestamp": sensor_data["timestamp"]
            }
        
        # Check if prediction meets confidence threshold
        if confidence < confidence_threshold:
            return {
                "status": "low_confidence",
                "message": "This doesn't appear to be a maize leaf with disease or the image quality is poor.",
                "disease": None,
                "confidence": None,
                "raw_predictions": raw_predictions,
                "class_confidences": class_confidences,
                "solution": None,
                "recommendation": "Please upload a clear image of a maize leaf. This model is specifically trained to identify maize leaf diseases.",
                "sensor_data": current_sensor_data
            }
        
        disease_details = disease_mapping.get(predicted_class, {"Name": "Unknown", "Solution": "No solution available"})
        
        # Check if disease name is null or empty
        if not disease_details["Name"] or disease_details["Name"] == "Unknown":
            return {
                "status": "unknown_disease",
                "message": "Unable to identify a specific disease.",
                "disease": None,
                "confidence": None,
                "raw_predictions": raw_predictions,
                "class_confidences": class_confidences,
                "solution": None,
                "recommendation": "Please try with a clearer image of the maize leaf.",
                "sensor_data": current_sensor_data
            }
        
        # Add soil moisture context to the solution if sensor data is available
        enhanced_solution = disease_details["Solution"]
        if current_sensor_data:
            # Add temperature and humidity context if available
            if current_sensor_data.get("temperature") is not None and current_sensor_data.get("humidity") is not None:
                temp = current_sensor_data["temperature"]
                humidity = current_sensor_data["humidity"]
                
                # Add temperature and humidity insights based on disease
                if predicted_class == 0:  # Blight
                    if temp > 27 and humidity > 75:
                        enhanced_solution += " High temperature and humidity are creating favorable conditions for blight. Consider improving ventilation."
                elif predicted_class == 1:  # Common Rust
                    if humidity > 80:
                        enhanced_solution += " The high humidity levels are favorable for rust development. Improve air circulation if possible."
                elif predicted_class == 2:  # Gray Leaf Spot
                    if temp > 25 and humidity > 85:
                        enhanced_solution += " Current temperature and humidity conditions are ideal for Gray Leaf Spot development. Monitor closely."
                
            # Add soil moisture recommendations
            if "soil_moisture" in current_sensor_data:
                moisture_pct = current_sensor_data["soil_moisture"]["percentage"]
                if moisture_pct < 30 and predicted_class != 3:  # If soil is dry and plant not healthy
                    enhanced_solution += " Consider increasing irrigation as soil moisture is low."
                elif moisture_pct > 80 and (predicted_class == 0 or predicted_class == 2):  # If soil is very wet and disease is Blight or Gray Leaf Spot
                    enhanced_solution += " Consider improving drainage as excess moisture may worsen the condition."
        
        return {
            "status": "success",
            "class": predicted_class,
            "confidence": confidence,
            "disease": disease_details["Name"],
            "solution": enhanced_solution,
            "raw_predictions": raw_predictions,
            "class_confidences": class_confidences,
            "sensor_data": current_sensor_data
        }
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        print(f"Error processing image: {str(e)}")
        print(f"Traceback: {traceback_str}")
        return {
            "status": "error",
            "message": f"Error processing image: {str(e)}",
            "traceback": traceback_str,
            "disease": None,
            "confidence": None,
            "solution": None,
            "sensor_data": sensor_data if sensor_data["timestamp"] is not None else None
        }

@app.get("/sensor-history")
async def get_sensor_history():
    """
    For a real implementation, you would store sensor readings in a database
    This is just a placeholder that returns the current reading and mock historical data
    """
    if sensor_data["timestamp"] is None:
        return {
            "status": "waiting",
            "message": "No sensor data received yet"
        }
    
    # In a real implementation, you would query your database here
    # This is just a placeholder
    current_time = time.time()
    
    # Create some fake historical data based on the current reading
    history = []
    for i in range(10):
        # Create slightly varying readings for demo purposes
        soil_reading = sensor_data["soil_moisture"] + (np.random.randint(-50, 50) if sensor_data["soil_moisture"] > 50 else 0)
        soil_reading = max(0, min(1023, soil_reading))  # Keep within valid range
        
        # Add temperature and humidity history if available
        temp_reading = None
        humidity_reading = None
        
        if sensor_data["temperature"] is not None:
            temp_reading = sensor_data["temperature"] + np.random.randint(-2, 3)
            temp_reading = max(15, min(40, temp_reading))  # Keep within reasonable range
            
        if sensor_data["humidity"] is not None:
            humidity_reading = sensor_data["humidity"] + np.random.randint(-5, 6)
            humidity_reading = max(30, min(100, humidity_reading))  # Keep within valid range
        
        entry = {
            "timestamp": current_time - (i * 3600),  # One hour intervals
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time - (i * 3600))),
            "soil_moisture": soil_reading,
            "soil_moisture_percentage": 100 - ((soil_reading / 1023) * 100)
        }
        
        if temp_reading is not None:
            entry["temperature"] = temp_reading
            
        if humidity_reading is not None:
            entry["humidity"] = humidity_reading
            
        history.append(entry)
    
    # Prepare current data response
    current_data = {
        "soil_moisture": sensor_data["soil_moisture"],
        "soil_moisture_percentage": 100 - ((sensor_data["soil_moisture"] / 1023) * 100),
        "timestamp": sensor_data["timestamp"],
        "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(sensor_data["timestamp"]))
    }
    
    # Add temperature and humidity if available
    if sensor_data["temperature"] is not None:
        current_data["temperature"] = sensor_data["temperature"]
        
    if sensor_data["humidity"] is not None:
        current_data["humidity"] = sensor_data["humidity"]
    
    return {
        "status": "success",
        "current": current_data,
        "history": history
    }

# Keep these validation and testing endpoints
@app.post("/validate-image/")
async def validate_image(file: UploadFile = File(...)):
    """
    Endpoint to check if an image appears to be a maize leaf
    """
    try:
        # Read image data
        image_data = await file.read()
        
        # Validate image data
        if not is_valid_image(image_data):
            return {
                "valid": False, 
                "reason": "Invalid or corrupted image file",
                "confidence": None
            }
        
        # Process the image
        processed_image = preprocess_image(image_data)
        
        # Get prediction (using existing model as approximation)
        prediction = model.predict(processed_image, verbose=0)  # Reduce verbosity
        max_confidence = float(np.max(prediction[0]))
        predicted_class = int(np.argmax(prediction[0]))
        
        # Include raw predictions for debugging
        raw_predictions = prediction[0].tolist()
        
        # Get the disease mapping
        disease_mapping = {
            0: "Blight",
            1: "Common_Rust",
            2: "Gray_Leaf_Spot",
            3: "Healthy"
        }
        
        # If the model is very confident in any class, it's likely a maize leaf
        # Lowered threshold for better sensitivity
        if max_confidence > 0.5:
            return {
                "valid": True, 
                "confidence": max_confidence,
                "predicted_class": predicted_class,
                "disease": disease_mapping[predicted_class] if predicted_class in disease_mapping else "Unknown",
                "raw_predictions": raw_predictions,
                "sensor_data": sensor_data if sensor_data["timestamp"] is not None else None
            }
        else:
            return {
                "valid": False, 
                "confidence": None,
                "raw_predictions": raw_predictions,
                "reason": "The image doesn't appear to be a maize leaf with disease",
                "sensor_data": sensor_data if sensor_data["timestamp"] is not None else None
            }
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        print(f"Error validating image: {str(e)}")
        print(f"Traceback: {traceback_str}")
        return {
            "valid": False, 
            "reason": str(e),
            "traceback": traceback_str,
            "confidence": None
        }

@app.post("/test-all-classes/")
async def test_all_classes(file: UploadFile = File(...)):
    """
    Testing endpoint that returns predictions for all classes
    Useful for debugging model behavior
    """
    disease_mapping = {
        0: "Blight",
        1: "Common_Rust",
        2: "Gray_Leaf_Spot",
        3: "Healthy"
    }
    
    try:
        # Read image data
        image_data = await file.read()
        
        # Validate image data
        if not is_valid_image(image_data):
            raise HTTPException(
                status_code=400,
                detail="Invalid or corrupted image. Please upload a valid image file."
            )
        
        # Process the image
        processed_image = preprocess_image(image_data)
        
        # Get prediction
        prediction = model.predict(processed_image, verbose=0)  # Reduce verbosity
        predicted_class = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][predicted_class])
        
        # Prepare response with detailed class information
        class_details = []
        for i in range(len(prediction[0])):
            class_details.append({
                "class_id": i,
                "disease": disease_mapping.get(i, "Unknown"),
                "confidence": float(prediction[0][i]),
                "is_predicted": (i == predicted_class)
            })
        
        # Sort by confidence (highest first)
        class_details.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Create sensor data response with all available sensors
        current_sensor_data = None
        if sensor_data["timestamp"] is not None:
            current_sensor_data = {
                "soil_moisture": sensor_data["soil_moisture"],
                "soil_moisture_percentage": 100 - ((sensor_data["soil_moisture"] / 1023) * 100),
                "timestamp": sensor_data["timestamp"]
            }
            
            if sensor_data["temperature"] is not None:
                current_sensor_data["temperature"] = sensor_data["temperature"]
                
            if sensor_data["humidity"] is not None:
                current_sensor_data["humidity"] = sensor_data["humidity"]
        
        return {
            "status": "success",
            "predicted_class": predicted_class,
            "predicted_disease": disease_mapping.get(predicted_class, "Unknown"),
            "primary_confidence": confidence,
            "all_classes": class_details,
            "raw_prediction": prediction[0].tolist(),
            "sensor_data": current_sensor_data
        }
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        print(f"Error testing image: {str(e)}")
        print(f"Traceback: {traceback_str}")
        return {
            "status": "error",
            "message": f"Error testing image: {str(e)}",
            "traceback": traceback_str
        }

if __name__ == "__main__":
    import uvicorn
    # Make sure to install required packages:
    # pip install fastapi uvicorn python-multipart tensorflow numpy pyserial opencv-python
    uvicorn.run(app, host="localhost", port=8000)
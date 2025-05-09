import logging
import traceback
from pyzbar.pyzbar import decode
import cv2
import numpy as np
from ultralytics import YOLO
import streamlit as st # Import Streamlit
from PIL import Image # Used by Streamlit for image uploads
import os

# --- Setup basic logging ---
# This configures logging for your application.
# Level=INFO means it will log messages at INFO level and above (INFO, WARNING, ERROR, CRITICAL).
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__) # Gets a logger for this specific module

# --- Load YOLO model ---
MODEL_PATH = 'best.pt' # Specifies the path to your YOLO model file
model = None
# Check if the model file exists before attempting to load it
if os.path.exists(MODEL_PATH):
    try:
        model = YOLO(MODEL_PATH) # Loads the YOLO model
        logger.info(f"YOLO model '{MODEL_PATH}' loaded successfully.")
    except Exception as e:
        # Logs a fatal error if the model cannot be loaded
        logger.error(f"Fatal Error: Could not load YOLO model from '{MODEL_PATH}'. Exception: {e}\n{traceback.format_exc()}")
        model = None # Set model to None to indicate failure
else:
    # Logs a fatal error if the model file is not found
    logger.error(f"Fatal Error: YOLO model file '{MODEL_PATH}' not found. Please ensure it's in the same directory as app.py.")


def decode_barcode_from_pyzbar(cropped_image):
    """
    Decodes the barcode or QR code from the cropped image using pyzbar.
    Applies image processing techniques to enhance decoding success.
    """
    # Check if the input image is empty (e.g., due to invalid cropping)
    if cropped_image.size == 0:
        logger.warning("Attempted to decode an empty cropped image.")
        return "No barcode detected (empty image)."

    try:
        # Convert the image to grayscale for efficiency and better barcode detection
        gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement.
        # This is particularly useful for images with varying lighting conditions.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(gray_image)

        # Apply adaptive thresholding to convert to a binary image.
        # THRESH_BINARY + THRESH_OTSU automatically determines the optimal threshold value.
        _, thresholded_image = cv2.threshold(enhanced_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Attempt to decode barcodes/QR codes using pyzbar on the thresholded image first.
        barcodes = decode(thresholded_image)

        if not barcodes:
            # If decoding from thresholded image fails, try the contrast-enhanced image.
            barcodes = decode(enhanced_image)

        if not barcodes:
            # If still no barcodes, try decoding from the original grayscale image.
            barcodes = decode(gray_image)

        if barcodes:
            # If barcodes are found, decode the data (assuming UTF-8 encoding) from the first detected barcode.
            barcode_data = barcodes[0].data.decode("utf-8")
            logger.info(f"Successfully decoded: {barcode_data}")
            return barcode_data
        
        # If no barcodes are detected after all attempts, log and return a specific message.
        logger.info("No barcode detected after image processing attempts.")
        return "No barcode detected"
    except Exception as e:
        # Log any errors that occur during decoding or image processing.
        logger.error(f"Error during pyzbar decoding or image processing: {e}\n{traceback.format_exc()}")
        return f"Decoding error: {e}" # Return an error message to the user

def detect_and_decode_streamlit(image_pil): # Function designed to accept a PIL Image object from Streamlit
    """
    Main function: Detects barcodes/QR codes using YOLO, crops them,
    and then attempts to decode them using pyzbar with enhanced image processing.
    Returns only the decoded text(s).
    """
    global model # Access the global YOLO model

    # Check if the model was loaded successfully
    if model is None:
        st.error("Error: YOLO model not loaded. Please check the application logs.")
        return "" # Return empty string if model not available

    # Check if an image was uploaded
    if image_pil is None:
        return "No image uploaded. Please upload an image."

    # Convert the PIL Image object (from Streamlit) to an OpenCV format (BGR NumPy array)
    image_cv_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    # --- Image Resizing for overall efficiency ---
    # Define a maximum dimension for resizing to prevent excessively large images from slowing down processing.
    max_dim = 1024
    h, w = image_cv_bgr.shape[:2] # Get current height and width

    # Resize if either dimension exceeds the maximum
    if max(h, w) > max_dim:
        scaling_factor = max_dim / max(h, w)
        image_cv_bgr = cv2.resize(image_cv_bgr, (int(w * scaling_factor), int(h * scaling_factor)),
                                    interpolation=cv2.INTER_AREA) # Use INTER_AREA for shrinking
        logger.info(f"Image resized to {image_cv_bgr.shape[1]}x{image_cv_bgr.shape[0]} for efficiency.")


    try:
        # Run YOLO inference on the OpenCV image.
        results = model(image_cv_bgr)
        logger.info(f"YOLO inference completed.")

        # Check if any objects (barcodes/QR codes) were detected by YOLO
        if not results or not results[0].boxes:
            return "No objects detected by YOLO."

        # Extract bounding box coordinates from the YOLO results.
        detected_boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()

        # If no bounding boxes, return a specific message.
        if detected_boxes_xyxy.shape[0] == 0:
            return "No barcodes or QR codes detected by YOLO."
    except Exception as e:
        # Log and return an error if YOLO inference fails.
        logger.error(f"Error during YOLO model inference: {e}\n{traceback.format_exc()}")
        return f"Error during YOLO model inference: {e}\n{traceback.format_exc()}"

    decoded_texts_for_output = [] # List to store decoded texts for output to the user
    for i, box in enumerate(detected_boxes_xyxy):
        # Extract integer coordinates for cropping
        x1, y1, x2, y2 = map(int, box[:4])

        # Ensure coordinates are within the image boundaries to prevent errors during cropping.
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image_cv_bgr.shape[1], x2)
        y2 = min(image_cv_bgr.shape[0], y2)

        # Validate crop dimensions: width and height must be positive.
        if x1 >= x2 or y1 >= y2:
            logger.warning(f"Object {i+1}: Invalid crop dimensions ({x1},{y1},{x2},{y2}). Skipping.")
            continue # Skip to the next detected object

        # Crop the image to the detected object's bounding box.
        cropped_object_cv = image_cv_bgr[y1:y2, x1:x2]

        # Ensure the cropped image is not empty after validation.
        if cropped_object_cv.size == 0:
            logger.warning(f"Object {i+1}: Cropped area is empty after validation. Skipping.")
            continue

        # Decode the cropped object using the pyzbar decoding function.
        decoded_text = decode_barcode_from_pyzbar(cropped_object_cv)

        # Append the decoded text or a relevant message if decoding fails for this object.
        if decoded_text and decoded_text != "No barcode detected":
            decoded_texts_for_output.append(decoded_text)
        else:
            decoded_texts_for_output.append("No QR/Barcode found or readable for this object.")

    # Join all decoded texts into a single string, or return a general message if nothing was decoded.
    return "\n".join(decoded_texts_for_output) if decoded_texts_for_output else "No barcodes or QR codes were found or could be decoded in the image."


# --- Streamlit Interface ---
# Set general page configuration
st.set_page_config(page_title="YOLO Barcode & QR Code Detector & Decoder", layout="centered")

# Add title and description to the Streamlit app
st.title("YOLO Barcode & QR Code Detector & Decoder")
st.markdown(
    """
    Upload an image containing one or more barcodes or QR codes.
    The system uses a YOLO model to detect them, then attempts to decode each one using enhanced image processing techniques.
    """
)

# File uploader widget for image input.
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Process the uploaded file if one exists
if uploaded_file is not None:
    try:
        # Open the uploaded file as a PIL Image
        image = Image.open(uploaded_file)
        # Display the uploaded image in the Streamlit app
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("") # Add some vertical space
        st.write("Processing image...") # Inform the user that processing is underway

        # Call the detection and decoding function
        decoded_output = detect_and_decode_streamlit(image)
        
        # Display the results in a text area
        st.subheader("Decoded Output:")
        st.text_area("Decoded Text(s)", decoded_output, height=150)

    except Exception as e:
        # Display an error message to the user if an exception occurs
        st.error(f"An error occurred during image processing: {e}\n{traceback.format_exc()}")
        # Log the full traceback for debugging purposes
        logger.error(f"Error in Streamlit UI: {e}\n{traceback.format_exc()}")
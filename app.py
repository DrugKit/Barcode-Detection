import gradio as gr
import torch
from PIL import Image
import numpy as np
import cv2
from pyzbar.pyzbar import decode

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

def preprocess_for_pyzbar(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 3)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def detect_and_decode(image):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Run YOLO inference
    results = model(image)
    boxes = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, cls]

    decoded_results = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cropped = image_cv[y1:y2, x1:x2]

        # First try direct decoding
        decoded = decode(cropped)
        if not decoded:
            # If fails, try enhanced version
            enhanced = preprocess_for_pyzbar(cropped)
            decoded = decode(enhanced)

        # Collect results
        if decoded:
            for obj in decoded:
                decoded_results.append(f"Decoded: {obj.data.decode('utf-8')}")
        else:
            decoded_results.append("Detected region, but could not decode.")

    return decoded_results if decoded_results else ["No barcode or QR code detected"]

iface = gr.Interface(
    fn=detect_and_decode,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="Decoded Output"),
    title="YOLO Barcode/QR Detector + Pyzbar",
    description="Upload an image, detect barcodes/QR codes using YOLO, crop them, enhance only if needed, and decode using Pyzbar."
)

if __name__ == "__main__":
    iface.launch()

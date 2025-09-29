import time
import serial
import gradio as gr
import numpy as np
from PIL import Image

# Serial config
PORT = "COM5"  # update to your port
BAUD = 115200
TARGET_SIZE = (28, 28)
PREVIEW_SIZE = (128, 128)  # upscale for UI preview

# Open Serial once
ser = serial.Serial(PORT, BAUD, timeout=5)
time.sleep(2)  # give Arduino initial reset time


def send_image_serial(image_path):
    img = Image.open(image_path).convert("L")
    img_resized = img.resize(TARGET_SIZE)
    arr = np.array(img_resized, dtype=np.uint8).flatten()
    ser.write(arr.tobytes())

    # Step 5: read prediction from Arduino
    result = ser.readline().decode().strip()
    preview = img.resize(PREVIEW_SIZE)
    return result, preview


# Gradio UI with banner
with gr.Blocks() as demo:
    gr.Markdown("## Arduino Nano 33 BLE Serial Inference")
    gr.Markdown("Upload an image → resize to 28x28 → upscale preview (128x128) → send via Serial → get prediction")
    gr.Image("arduino-nano-33-BLE.jpg", show_label=False, elem_id="banner")

    with gr.Row():
        inp = gr.Image(type="filepath", label="Upload Image")
        out_text = gr.Textbox(label="Prediction")
        out_preview = gr.Image(label="Preview Sent to Arduino")

    inp.change(fn=send_image_serial, inputs=inp, outputs=[out_text, out_preview])

if __name__ == "__main__":
    demo.launch()

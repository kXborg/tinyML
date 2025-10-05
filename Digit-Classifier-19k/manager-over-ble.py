import time
import asyncio
import gradio as gr
import numpy as np
from PIL import Image
from bleak import BleakClient

# BLE config
DEVICE_ADDR = "84:45:7d:35:39:74"  # Replace with your Arduino Nano 33 BLE MAC
IMG_UUID = "19b10001-e8f2-537e-4f6c-d104768a1214"   # image write characteristic
RESULT_UUID = "19b10002-e8f2-537e-4f6c-d104768a1214"  # result notify characteristic

TARGET_SIZE = (28, 28)
PREVIEW_SIZE = (128, 128)
CHUNK = 100  # bytes per BLE write

# BLE send + receive 
async def send_image_ble(image_path):
    img = Image.open(image_path).convert("L")
    img_resized = img.resize(TARGET_SIZE)
    arr = np.array(img_resized, dtype=np.uint8).flatten()

    # Normalize and map to int8 (-128..127)
    arr = np.array(img_resized, dtype=np.float32).flatten()
    arr = (arr / 255.0 * 255 - 128).astype(np.int8)
    arr_uint8 = (arr.astype(np.int16) + 128).astype(np.uint8)
    data_bytes = arr_uint8.tobytes()

    async with BleakClient(DEVICE_ADDR) as client:
        if not client.is_connected:
            raise Exception("BLE connection failed")

        print("Connected to BLE device")

        result_text = None

        # Callback for receiving result
        def callback(sender, data):
            nonlocal result_text
            result_text = data.decode().strip()
            print("Received result:", result_text)

        await client.start_notify(RESULT_UUID, callback)

        # Send image in small chunks
        for i in range(0, len(data_bytes), CHUNK):
            await client.write_gatt_char(IMG_UUID, bytearray(data_bytes[i:i+CHUNK]), response=False)
            await asyncio.sleep(0.02)  # small delay for BLE stability

        # Wait for result
        print("Waiting for inference result...")
        for _ in range(100):  # wait up to ~5 seconds
            if result_text:
                break
            await asyncio.sleep(0.05)

        await client.stop_notify(RESULT_UUID)
        if result_text is None:
            result_text = "No response"

        return result_text, img.resize(PREVIEW_SIZE)


def send_image_sync(image_path):
    """Sync wrapper for Gradio."""
    return asyncio.run(send_image_ble(image_path))


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Arduino Nano 33 BLE Image Classifier")
    gr.Markdown(
        "Upload a grayscale image → resized to 28×28 → sent via BLE → model predicts digit."
    )
    gr.Image("../arduino-nano-33-BLE.jpg", show_label=False, elem_id="banner")

    with gr.Row():
        inp = gr.Image(type="filepath", label="Upload Image")
        out_text = gr.Textbox(label="Prediction")
        out_preview = gr.Image(label="Preview Sent to Arduino")

    inp.change(fn=send_image_sync, inputs=inp, outputs=[out_text, out_preview])

if __name__ == "__main__":
    demo.launch()

import asyncio
import gradio as gr
import numpy as np
from PIL import Image
from bleak import BleakClient

# BLE config
DEVICE_ADDR = "84:45:7d:35:39:74"  # Nano MAC
CHAR_UUID = "19b10001-e8f2-537e-4f6c-d104768a1214"  # image characteristic
RESULT_UUID = "19b10002-e8f2-537e-4f6c-d104768a1214"  # result characteristic

TARGET_SIZE = (8, 8)   # downsample to 8x8 for Arduino
PREVIEW_SIZE = (128, 128)  # upscale preview for humans

async def send_image_ble(flat_img):
    async with BleakClient(DEVICE_ADDR) as client:
        loop = asyncio.get_event_loop()
        result_future = loop.create_future()

        # notification callback
        def handle_result(sender, data):
            if not result_future.done():
                result_future.set_result(data.decode())

        # subscribe to result notifications
        await client.start_notify(RESULT_UUID, handle_result)

        # send image in chunks (20 bytes max per BLE packet)
        for i in range(0, len(flat_img), 20):
            chunk = bytearray(flat_img[i:i+20])
            await client.write_gatt_char(CHAR_UUID, chunk)
            await asyncio.sleep(0.01)

        # wait for Arduino prediction
        result = await result_future

        await client.stop_notify(RESULT_UUID)
        return result

def process_and_send(image):
    # Step 1: open and convert to grayscale
    img = Image.open(image).convert("L")
    # Step 2: resize to target size (for Arduino)
    img_resized = img.resize(TARGET_SIZE)
    # Step 3: normalize to 0–255 and flatten
    arr = np.array(img_resized, dtype=np.uint8).flatten().tolist()

    # Step 4: run async BLE send
    prediction = asyncio.run(send_image_ble(arr))

    # Step 5: upscale preview for UI
    preview = img_resized.resize(PREVIEW_SIZE, Image.NEAREST)

    return prediction, preview

# # Gradio UI
# demo = gr.Interface(
#     fn=process_and_send,
#     inputs=gr.Image(type="filepath", label="Upload Image"),
#     outputs=[
#         gr.Textbox(label="Prediction"),
#         gr.Image(label="Downsampled Preview (Upscaled)")
#     ],
#     title="Arduino BLE Inference",
#     description="Upload an image → downsample to 8x8 → upscale preview (128x128) → send via BLE to Nano 33 BLE → get prediction"
# )

# Gradio UI with banner
with gr.Blocks() as demo:
    gr.Markdown("## Arduino BLE Inference")
    gr.Markdown("Upload an image → downsample to 8x8 → upscale preview (128x128) → send via BLE to Nano 33 BLE → get prediction")
    gr.Image("arduino-nano-33-BLE.jpg", show_label=False, elem_id="banner")

    with gr.Row():
        inp = gr.Image(type="filepath", label="Upload Image")
        out_text = gr.Textbox(label="Prediction")
        out_preview = gr.Image(label="Downsampled Preview (Upscaled)")

    inp.change(fn=process_and_send, inputs=inp, outputs=[out_text, out_preview])


if __name__ == "__main__":
    demo.launch()

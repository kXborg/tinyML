import asyncio
from bleak import BleakClient

DEVICE_ADDR = "84:45:7d:35:39:74"  # Nano MAC
CHAR_UUID = "19b10001-e8f2-537e-4f6c-d104768a1214"  # image
RESULT_UUID = "19b10002-e8f2-537e-4f6c-d104768a1214"  # result

digit_image = [
    0,0,255,255,255,0,0,0,
    0,255,0,0,0,255,0,0,
    0,255,0,0,0,255,0,0,
    0,255,0,0,0,255,0,0,
    0,255,0,0,0,255,0,0,
    0,255,0,0,0,255,0,0,
    0,0,255,255,255,0,0,0,
    0,0,0,0,0,0,0,0
]

async def run():
    async with BleakClient(DEVICE_ADDR) as client:
        result_future = asyncio.get_event_loop().create_future()

        # notification callback
        def handle_result(sender, data):
            result_future.set_result(data.decode())

        # subscribe to notifications
        await client.start_notify(RESULT_UUID, handle_result)

        # send image in chunks
        for i in range(0, 64, 20):
            chunk = bytearray(digit_image[i:i+20])
            await client.write_gatt_char(CHAR_UUID, chunk)
            await asyncio.sleep(0.01)

        print("Image sent! Waiting for prediction...")

        # wait for notification
        result = await result_future
        print("Prediction:", result)

        # stop notifications
        await client.stop_notify(RESULT_UUID)

asyncio.run(run())

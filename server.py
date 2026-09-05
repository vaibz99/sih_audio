import asyncio
import websockets
from datetime import datetime

async def handler(websocket):
    print(f"[SERVER] Client connected.")
    async for message in websocket:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] Received: {message}")

async def main():
    print("Server listening on ws://localhost:8765 ...")
    async with websockets.serve(handler, "localhost", 8765):
        await asyncio.Future()  # run forever

asyncio.run(main())
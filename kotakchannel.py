from telethon import TelegramClient
import os

api_id = '22223100'
api_hash = '181f5cc4377c18f34dc7e93fbcd2c713'
channel_id = -1001780591143  # Use your actual channel ID

client = TelegramClient('session_name', api_id, api_hash)

async def main():
    # Make a folder for downloads if it doesn't exist
    os.makedirs('downloads', exist_ok=True)
    # Fetch messages from the channel
    async for message in client.iter_messages(channel_id, limit=100):  # adjust limit as needed
        # Check if the message has a document and it's a PDF
        if message.document and message.document.mime_type == 'application/pdf':
            file_name = message.file.name or f"{message.id}.pdf"
            file_path = os.path.join('downloads', file_name)
            print(f"Downloading: {file_name}")
            await message.download_media(file_path)

with client:
    client.loop.run_until_complete(main())
from telethon import TelegramClient
import os
import re
import pdfplumber
import pandas as pd
from datetime import datetime, timedelta


# === Configuration ===
api_id = '22223100'
api_hash = '181f5cc4377c18f34dc7e93fbcd2c713'
channel_id = -1001780591143

PDF_FOLDER = r'E:\Projects\MCX-Silver\downloads'
EXCEL_PATH = r'E:\Projects\MCX-Silver\silver_combined_news1113.xlsx'

include_keywords = ["gold", "silver", "strikes", "strike", "inflation", "interest rates", "central bank", "missile", "US", "donald trump", "president", "strategic", "war", "attack"]
exclude_keywords = ["hi", "kcalrt"]
min_length = 75

# Set how many PDFs to download
MAX_PDFS_TO_DOWNLOAD = 500

client = TelegramClient('session_name', api_id, api_hash)


def clean_text(text):
    """
    Clean text by removing links and non-alphanumeric characters (except % and &)
    """
    if not text:
        return text
    
    # Remove URLs (http/https links)
    text = re.sub(r'https?://[^\s]+', '', text)
    
    # Remove other common link patterns (www.example.com, t.me/channel, etc.)
    text = re.sub(r'(?:www\.)[^\s]+', '', text)
    text = re.sub(r'(?:t\.me/)[^\s]+', '', text)
    text = re.sub(r'(?:telegram\.me/)[^\s]+', '', text)
    
    # Remove non-alphanumeric characters except % and & (keep spaces, letters, numbers, % and &)
    text = re.sub(r'[^a-zA-Z0-9\s%&,.]', ' ', text)
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def extract_bullion_news_from_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ''
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + '\n'

        BULLION_PATTERN = re.compile(
            r'Bullion\s*[\u2013-]?\s*(.*?)(?=\n[A-Z][a-zA-Z ]+:|\nCrude Oil|\nENERGY|\nBASE METALS|\nLME BASE METALS|\nCopper|\nAluminium|\nBrent Crude Oil)',
            re.DOTALL
        )
        bullion_match = BULLION_PATTERN.search(text)
        bullion_news = bullion_match.group(1).replace('\n', ' ').strip() if bullion_match else ''
        return bullion_news
    except Exception as e:
        print(f"❌ Error extracting from PDF {pdf_path}: {e}")
        return ''

def extract_date_from_filename(filename):
    date_pattern = re.compile(r'-(\d{1,2}-[A-Za-z]{3} \d{4})')
    match = date_pattern.search(filename)
    if match:
        try:
            return pd.to_datetime(match.group(1), format='%d-%b %Y')
        except Exception:
            return None
    return None


def load_existing_data(excel_path):
    if os.path.exists(excel_path):
        try:
            return pd.read_excel(excel_path)
        except Exception as e:
            print(f"❌ Error loading existing Excel file: {e}")
            return pd.DataFrame(columns=['headline', 'date', 'source'])
    return pd.DataFrame(columns=['headline', 'date', 'source'])


async def extract_telegram_news():
    telegram_rows = []
    pdf_download_count = 0
    total_messages = 0
    pdf_messages_found = 0

    # Precompile regex patterns for efficiency
    include_pattern = re.compile(r'(' + '|'.join([re.escape(k) for k in include_keywords]) + r')', re.IGNORECASE)
    exclude_pattern = re.compile(r'(' + '|'.join([re.escape(k) for k in exclude_keywords]) + r')', re.IGNORECASE)

    print("🔄 Starting to process Telegram messages...")

    async for message in client.iter_messages(channel_id, limit=17000):
        total_messages += 1

        # Progress indicator
        if total_messages % 100 == 0:
            print(f"📊 Processed {total_messages} messages, found {pdf_messages_found} PDFs, downloaded {pdf_download_count}")

        message_text = (message.text or "").strip()

        # Handle PDF downloads
        if (
            message.document
            and message.document.mime_type == 'application/pdf'
            and hasattr(message.file, 'name')
            and message.file.name
            and message.file.name.startswith("KSEC- Morning Insight")
        ):
            pdf_messages_found += 1
            file_name = message.file.name
            file_path = os.path.join(PDF_FOLDER, file_name)

            print(f"🔍 Found PDF: {file_name} (Date: {message.date.strftime('%Y-%m-%d')})")

            # Skip if already downloaded
            if os.path.exists(file_path):
                print(f"⏩ Already exists: {file_name}")
                continue

            if pdf_download_count >= MAX_PDFS_TO_DOWNLOAD:
                print("📦 Reached PDF download limit.")
                break

            os.makedirs(PDF_FOLDER, exist_ok=True)
            try:
                print(f"📥 Downloading PDF: {file_name}")
                await message.download_media(file_path)
                pdf_download_count += 1
                print(f"✅ Downloaded: {file_name}")
            except Exception as e:
                print(f"❌ Download failed for {file_name}: {e}")

        # Handle text messages: clean text and apply filters
        elif (
            not message.document  # Not a document
            and message_text  # Has text
        ):
            # Clean the text first
            cleaned_text = clean_text(message_text)
            
            # Apply filters on cleaned text
            if (
                cleaned_text  # Has text after cleaning
                and len(cleaned_text) >= min_length
                and include_pattern.search(cleaned_text)  # Must include at least one include_keyword
                and not exclude_pattern.search(cleaned_text)  # Must NOT include any exclude_keyword
            ):
                message_date = message.date.strftime("%Y-%m-%d %H:%M:%S")
                telegram_rows.append({
                    'headline': cleaned_text,
                    'date': message_date,
                    'source': 'telegram'
                })
                print(f"📝 Added cleaned text message from {message.date.strftime('%Y-%m-%d')}")

    print(f"\n📊 Final Stats:")
    print(f"   Total messages processed: {total_messages}")
    print(f"   PDF messages found: {pdf_messages_found}")
    print(f"   PDFs downloaded: {pdf_download_count}")
    print(f"   Text messages added: {len(telegram_rows)}")

    return telegram_rows

def extract_pdf_news():
    print("\n🔄 Extracting news from downloaded PDFs...")
    extracted_data = []
    
    if not os.path.exists(PDF_FOLDER):
        print(f"❌ PDF folder doesn't exist: {PDF_FOLDER}")
        return extracted_data
    
    pdf_files = [f for f in os.listdir(PDF_FOLDER) 
                 if f.startswith('KSEC- Morning Insight') and f.endswith('.pdf')]
    
    print(f"📁 Found {len(pdf_files)} PDF files to process")
    
    for filename in pdf_files:
        pdf_path = os.path.join(PDF_FOLDER, filename)
        try:
            news = extract_bullion_news_from_pdf(pdf_path)
            date = extract_date_from_filename(filename)
            
            if news and date:
                extracted_data.append({
                    'headline': news, 
                    'date': date, 
                    'source': 'pdf'
                })
                print(f"✅ Extracted from PDF: {filename}")
            else:
                print(f"⚠️ Incomplete extraction from: {filename} (news: {bool(news)}, date: {bool(date)})")
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
    
    print(f"📊 Successfully extracted from {len(extracted_data)} PDFs")
    return extracted_data


async def main():
    print("🚀 Starting MCX Silver News Extraction")
    print(f"📅 Current date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load existing data
    print("\n📂 Loading existing data...")
    combined_df = load_existing_data(EXCEL_PATH)
    print(f"📊 Existing records: {len(combined_df)}")
    
    # Extract new data from Telegram
    print("\n📱 Extracting from Telegram...")
    telegram_data = await extract_telegram_news()
    
    # Extract data from PDFs
    pdf_data = extract_pdf_news()
    
    # Combine all new data
    all_new_data = telegram_data + pdf_data
    new_df = pd.DataFrame(all_new_data)
    
    if new_df.empty:
        print("\n📭 No new data to add.")
        return

    print(f"\n🔄 Processing {len(new_df)} new records...")
    
    # Combine with existing data
    combined_df = pd.concat([combined_df, new_df], ignore_index=True)
    
    # Remove duplicates
    initial_count = len(combined_df)
    combined_df.drop_duplicates(subset=['headline'], inplace=True)
    final_count = len(combined_df)
    duplicates_removed = initial_count - final_count
    
    # Clean and process data
    combined_df['date'] = pd.to_datetime(combined_df['date'], errors='coerce')
    combined_df = combined_df.dropna(subset=['date'])
    combined_df['source'] = combined_df['source'].fillna("unknown")
    combined_df.sort_values(by='date', ascending=False, inplace=True)
    combined_df.reset_index(drop=True, inplace=True)

    # Save to Excel
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(EXCEL_PATH), exist_ok=True)
        combined_df.to_excel(EXCEL_PATH, index=False)
        print(f"\n✅ Excel file updated successfully: {EXCEL_PATH}")
        print(f"📊 Total records: {len(combined_df)}")
        print(f"🆕 New records added: {len(all_new_data)}")
        print(f"🗑️ Duplicates removed: {duplicates_removed}")
        
        # Show latest records
        if len(combined_df) > 0:
            print(f"\n📅 Latest record date: {combined_df.iloc[0]['date']}")
            print(f"📅 Oldest record date: {combined_df.iloc[-1]['date']}")
            
    except Exception as e:
        print(f"❌ Error saving Excel file: {e}")


if __name__ == "__main__":
    try:
        with client:
            client.loop.run_until_complete(main())
        print("\n🎉 Process completed successfully!")
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        import traceback
        traceback.print_exc()
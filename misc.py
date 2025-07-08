import pandas as pd

# Load your Excel file
file_path = r"E:\Projects\MCX-Silver\combined.xlsx"  # Replace with your actual path
df = pd.read_excel(file_path)

# Assign importance values based on the source
df['importance'] = df['Source'].apply(
    lambda src: 0.80 if src == 'telegram' else 0.95 if src == 'pdf' else 0.2 if src == 'mint' else 0.4
)

# Optionally, save the updated DataFrame to a new Excel file
output_path = r"E:\Projects\MCX-Silver\combined.xlsx" # Change if needed
df.to_excel(output_path, index=False)

import yfinance as yf

# Define the ticker symbol for US Dollar Index
usd_index_ticker = "DX-Y.NYB"  # Try "DX=F" if this doesn't work

# Download historical data (you can set your own start/end dates)
df = yf.download(usd_index_ticker, start="2023-01-01", end="2025-07-04")

# Show the first few rows
print(df.head())

# Save to CSV
df.to_csv("E:\Projects\MCX-Silver\Data\dollar_index.csv")
print("USD Index data saved to usd_index_yfinance.csv")



print("Importance column added and file saved.")

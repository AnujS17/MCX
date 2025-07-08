import pandas as pd
import numpy as np
from datetime import datetime
import os
import glob

def load_and_prepare_data():
    """Load all data files and prepare them for merging"""

    # Load Gold Futures Historical Data
    try:
        gold_futures = pd.read_csv(r"E:\Projects\MCX-Silver\Data\Gold_Futures_Historical_Data.csv")
        def parse_gold_date(date_str):
            date_str = str(date_str).strip()
            try:
                return pd.to_datetime(date_str, format='%m/%d/%Y')
            except:
                try:
                    return pd.to_datetime(date_str, format='%m-%d-%Y')
                except:
                    try:
                        return pd.to_datetime(date_str)
                    except:
                        return pd.NaT
        gold_futures['Date'] = gold_futures['Date'].apply(parse_gold_date)
        gold_futures = gold_futures.dropna(subset=['Date'])
        price_columns = ['Price', 'Open', 'High', 'Low']
        for col in price_columns:
            if col in gold_futures.columns:
                gold_futures[col] = gold_futures[col].astype(str)
                gold_futures[col] = gold_futures[col].str.replace('"', '', regex=False)
                gold_futures[col] = gold_futures[col].str.replace(',', '', regex=False)
                gold_futures[col] = pd.to_numeric(gold_futures[col], errors='coerce')
        gold_futures = gold_futures.rename(columns={
            'Price': 'Gold_Global_Rate_USD',
            'Open': 'Gold_Open_USD',
            'High': 'Gold_High_USD',
            'Low': 'Gold_Low_USD',
            'Vol.': 'Gold_Volume',
            'Change %': 'Gold_Change_Pct'
        })
    except Exception as e:
        print(f"Error loading gold futures data: {e}")
        gold_futures = pd.DataFrame()

    # Load Silver Futures Historical Data
    try:
        silver_futures = pd.read_csv(r"E:\Projects\MCX-Silver\Data\Silver_Futures_Historical_Data.csv")
        def parse_silver_date(date_str):
            date_str = str(date_str).strip()
            try:
                return pd.to_datetime(date_str, format='%m/%d/%Y')
            except:
                try:
                    return pd.to_datetime(date_str, format='%m-%d-%Y')
                except:
                    try:
                        return pd.to_datetime(date_str)
                    except:
                        return pd.NaT
        silver_futures['Date'] = silver_futures['Date'].apply(parse_silver_date)
        silver_futures = silver_futures.dropna(subset=['Date'])
        price_columns = ['Price', 'Open', 'High', 'Low']
        for col in price_columns:
            if col in silver_futures.columns:
                silver_futures[col] = silver_futures[col].astype(str)
                silver_futures[col] = silver_futures[col].str.replace('"', '', regex=False)
                silver_futures[col] = silver_futures[col].str.replace(',', '', regex=False)
                silver_futures[col] = pd.to_numeric(silver_futures[col], errors='coerce')
        silver_futures = silver_futures.rename(columns={
            'Price': 'Silver_Global_Rate_USD',
            'Open': 'Silver_Open_USD',
            'High': 'Silver_High_USD',
            'Low': 'Silver_Low_USD',
            'Vol.': 'Silver_Volume',
            'Change %': 'Silver_Change_Pct'
        })
    except Exception as e:
        print(f"Silver futures file not found or error: {e}")
        silver_futures = pd.DataFrame()

    # Load Dollar Index data
    try:
        dollar_index = pd.read_csv(r"E:\Projects\MCX-Silver\Data\dollar_index.csv")
        dollar_index = dollar_index.iloc[3:].copy()
        dollar_index.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
        dollar_index['Date'] = pd.to_datetime(dollar_index['Date'], errors='coerce')
        dollar_index['USD_Index_Value'] = pd.to_numeric(dollar_index['Close'], errors='coerce')
        dollar_index = dollar_index[['Date', 'USD_Index_Value']].dropna()
    except Exception as e:
        print(f"Error loading dollar index data: {e}")
        dollar_index = pd.DataFrame()

    # Load India Inflation data
    try:
        inflation_data = pd.read_excel(r"E:\Projects\MCX-Silver\Data\india_inflation_jan2023_to_may2025.xlsx")
        inflation_data['Month'] = pd.to_datetime(inflation_data['Month'], format='%b %Y', errors='coerce')
    except Exception as e:
        print(f"Error loading inflation data: {e}")
        inflation_data = pd.DataFrame()

    # Load India Interest Rates data
    try:
        interest_rates = pd.read_excel(r"E:\Projects\MCX-Silver\Data\india_interest_rates_rbi_2020_2025.xlsx")
        try:
            interest_rates['Date'] = pd.to_datetime(interest_rates['Date'], format='%d %b %Y', errors='coerce')
        except:
            interest_rates['Date'] = pd.to_datetime(interest_rates['Date'], errors='coerce')
    except Exception as e:
        print(f"Error loading interest rates data: {e}")
        interest_rates = pd.DataFrame()

    # Load and combine MCX Silver data from monthly files
    mcx_data = load_mcx_monthly_files()

    # Load news data
    try:
        news_data = pd.read_excel(r"E:\Projects\MCX-Silver\silver_combined_news111.xlsx")
        if 'source' not in news_data.columns:
            news_data['source'] = 'Unknown'
        try:
            news_data['date'] = pd.to_datetime(news_data['date'], errors='coerce')
        except:
            news_data['date'] = pd.Timestamp.now()
    except Exception as e:
        print(f"Error loading news data: {e}")
        news_data = pd.DataFrame()

    return gold_futures, silver_futures, dollar_index, inflation_data, interest_rates, mcx_data, news_data

def load_mcx_monthly_files(folder_path=r'E:\Projects\MCX-Silver\Data'):
    """Load and combine MCX silver data from multiple monthly Excel files"""
    file_pattern = os.path.join(folder_path, 'filtered_data_*.xlsx')
    mcx_files = glob.glob(file_pattern)
    if not mcx_files:
        print("No MCX data files found! Please check the file pattern.")
        return pd.DataFrame()
    all_mcx_data = []
    for file_path in mcx_files:
        try:
            excel_file = pd.ExcelFile(file_path)
            if 'Silver Futures' in excel_file.sheet_names:
                mcx_monthly = pd.read_excel(file_path, sheet_name='Silver Futures')
                mcx_monthly = mcx_monthly.iloc[3:].copy()
                expected_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Open_Interest', 'Change', 'Change_%']
                if len(mcx_monthly.columns) >= len(expected_columns):
                    mcx_monthly.columns = expected_columns[:len(mcx_monthly.columns)]
                mcx_monthly['Date'] = pd.to_datetime(mcx_monthly['Date'], errors='coerce')
                mcx_monthly = mcx_monthly.dropna(subset=['Date'])
                mcx_monthly_first = mcx_monthly.groupby('Date').first().reset_index()
                column_mapping = {
                    'Open': 'MCX_Silver_Open_INR',
                    'High': 'MCX_Silver_High_INR',
                    'Low': 'MCX_Silver_Low_INR',
                    'Close': 'MCX_Silver_Close_INR',
                    'Volume': 'MCX_Silver_Volume_Lots',
                    'Open_Interest': 'MCX_Silver_Open_Interest'
                }
                for old_col, new_col in column_mapping.items():
                    if old_col in mcx_monthly_first.columns:
                        mcx_monthly_first = mcx_monthly_first.rename(columns={old_col: new_col})
                required_columns = [
                    'MCX_Silver_Settlement_INR', 
                    'MCX_Silver_Spot_INR'
                ]
                for col in required_columns:
                    if col not in mcx_monthly_first.columns:
                        mcx_monthly_first[col] = np.nan
                all_mcx_data.append(mcx_monthly_first)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    if all_mcx_data:
        combined_mcx = pd.concat(all_mcx_data, ignore_index=True)
        combined_mcx = combined_mcx.drop_duplicates(subset=['Date']).sort_values('Date')
        return combined_mcx
    else:
        return pd.DataFrame()

def move_weekend_to_monday(date_series):
    """
    For a pandas Series of datetime, move any Saturday/Sunday to the following Monday.
    """
    dates = pd.to_datetime(date_series, errors='coerce')
    # Weekday: Monday=0, ..., Saturday=5, Sunday=6
    dates = dates + dates.dt.weekday.map({5: 2, 6: 1}).fillna(0).astype('timedelta64[D]')
    return dates

def create_headline_based_excel():
    """Create Excel file where each row represents a unique headline with corresponding market data and source"""
    gold_futures, silver_futures, dollar_index, inflation_data, interest_rates, mcx_data, news_data = load_and_prepare_data()
    if news_data.empty:
        print("No news data available. Creating sample data structure.")
        return pd.DataFrame()
    combined_df = news_data.copy()
    combined_df = combined_df.rename(columns={'date': 'Date', 'headline': 'Headlines'})
    if 'source' not in combined_df.columns:
        combined_df['source'] = 'Unknown'
    else:
        combined_df['source'] = combined_df['source'].fillna('Unknown')
    combined_df['Date'] = pd.to_datetime(combined_df['Date']).dt.date
    combined_df['Date'] = pd.to_datetime(combined_df['Date'])
    # Move weekend-dated headlines to Monday
    combined_df['Date'] = move_weekend_to_monday(combined_df['Date'])
    # Merge Gold data based on date
    if not gold_futures.empty and 'Gold_Global_Rate_USD' in gold_futures.columns:
        gold_futures['Date'] = pd.to_datetime(gold_futures['Date']).dt.date
        gold_futures['Date'] = pd.to_datetime(gold_futures['Date'])
        gold_select = gold_futures[['Date', 'Gold_Global_Rate_USD']].copy()
        combined_df = pd.merge(combined_df, gold_select, on='Date', how='left')
    # Merge Silver data based on date
    if not silver_futures.empty and 'Silver_Global_Rate_USD' in silver_futures.columns:
        silver_futures['Date'] = pd.to_datetime(silver_futures['Date']).dt.date
        silver_futures['Date'] = pd.to_datetime(silver_futures['Date'])
        silver_select = silver_futures[['Date', 'Silver_Global_Rate_USD']].copy()
        combined_df = pd.merge(combined_df, silver_select, on='Date', how='left')
    # Merge USD Index data based on date
    if not dollar_index.empty:
        dollar_index['Date'] = pd.to_datetime(dollar_index['Date']).dt.date
        dollar_index['Date'] = pd.to_datetime(dollar_index['Date'])
        usd_select = dollar_index[['Date', 'USD_Index_Value']].copy()
        combined_df = pd.merge(combined_df, usd_select, on='Date', how='left')
    # Merge MCX Silver data
    if not mcx_data.empty:
        mcx_data['Date'] = pd.to_datetime(mcx_data['Date']).dt.date
        mcx_data['Date'] = pd.to_datetime(mcx_data['Date'])
        mcx_columns = ['Date']
        available_mcx_columns = [col for col in [
            'MCX_Silver_Open_INR', 'MCX_Silver_High_INR',
            'MCX_Silver_Low_INR', 'MCX_Silver_Close_INR',
            'MCX_Silver_Settlement_INR', 'MCX_Silver_Spot_INR',
            'MCX_Silver_Volume_Lots', 'MCX_Silver_Open_Interest'
        ] if col in mcx_data.columns]
        mcx_columns.extend(available_mcx_columns)
        mcx_select = mcx_data[mcx_columns].copy()
        combined_df = pd.merge(combined_df, mcx_select, on='Date', how='left')
    # Add inflation data
    if not inflation_data.empty:
        combined_df['Month'] = combined_df['Date'].dt.to_period('M').dt.to_timestamp()
        inflation_select = inflation_data[['Month', 'Inflation Rate (%)']].copy()
        combined_df = pd.merge(combined_df, inflation_select, on='Month', how='left')
        combined_df = combined_df.drop(columns=['Month'])
    # Add interest rates data
    if not interest_rates.empty:
        combined_df_sorted = combined_df.sort_values('Date').reset_index(drop=True)
        interest_rates_sorted = interest_rates.sort_values('Date').reset_index(drop=True)
        combined_df_final = pd.merge_asof(
            combined_df_sorted, 
            interest_rates_sorted[['Date', 'Repo Rate (%)']],
            on='Date',
            direction='backward'
        )
    else:
        combined_df_final = combined_df.copy()
    # Rename source column
    combined_df_final = combined_df_final.rename(columns={'source': 'Source'})
    # Reorder columns
    base_columns = [
        'Headlines',
        'Date',
        'Source'
    ]
    optional_columns = [
        'Repo Rate (%)',
        'Inflation Rate (%)',
        'Gold_Global_Rate_USD',
        'Silver_Global_Rate_USD',
        'USD_Index_Value'
    ]
    mcx_columns = [
        'MCX_Silver_Open_INR',
        'MCX_Silver_High_INR',
        'MCX_Silver_Low_INR',
        'MCX_Silver_Close_INR',
        'MCX_Silver_Settlement_INR',
        'MCX_Silver_Spot_INR',
        'MCX_Silver_Volume_Lots',
        'MCX_Silver_Open_Interest'
    ]
    final_columns = base_columns.copy()
    for col in optional_columns + mcx_columns:
        if col in combined_df_final.columns:
            final_columns.append(col)
    combined_df_final = combined_df_final[final_columns]
    combined_df_final = combined_df_final.sort_values('Date')
    output_filename = 'combined.xlsx'
    combined_df_final.to_excel(output_filename, index=False)
    print(f"\nExcel file created successfully: {output_filename}")
    print(f"Total headlines (rows): {len(combined_df_final)}")
    if len(combined_df_final) > 0:
        print(f"Date range: {combined_df_final['Date'].min().strftime('%Y-%m-%d')} to {combined_df_final['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Columns: {list(combined_df_final.columns)}")
    return combined_df_final

if __name__ == "__main__":
    result_df = create_headline_based_excel()

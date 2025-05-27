import pandas as pd
import numpy as np
from multiprocessing import Pool
from io import StringIO
import re

# BAs export from sierra
data = """
Drawing Type | Symbol | Locked | Hidden | Alert | Anchors | ID | Text
Horizontal Line Non-Extended | ESM24 |  |  |  | A1(2024-04-15  12:30:00, 5071.75), A2(2024-04-18  15:00:03, 5071.75) | -1142 | PBAL
Horizontal Ray | ESM24 |  |  |  | A1(2024-05-02  15:00:02, 5112.75) | -1148 | 
Horizontal Ray | ESM24 |  |  |  | A1(2024-02-20  14:30:00, 4955) | -1136 | 
Horizontal Ray | ESM24 |  |  |  | A1(2024-04-09  09:00:00, 5221.75) | -1156 | 
Extending Rectangle | ESM24 |  |  |  | A1(2024-04-09  15:00:03, 5264.25), A2(2024-04-11  15:00:06, 5233.25) | -1139 | 
Rectangle | ESM24 |  |  |  | A1(2024-04-16  15:00:03, 5111.25), A2(2024-04-24  15:00:03, 5090.25) | -1141 | 
Rectangle | ESM24 |  |  |  | A1(2024-04-11  15:00:04, 5211.75), A2(2024-05-08  15:00:03, 5194) | -1140 | 
Extending Rectangle | ESM24 |  |  |  | A1(2024-05-08  15:00:03, 5219.5), A2(2024-05-09  15:00:02, 5205.5) | -1149 | 
Rectangle | ESM24 |  |  |  | A1(2022-02-23  11:00:03, 4326.25), A2(2022-03-04  15:00:03, 4284.25) | -1085 | 
Rectangle | ESM24 |  |  |  | A1(2022-02-08  15:00:03, 4512.75), A2(2022-03-24  15:00:03, 4473.25) | -1083 | 
Rectangle | ESM24 |  |  |  | A1(2022-03-28  15:00:03, 4538.25), A2(2023-07-17  15:00:03, 4508.75) | -1089 | 
Rectangle | ESM24 |  |  |  | A1(2022-03-24  15:00:03, 4503.75), A2(2022-04-08  15:00:03, 4469.75) | -1088 | 
Rectangle | ESM24 |  |  |  | A1(2023-02-16  15:00:03, 4149.5), A2(2023-04-17  15:00:03, 4121) | -1095 | 
Rectangle | ESM24 |  |  |  | A1(2023-04-19  15:00:03, 4182.5), A2(2023-10-31  11:00:03, 4169.5) | -1097 | 
Rectangle | ESM24 |  |  |  | A1(2023-04-17  15:00:03, 4158.75), A2(2023-04-24  15:00:03, 4115) | -1151 | 
Rectangle | ESM24 |  |  |  | A1(2023-04-24  15:00:03, 4159.75), A2(2023-05-17  11:00:03, 4145.25) | -1096 | 
Rectangle | ESM24 |  |  |  | A1(2022-04-19  15:00:03, 4399), A2(2023-06-14  15:00:03, 4454.75) | -1092 | 
Rectangle | ESM24 |  |  |  | A1(2023-06-22  15:00:03, 4419.5), A2(2023-10-12  15:00:03, 4406.25) | -1101 | 
Horizontal Ray | ESM24 |  |  |  | A1(2023-04-24  15:00:03, 4115) | -1152 | PBAL
Horizontal Line Non-Extended | ESM24 |  |  |  | A1(2022-03-04  15:00:02, 4284.25), A2(2023-06-08  15:00:03, 4284.25) | -1150 | PBAL
Rectangle | ESM24 |  |  |  | A1(2022-02-16  15:00:03, 4463.5), A2(2022-04-08  15:00:03, 4434.75) | -1084 | 
Rectangle | ESM24 |  |  |  | A1(2022-03-04  15:00:03, 4383.75), A2(2023-06-26  15:00:03, 4311.25) | -1086 | 
Rectangle | ESM24 |  |  |  | A1(2023-05-17  11:00:03, 4151.5), A2(2023-10-31  11:00:03, 4130) | -1098 | 
Rectangle | ESM24 |  |  |  | A1(2023-06-14  15:00:03, 4423.25), A2(2023-06-22  15:00:03, 4401.25) | -1100 | 
Rectangle | ESM24 |  |  |  | A1(2022-04-08  15:00:03, 4493.25), A2(2023-07-05  15:00:03, 4454.25) | -1091 | 
Rectangle | ESM24 |  |  |  | A1(2022-03-31  13:30:03, 4610.75), A2(2023-07-26  15:00:03, 4580.5) | -1090 | 
Extending Rectangle | ESM24 |  |  |  | A1(2023-08-01  15:00:03, 4610.25), A2(2023-08-03  08:30:02, 4597.5) | -1109 | 
Rectangle | ESM24 |  |  |  | A1(2023-07-17  15:00:03, 4557), A2(2023-08-07  15:00:03, 4536) | -1107 | 
Rectangle | ESM24 |  |  |  | A1(2023-07-05  15:00:03, 4491), A2(2023-08-14  15:00:03, 4480) | -1105 | 
Extending Rectangle | ESM24 |  |  |  | A1(2023-08-14  15:00:03, 4506), A2(2023-11-17  15:00:08, 4475) | -1112 | 
Horizontal Line Non-Extended | ESM24 |  |  |  | A1(2023-05-17  10:00:00, 4160), A2(2023-10-31  11:00:03, 4160) | -1158 | PBAH
Horizontal Line Non-Extended | ESM24 |  |  |  | A1(2023-07-26  11:00:00, 4610.75), A2(2023-08-01  15:00:03, 4610.75) | -1159 | PBAH
Rectangle | ESM24 |  |  |  | A1(2023-08-07  15:00:03, 4548), A2(2023-09-05  15:00:03, 4518) | -1110 | 
Horizontal Ray | ESM24 |  |  |  | A1(2023-09-05  14:00:00, 4548) | -1161 | PBAH
Horizontal Line Non-Extended | ESM24 |  |  |  | A1(2023-07-05  14:30:00, 4454.25), A2(2023-09-08  15:00:03, 4454.25) | -1104 | PBAL
Extending Rectangle | ESM24 |  |  |  | A1(2023-09-08  15:00:03, 4471.75), A2(2023-09-12  08:30:07, 4450.75) | -1114 | 
Rectangle | ESM24 |  |  |  | A1(2023-07-10  15:00:03, 4450.25), A2(2023-09-08  15:00:03, 4430) | -1106 | 
Horizontal Ray | ESM24 |  |  |  | A1(2023-09-08  15:00:02, 4430) | -1115 | PBAL
Rectangle | ESM24 |  |  |  | A1(2023-09-05  15:00:03, 4528.25), A2(2023-09-14  15:00:03, 4509.75) | -1113 | 
Horizontal Line Non-Extended | ESM24 |  |  |  | A1(2023-06-26  15:00:03, 4311.25), A2(2023-10-02  15:00:02, 4311.25) | -1103 | PBAL
Extending Rectangle | ESM24 |  |  |  | A1(2023-10-02  15:00:03, 4346.25), A2(2023-10-05  08:30:02, 4308.75) | -1117 | 
Rectangle | ESM24 |  |  |  | A1(2023-06-08  15:00:03, 4295.25), A2(2023-10-06  15:00:04, 4276.75) | -1099 | 
Rectangle | ESM24 |  |  |  | A1(2023-06-26  15:00:03, 4397.5), A2(2023-10-12  15:00:03, 4378.75) | -1102 | 
Rectangle | ESM24 |  |  |  | A1(2023-10-12  15:00:03, 4411.5), A2(2023-10-17  15:00:03, 4390.5) | -1119 | 
Horizontal Ray | ESM24 |  |  |  | A1(2023-10-12  15:00:02, 4419.5) | -1162 | PBAH
Extending Rectangle | ESM24 |  |  |  | A1(2023-10-25  15:00:03, 4280.75), A2(2023-10-27  10:30:00, 4254) | -1121 | 
Rectangle | ESM24 |  |  |  | A1(2023-10-06  15:00:03, 4286.25), A2(2023-10-25  15:00:03, 4257) | -1118 | 
Extending Rectangle | ESM24 |  |  |  | A1(2023-10-31  11:00:03, 4189.25), A2(2023-11-02  11:30:00, 4154.25) | -1164 | 
Horizontal Ray | ESM24 |  |  |  | A1(2023-10-31  11:00:02, 4130) | -1165 | PBAL
Extending Rectangle | ESM24 |  |  |  | A1(2023-11-06  15:00:03, 4387.25), A2(2023-11-09  12:00:09, 4372) | -1122 | 
Rectangle | ESM24 |  |  |  | A1(2023-10-17  15:00:03, 4410.25), A2(2023-11-09  15:00:03, 4388) | -1120 | 
Extending Rectangle | ESM24 |  |  |  | A1(2023-11-09  15:00:03, 4402.75), A2(2023-11-10  08:30:00, 4389.25) | -1123 | 
Rectangle | ESM24 |  |  |  | A1(2023-09-14  15:00:03, 4532.5), A2(2023-11-17  15:00:03, 4514.75) | -1116 | 
Extending Rectangle | ESM24 |  |  |  | A1(2023-11-17  15:00:03, 4526), A2(2023-11-21  11:00:00, 4508.5) | -1124 | 
Horizontal Line Non-Extended | ESM24 |  |  |  | A1(2023-07-17  15:00:02, 4508.75), A2(2023-11-17  15:00:03, 4508.75) | -1157 | PBAL
Rectangle | ESM24 |  |  |  | A1(2023-07-26  15:00:03, 4592.5), A2(2023-12-07  15:00:03, 4570) | -1108 | 
Extending Rectangle | ESM24 |  |  |  | A1(2023-12-07  15:00:03, 4577.75), A2(2023-12-11  08:30:00, 4555.5) | -1125 | 
Horizontal Line Non-Extended | ESM24 |  |  |  | A1(2023-08-07  15:00:03, 4557), A2(2023-12-07  15:00:03, 4557) | -1160 | PBAH
Horizontal Ray | ESM24 |  |  |  | A1(2023-12-07  15:00:01, 4592.5) | -1166 | PBAH
Extending Rectangle | ESM24 |  |  |  | A1(2023-12-15  15:00:03, 4778.75), A2(2023-12-18  15:00:06, 4762.25) | -1126 | 
Extending Rectangle | ESM24 |  |  |  | A1(2024-01-05  15:00:03, 4762.5), A2(2024-01-09  08:30:00, 4736) | -1128 | 
Rectangle | ESM24 |  |  |  | A1(2023-12-29  15:00:03, 4834), A2(2024-01-16  15:00:03, 4814) | -1127 | 
Extending Rectangle | ESM24 |  |  |  | A1(2024-01-16  15:00:03, 4818.25), A2(2024-01-18  14:00:00, 4790.25) | -1129 | 
Horizontal Ray | ESM24 |  |  |  | A1(2024-01-16  15:00:01, 4834) | -1167 | PBAH
Extending Rectangle | ESM24 |  |  |  | A1(2024-01-23  15:00:03, 4887.5), A2(2024-01-26  08:30:00, 4877) | -1130 | 
Extending Rectangle | ESM24 |  |  |  | A1(2024-01-29  15:00:03, 4926.5), A2(2024-01-30  10:30:00, 4909.25) | -1131 | 
Rectangle | ESM24 |  |  |  | A1(2024-02-06  15:00:03, 4980), A2(2024-02-21  15:00:03, 4955) | -1132 | 
Extending Rectangle | ESM24 |  |  |  | A1(2024-02-21  15:00:03, 4991.25), A2(2024-02-22  15:00:02, 4972.25) | -1135 | 
Rectangle | ESM24 |  |  |  | A1(2024-02-29  15:00:03, 5100), A2(2024-04-16  15:00:03, 5071.75) | -1137 | 
Rectangle | ESM24 |  |  |  | A1(2024-03-13  15:00:03, 5243.5), A2(2024-03-20  15:00:03, 5227.25) | -1154 | 
Rectangle | ESM24 |  |  |  | A1(2024-03-20  15:00:03, 5244), A2(2024-04-09  15:00:03, 5221.75) | -1155 | 
Extending Rectangle | ESM24 |  |  |  | A1(2024-04-01  15:00:03, 5308.75), A2(2024-04-03  09:00:00, 5275.75) | -1138 | 
Extending Rectangle | ESM24 |  |  |  | A1(2024-04-22  15:00:04, 5037.5), A2(2024-04-23  10:30:00, 5004.75) | -1144 | 
Rectangle | ESM24 |  |  |  | A1(2024-02-19  11:30:03, 5042), A2(2024-04-22  15:00:04, 5018.5) | -1134 | 
Rectangle | ESM24 |  |  |  | A1(2024-02-08  15:00:03, 5016), A2(2024-04-22  15:00:04, 5006.5) | -1133 | 
Extending Rectangle | ESM24 |  |  |  | A1(2024-05-01  08:30:00, 5146), A2(2024-05-02  09:00:00, 5128.75) | -1146 | 
Rectangle | ESM24 |  |  |  | A1(2024-04-24  15:00:03, 5112.75), A2(2024-05-02  15:00:03, 5092.25) | -1145 | 
Rectangle | ESM24 |  |  |  | A1(2024-04-18  15:00:03, 5082.5), A2(2024-05-02  15:00:03, 5045.5) | -1143 | 
Extending Rectangle | ESM24 |  |  |  | A1(2024-05-02  15:00:03, 5094), A2(2024-05-06  15:00:10, 5044.5) | -1147 | 
Horizontal Line Non-Extended | ESM24 |  |  |  | A1(2023-11-06  15:00:03, 4378.75), A2(2023-10-12  11:30:12, 4378.75) | -1163 | PBAL
"""
data_io = StringIO(data)

# Create DataFrame for levels
levels_df = pd.read_csv(data_io, sep="|", engine='python')
levels_df.columns = levels_df.columns.str.strip()
levels_df = levels_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Function to parse levels
def expand_levels_to_dataframe(levels_df):
    new_rows = []

    for index, row in levels_df.iterrows():
        anchors = row['Parsed Anchors']
        if len(anchors) == 1:
            # Handle single anchor points
            new_row = {
                'Start Date': anchors[0][0],
                'End Date': pd.Timestamp('2024-05-9 15:00:00'),  # You can adjust or dynamically determine this end date
                'Level Price': anchors[0][1]
            }
            new_rows.append(new_row)
        elif len(anchors) == 2:
            # Handle two anchors defining a common date range with possibly different prices
            # Define common date range from the earliest start to the latest end
            common_start_date = min(anchors[0][0], anchors[1][0])
            common_end_date = max(anchors[0][0], anchors[1][0])

            # Create an entry for each price point
            for anchor in anchors:
                new_row = {
                    'Start Date': common_start_date,
                    'End Date': common_end_date,
                    'Level Price': anchor[1]
                }
                new_rows.append(new_row)

    return pd.DataFrame(new_rows)

# Function to parse levels
def parse_anchors(anchor_string):
    pattern = re.compile(r"A\d\((\d{4}-\d{2}-\d{2} +\d{2}:\d{2}:\d{2}), +(\d+(?:\.\d+)?)\)")
    matches = pattern.findall(anchor_string)
    return [(pd.to_datetime(dt.strip()), float(price.strip())) for dt, price in matches]

levels_df['Parsed Anchors'] = levels_df['Anchors'].apply(parse_anchors)
new_levels_df = expand_levels_to_dataframe(levels_df)

def check_within_level(bar_datetime, last_price, new_levels_df):
    tick_size = 0.25 # ES Tick size
    max_distance = 12  # Number of ticks a bar needs to be within a level in order to calculate fwd returns

    for idx, level in new_levels_df.iterrows():
        if level['Start Date'] <= bar_datetime <= level['End Date']:
            price_distance = abs((last_price - level['Level Price']) / tick_size)
            # DEBUG
            # print(f"Checking bar at {bar_datetime}, price: {last_price}, level: {level['Level Price']}, distance: {price_distance} ticks")
            if price_distance <= max_distance:
                # DEBUG
                # print("Level match found!")
                return price_distance

def process_session(session_data, new_levels_df):
    session_data.reset_index(drop=True, inplace=True)
    
    num_forward_bars = [5, 10, 20, 30, 40, 50, 75, 100]  # Define forward return bins
    session_data['Level Offset Ticks'] = np.nan
    session_data['MFE'] = np.nan  # Initialize new column for MFE
    
    for n in num_forward_bars:
        session_data[f'diff_{n}'] = np.nan  # Initialize new columns for storing forward returns
        session_data[f'stop_diff_{n}'] = np.nan  # Initialize new columns for stop loss checks
    
    # Populate Level Offset Ticks rows if a bar 
    for index, row in session_data.iterrows():
        level_offset_ticks = check_within_level(row['DateTime'], row['Last'], new_levels_df)
        session_data.at[index, 'Level Offset Ticks'] = level_offset_ticks
        
        
        if (level_offset_ticks is not None) and ((row['Inversion'] == 1) | (row['Exhaustion'] == 1)): # Only calculate fwd returns if the level offset is within max_distance and the bar has Inversion

            for n in num_forward_bars: # Loops for each fwd return bin
                forward_idx = index + n # index for n_th bar ahead
                
                if forward_idx < len(session_data): # Make sure dataset has enough rows left
                    # Check if the 'Last' price at the forward index is not NaN
                    if pd.notna(session_data.at[forward_idx, 'Last']):
                        future_price = session_data.at[forward_idx, 'Last']
                        stopped_out = False
                        
                        if row['Buy INV'] == 1 or row['Buy EX'] == 1: # Calculate stuff if it's a buy INV bar
                            stop_loss_price = row['Low'] - 0.25
                            min_low = session_data.loc[index+1:forward_idx, 'Low'].min() # Calculate potential MAE
                            max_high = session_data.loc[index+1:forward_idx, 'High'].max()  # Calculate potential MFE
                            
                            if min_low <= stop_loss_price: # Check if stop loss has been hit
                                session_data.at[index, f'diff_{n}'] = 4 * (stop_loss_price - row['Last'])
                                session_data.at[index, f'stop_diff_{n}'] = 1 # Populate stopped out column
                                stopped_out = True
                            else:
                                session_data.at[index, f'diff_{n}'] = 4 * (future_price - row['Last'])
                                session_data.at[index, f'stop_diff_{n}'] = 0
                            
                            # Update MFE only if not stopped out
                            if not stopped_out:
                                current_mfe = 4 * (max_high - row['Last'])
                                if pd.isna(session_data.at[index, 'MFE']) or current_mfe > session_data.at[index, 'MFE']:
                                    session_data.at[index, 'MFE'] = current_mfe
                            
                        elif row['Sell INV'] == 1 or row['Sell EX'] == 1:  # Calculate stuff if it's a sell INV bar
                            stop_loss_price = row['High'] + 0.25
                            max_high = session_data.loc[index+1:forward_idx, 'High'].max() # Calculate potential MAE
                            min_low = session_data.loc[index+1:forward_idx, 'Low'].min()  # Calculate potential MFE
                            
                            if max_high >= stop_loss_price:
                                session_data.at[index, f'diff_{n}'] = 4 * (row['Last'] - stop_loss_price)
                                session_data.at[index, f'stop_diff_{n}'] = 1
                                stopped_out = True
                            else:
                                session_data.at[index, f'diff_{n}'] = 4 * (row['Last'] - future_price)
                                session_data.at[index, f'stop_diff_{n}'] = 0
                            
                            # Update MFE only if not stopped out
                            if not stopped_out:
                                current_mfe = 4 * (row['Last'] - min_low)
                                if pd.isna(session_data.at[index, 'MFE']) or current_mfe > session_data.at[index, 'MFE']:
                                    session_data.at[index, 'MFE'] = current_mfe
                            
                    else:
                        session_data.at[index, f'diff_{n}'] = np.nan
                        session_data.at[index, f'stop_diff_{n}'] = np.nan
                else:
                    session_data.at[index, f'diff_{n}'] = np.nan
                    session_data.at[index, f'stop_diff_{n}'] = np.nan

    return session_data, session_data['Session'].iloc[0]

def main():
    # Load 8 tick data
    data_8_tick = pd.read_csv('C:\\SierraChart\\SierraChartInstance_2\\Data\\ES_8tick_250D.csv')
    data_8_tick.columns = data_8_tick.columns.str.strip()
    data_8_tick['DateTime'] = pd.to_datetime(data_8_tick['Date'] + ' ' + data_8_tick['Time'])
    
    # Define session boundaries
    session_start = pd.to_datetime('08:30:00').time()
    
    # Add session column to 8 tick data
    data_8_tick['Session'] = data_8_tick['DateTime'].apply(lambda dt: dt.date() if dt.time() >= session_start else (dt - pd.Timedelta(days=1)).date())

    data_8_tick['Range'] = data_8_tick['High'] - data_8_tick['Low']
    
    # Identify inversion signals in 8 tick data
    inversion_signals = ['Sell INV Weak', 'Sell INV Strong', 'Buy INV Weak', 'Buy INV Strong']
    for signal in inversion_signals:
        data_8_tick[signal] = (data_8_tick[signal] > 0).astype(int)
    
    # Create and populate total Buy and Sell INV columns
    data_8_tick['Buy INV'] = np.where((data_8_tick['Buy INV Weak'] == 1) | (data_8_tick['Buy INV Strong'] == 1), 1, 0)
    data_8_tick['Sell INV'] = np.where((data_8_tick['Sell INV Weak'] == 1) | (data_8_tick['Sell INV Strong'] == 1), 1, 0)
    
    # If there is a strong invserion column, make the strong column true and and weak column false
    data_8_tick.loc[data_8_tick['Buy INV Strong'] == 1, 'Buy INV Weak'] = 0
    data_8_tick.loc[data_8_tick['Sell INV Strong'] == 1, 'Sell INV Weak'] = 0
    
    # Create and populate Inversion column
    data_8_tick['Inversion'] = ((data_8_tick['Buy INV'] == 1) | (data_8_tick['Sell INV'] == 1)).astype(int) 
   
    # Create and populate Strong and Weak inversion columns
    data_8_tick['Strong Inversion'] = ((data_8_tick['Buy INV Strong' ] == 1) | (data_8_tick['Sell INV Strong'] == 1)).astype(int) 
    data_8_tick['Weak Inversion'] = ((data_8_tick['Buy INV Weak'] == 1) | (data_8_tick['Sell INV Weak'] == 1)).astype(int) 
    
    # Identify exhaustion signals in 8 tick data
    exhaustion_signals = ['2x Sell EX', 'Accel Sell EX', '2x Buy Ex', 'Accel Buy EX']
    for signal in exhaustion_signals:
        data_8_tick[signal] = (data_8_tick[signal] > 0).astype(int)
    
    # Create and populate total Buy and Sell EX columns
    data_8_tick['Buy EX'] = np.where((data_8_tick['2x Buy Ex'] == 1) | (data_8_tick['Accel Buy EX'] == 1), 1, 0)
    data_8_tick['Sell EX'] = np.where((data_8_tick['2x Sell EX'] == 1) | (data_8_tick['Accel Sell EX'] == 1), 1, 0)

    # If there is a strong EX column, make the strong column true and and weak column false
    data_8_tick.loc[data_8_tick['Accel Buy EX'] == 1, '2x Buy Ex'] = 0
    data_8_tick.loc[data_8_tick['Accel Sell EX'] == 1, '2x Sell EX'] = 0
    
    # Rename columns
    data_8_tick.rename(columns={'Accel Buy EX': 'Buy EX Strong', '2x Buy Ex': 'Buy EX Weak', 'Accel Sell EX': 'Sell EX Strong', '2x Sell EX': 'Sell EX Weak'}, inplace=True)
    
    # Create and populate Exhaustion column
    data_8_tick['Exhaustion'] = ((data_8_tick['Buy EX'] == 1) | (data_8_tick['Sell EX'] == 1)).astype(int) 
    
    # Create and populate Strong and Weak exhaustion columns
    data_8_tick['Strong EX'] = ((data_8_tick['Buy EX Strong' ] == 1) | (data_8_tick['Sell EX Strong'] == 1)).astype(int) 
    data_8_tick['Weak EX'] = ((data_8_tick['Buy EX Weak'] == 1) | (data_8_tick['Sell EX Weak'] == 1)).astype(int) 
    
    # Process each session in parallel for faster compute times
    with Pool() as pool:
       results = pool.starmap(process_session, [(data_8_tick[data_8_tick['Session'] == session], new_levels_df) for session in data_8_tick['Session'].unique()])
       print(f"Total sessions to process: {len(results)}")
    
    # Separate the session DataFrames and session dates
    session_dfs, session_dates = zip(*results)
    
    # Concatenate the session DataFrames and reset the index
    data_8_tick = pd.concat(session_dfs).reset_index(drop=True)
    
    # Sort the combined DataFrame by the session date and the original index
    data_8_tick['OriginalIndex'] = data_8_tick.index
    data_8_tick['SessionDate'] = pd.to_datetime(data_8_tick['Session'])
    data_8_tick.sort_values(['SessionDate', 'OriginalIndex'], inplace=True)
    data_8_tick.drop(['OriginalIndex', 'SessionDate'], axis=1, inplace=True)
    
    # Write processed data to CSV file
    output_csv = '8_tick_inv_and_ex.csv'
    data_8_tick.to_csv(output_csv, index=False)
    
    # DEBUG
    #print(data_8_tick['Session'].unique())
    
    print(f'Successfully written to {output_csv}')

if __name__ == "__main__":
    main()
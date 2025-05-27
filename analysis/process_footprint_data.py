import pandas as pd
import numpy as np
from multiprocessing import Pool
from io import StringIO
import re
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class TradingConfig:
    """Configuration constants for trading analysis."""
    TICK_SIZE = 0.25
    MAX_LEVEL_DISTANCE = 12  # ticks
    SESSION_START_TIME = pd.to_datetime('08:30:00').time()
    FORWARD_RETURN_PERIODS = [5, 10, 20, 30, 40, 50, 75, 100]
    CONTRACT_MULTIPLIER = 4  # ES contract tick value multiplier


class LevelProcessor:
    """Handles support/resistance level processing and validation."""

    def __init__(self, levels_data: str):
        self.levels_df = self._parse_levels_data(levels_data)

    def _parse_levels_data(self, data: str) -> pd.DataFrame:
        """Parse Sierra Chart level export data."""
        data_io = StringIO(data)
        df = pd.read_csv(data_io, sep="|", engine='python')
        df.columns = df.columns.str.strip()
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        df['Parsed Anchors'] = df['Anchors'].apply(self._parse_anchors)
        return self._expand_levels_to_dataframe(df)

    def _parse_anchors(self, anchor_string: str) -> List[Tuple]:
        """Extract datetime and price from anchor string."""
        pattern = re.compile(r"A\d\((\d{4}-\d{2}-\d{2} +\d{2}:\d{2}:\d{2}), +(\d+(?:\.\d+)?)\)")
        matches = pattern.findall(anchor_string)
        return [(pd.to_datetime(dt.strip()), float(price.strip())) for dt, price in matches]

    def _expand_levels_to_dataframe(self, levels_df: pd.DataFrame) -> pd.DataFrame:
        """Convert anchor points to level ranges."""
        new_rows = []
        max_date = pd.Timestamp('2024-05-9 15:00:00')

        for _, row in levels_df.iterrows():
            anchors = row['Parsed Anchors']

            if len(anchors) == 1:
                new_rows.append({
                    'Start Date': anchors[0][0],
                    'End Date': max_date,
                    'Level Price': anchors[0][1]
                })
            elif len(anchors) == 2:
                common_start = min(anchors[0][0], anchors[1][0])
                common_end = max(anchors[0][0], anchors[1][0])

                for anchor in anchors:
                    new_rows.append({
                        'Start Date': common_start,
                        'End Date': common_end,
                        'Level Price': anchor[1]
                    })

        return pd.DataFrame(new_rows)

    def get_level_distance(self, bar_datetime: pd.Timestamp, price: float) -> Optional[float]:
        """Calculate distance from price to nearest level in ticks."""
        for _, level in self.levels_df.iterrows():
            if level['Start Date'] <= bar_datetime <= level['End Date']:
                distance = abs((price - level['Level Price']) / TradingConfig.TICK_SIZE)
                if distance <= TradingConfig.MAX_LEVEL_DISTANCE:
                    return distance
        return None


class SignalProcessor:
    """Handles trading signal processing and classification."""

    @staticmethod
    def process_signals(data: pd.DataFrame) -> pd.DataFrame:
        """Process and classify inversion and exhaustion signals."""
        # Process inversion signals
        inversion_signals = ['Sell INV Weak', 'Sell INV Strong', 'Buy INV Weak', 'Buy INV Strong']
        for signal in inversion_signals:
            data[signal] = (data[signal] > 0).astype(int)

        # Create buy/sell inversion columns
        data['Buy INV'] = ((data['Buy INV Weak'] == 1) | (data['Buy INV Strong'] == 1)).astype(int)
        data['Sell INV'] = ((data['Sell INV Weak'] == 1) | (data['Sell INV Strong'] == 1)).astype(int)

        # Handle strong signal priority
        data.loc[data['Buy INV Strong'] == 1, 'Buy INV Weak'] = 0
        data.loc[data['Sell INV Strong'] == 1, 'Sell INV Weak'] = 0

        # Create aggregate columns
        data['Inversion'] = ((data['Buy INV'] == 1) | (data['Sell INV'] == 1)).astype(int)
        data['Strong Inversion'] = ((data['Buy INV Strong'] == 1) | (data['Sell INV Strong'] == 1)).astype(int)
        data['Weak Inversion'] = ((data['Buy INV Weak'] == 1) | (data['Sell INV Weak'] == 1)).astype(int)

        # Process exhaustion signals
        exhaustion_signals = ['2x Sell EX', 'Accel Sell EX', '2x Buy Ex', 'Accel Buy EX']
        for signal in exhaustion_signals:
            data[signal] = (data[signal] > 0).astype(int)

        # Create buy/sell exhaustion columns
        data['Buy EX'] = ((data['2x Buy Ex'] == 1) | (data['Accel Buy EX'] == 1)).astype(int)
        data['Sell EX'] = ((data['2x Sell EX'] == 1) | (data['Accel Sell EX'] == 1)).astype(int)

        # Handle strong exhaustion priority
        data.loc[data['Accel Buy EX'] == 1, '2x Buy Ex'] = 0
        data.loc[data['Accel Sell EX'] == 1, '2x Sell EX'] = 0

        # Rename columns for consistency
        data.rename(columns={
            'Accel Buy EX': 'Buy EX Strong',
            '2x Buy Ex': 'Buy EX Weak',
            'Accel Sell EX': 'Sell EX Strong',
            '2x Sell EX': 'Sell EX Weak'
        }, inplace=True)

        # Create aggregate exhaustion columns
        data['Exhaustion'] = ((data['Buy EX'] == 1) | (data['Sell EX'] == 1)).astype(int)
        data['Strong EX'] = ((data['Buy EX Strong'] == 1) | (data['Sell EX Strong'] == 1)).astype(int)
        data['Weak EX'] = ((data['Buy EX Weak'] == 1) | (data['Sell EX Weak'] == 1)).astype(int)

        return data


class ForwardReturnCalculator:
    """Calculates forward returns with stop-loss and MFE tracking."""

    def __init__(self, config: TradingConfig):
        self.config = config

    def calculate_returns(self, session_data: pd.DataFrame, level_processor: LevelProcessor) -> pd.DataFrame:
        """Calculate forward returns for all signal bars in session."""
        session_data = session_data.reset_index(drop=True)

        # Initialize columns
        session_data['Level Offset Ticks'] = np.nan
        session_data['MFE'] = np.nan

        for period in self.config.FORWARD_RETURN_PERIODS:
            session_data[f'diff_{period}'] = np.nan
            session_data[f'stop_diff_{period}'] = np.nan

        # Process each bar
        for index, row in session_data.iterrows():
            level_distance = level_processor.get_level_distance(row['DateTime'], row['Last'])
            session_data.at[index, 'Level Offset Ticks'] = level_distance

            if self._should_calculate_returns(row, level_distance):
                self._calculate_bar_returns(session_data, index, row)

        return session_data

    def _should_calculate_returns(self, row: pd.Series, level_distance: Optional[float]) -> bool:
        """Determine if forward returns should be calculated for this bar."""
        return (level_distance is not None and
                (row['Inversion'] == 1 or row['Exhaustion'] == 1))

    def _calculate_bar_returns(self, data: pd.DataFrame, index: int, row: pd.Series) -> None:
        """Calculate forward returns for a single bar."""
        for period in self.config.FORWARD_RETURN_PERIODS:
            forward_idx = index + period

            if forward_idx >= len(data) or pd.isna(data.at[forward_idx, 'Last']):
                continue

            future_price = data.at[forward_idx, 'Last']

            if row['Buy INV'] == 1 or row['Buy EX'] == 1:
                self._process_long_signal(data, index, forward_idx, row, future_price, period)
            elif row['Sell INV'] == 1 or row['Sell EX'] == 1:
                self._process_short_signal(data, index, forward_idx, row, future_price, period)

    def _process_long_signal(self, data: pd.DataFrame, index: int, forward_idx: int,
                             row: pd.Series, future_price: float, period: int) -> None:
        """Process long signal with stop-loss and MFE calculation."""
        stop_price = row['Low'] - self.config.TICK_SIZE
        min_low = data.loc[index + 1:forward_idx, 'Low'].min()
        max_high = data.loc[index + 1:forward_idx, 'High'].max()

        if min_low <= stop_price:
            data.at[index, f'diff_{period}'] = self.config.CONTRACT_MULTIPLIER * (stop_price - row['Last'])
            data.at[index, f'stop_diff_{period}'] = 1
        else:
            data.at[index, f'diff_{period}'] = self.config.CONTRACT_MULTIPLIER * (future_price - row['Last'])
            data.at[index, f'stop_diff_{period}'] = 0

            # Update MFE
            current_mfe = self.config.CONTRACT_MULTIPLIER * (max_high - row['Last'])
            if pd.isna(data.at[index, 'MFE']) or current_mfe > data.at[index, 'MFE']:
                data.at[index, 'MFE'] = current_mfe

    def _process_short_signal(self, data: pd.DataFrame, index: int, forward_idx: int,
                              row: pd.Series, future_price: float, period: int) -> None:
        """Process short signal with stop-loss and MFE calculation."""
        stop_price = row['High'] + self.config.TICK_SIZE
        max_high = data.loc[index + 1:forward_idx, 'High'].max()
        min_low = data.loc[index + 1:forward_idx, 'Low'].min()

        if max_high >= stop_price:
            data.at[index, f'diff_{period}'] = self.config.CONTRACT_MULTIPLIER * (row['Last'] - stop_price)
            data.at[index, f'stop_diff_{period}'] = 1
        else:
            data.at[index, f'diff_{period}'] = self.config.CONTRACT_MULTIPLIER * (row['Last'] - future_price)
            data.at[index, f'stop_diff_{period}'] = 0

            # Update MFE
            current_mfe = self.config.CONTRACT_MULTIPLIER * (row['Last'] - min_low)
            if pd.isna(data.at[index, 'MFE']) or current_mfe > data.at[index, 'MFE']:
                data.at[index, 'MFE'] = current_mfe


class DataProcessor:
    """Main data processing pipeline coordinator."""

    def __init__(self, levels_data: str):
        self.config = TradingConfig()
        self.level_processor = LevelProcessor(levels_data)
        self.signal_processor = SignalProcessor()
        self.return_calculator = ForwardReturnCalculator(self.config)

    def load_and_prepare_data(self, file_path: str) -> pd.DataFrame:
        """Load and prepare market data."""
        data = pd.read_csv(file_path)
        data.columns = data.columns.str.strip()
        data['DateTime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
        data['Session'] = data['DateTime'].apply(self._get_session_date)
        data['Range'] = data['High'] - data['Low']

        return self.signal_processor.process_signals(data)

    def _get_session_date(self, dt: pd.Timestamp) -> pd.Timestamp:
        """Determine session date based on time."""
        if dt.time() >= self.config.SESSION_START_TIME:
            return dt.date()
        return (dt - pd.Timedelta(days=1)).date()

    def process_session(self, session_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Timestamp]:
        """Process a single trading session."""
        processed_data = self.return_calculator.calculate_returns(session_data, self.level_processor)
        return processed_data, session_data['Session'].iloc[0]

    def process_all_sessions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process all sessions using multiprocessing."""
        sessions = data['Session'].unique()

        with Pool() as pool:
            results = pool.starmap(
                self.process_session,
                [(data[data['Session'] == session],) for session in sessions]
            )

        print(f"Processed {len(results)} sessions")

        # Combine results
        session_dfs, _ = zip(*results)
        combined_data = pd.concat(session_dfs).reset_index(drop=True)

        # Sort by session and time
        combined_data['OriginalIndex'] = combined_data.index
        combined_data['SessionDate'] = pd.to_datetime(combined_data['Session'])
        combined_data.sort_values(['SessionDate', 'OriginalIndex'], inplace=True)
        combined_data.drop(['OriginalIndex', 'SessionDate'], axis=1, inplace=True)

        return combined_data


def main():
    """Main execution function."""
    # Sierra Chart levels data
    LEVELS_DATA = """
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

    # Initialize processor
    processor = DataProcessor(LEVELS_DATA)

    # Load and process data
    input_file = 'ES_8tick_250D.csv'
    data = processor.load_and_prepare_data(input_file)

    # Process all sessions
    processed_data = processor.process_all_sessions(data)

    # Save output
    output_file = '8_tick_inv_and_ex.csv'
    processed_data.to_csv(output_file, index=False)

    print(f'Successfully processed data and saved to {output_file}')


if __name__ == "__main__":
    main()
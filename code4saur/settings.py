data_settings = {
    'ETTh1': {'data': 'ETTh1.csv', 'T': 'OT', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1]},
    'ETTh2': {'data': 'ETTh2.csv', 'T': 'OT', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1]},
    'ETTm1': {'data': 'ETTm1.csv', 'T': 'OT', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1]},
    'ETTm2': {'data': 'ETTm2.csv', 'T': 'OT', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1]},
    'Weather': {'data': 'weather.csv', 'T': 'OT', 'M': [21, 21], 'S': [1, 1], 'MS': [21, 1]},
    'Exchange': {'data': 'exchange_rate.csv', 'T': 'OT', 'M': [8, 8], 'S': [1, 1], 'MS': [8, 1]},
    'Traffic': {'data': 'traffic.csv', 'T': 'OT', 'M': [862, 862], 'S': [1, 1], 'MS': [862, 1]},
    'Solar': {'data': 'solar_AL_dates.csv', 'T': 136, 'M': [137, 137], 'S': [1, 1], 'MS': [137, 1]},
    'Stock': {'data': 'stock.csv', 'T': 'DailyPrice', 'M': [8, 8], 'S': [1, 1], 'MS': [8, 1]},
    'Crime': {'data': 'CHI_Crime_Cleaned_Hourly.csv', 'T': 'THEFT', 'M': [4, 4], 'S': [1, 1], 'MS': [4, 1]},
}

def get_borders(args):


    args.borders = None
    args.ratio = (0.2, 0.75)

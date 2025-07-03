import os
import pandas as pd

input_folder = '../data/Training Data'
output_folder = '../data/Training Data/sample'

sample_size = 100

os.makedirs(output_folder, exist_ok=True)

# Define the allowed environments
allowed_environments = [
    'DEH1_2019', 'TXH2_2019', 'NCH1_2019', 'SCH1_2019', 'IAH3_2019', 'MNH1_2019', 'IAH2_2019', 'TXH3_2019', 'NYH3_2019', 'ILH1_2019',
    'WIH1_2019', 'GAH1_2019', 'WIH2_2019', 'TXH1_2019', 'IAH4_2019', 'MIH1_2019', 'INH1_2019', 'GEH1_2019', 'IAH1_2019', 'NYH2_2019',
    'GAH2_2019', 'NEH2_2019', 'NEH1_2019',
    'DEH1_2020', 'GAH1_2020', 'GAH2_2020', 'GEH1_2020', 'IAH1_2020', 'INH1_2020', 'MIH1_2020', 'MNH1_2020', 'NCH1_2020', 'NEH1_2020', 'NEH2_2020',
    'NEH3_2020', 'NYH2_2020', 'NYH3_2020', 'NYS1_2020', 'SCH1_2020','TXH1_2020', 'TXH2_2020', 'TXH3_2020', 'WIH1_2020', 'WIH2_2020', 'WIH3_2020',
    'COH1_2021', 'DEH1_2021', 'GAH1_2021', 'GAH2_2021', 'GEH1_2021', 'IAH1_2021', 'IAH2_2021', 'IAH3_2021', 'IAH4_2021', 'ILH1_2021', 'INH1_2021', 'MIH1_2021',
    'MNH1_2021', 'NCH1_2021', 'NEH1_2021', 'NEH2_2021', 'NEH3_2021', 'NYH2_2021', 'NYH3_2021', 'NYS1_2021', 'SCH1_2021', 'TXH1_2021', 'TXH2_2021', 'TXH3_2021',
    'WIH1_2021', 'WIH2_2021', 'WIH3_2021'
]


for filename in os.listdir(input_folder):
    if filename.endswith('csv'):
        # Extract environment name from filename (assuming format like 'ENV_NAME.csv')
        env_name = filename.replace('.csv', '')

        if env_name in allowed_environments:
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            df = pd.read_csv(input_path, encoding='ISO-8859-1')

            # Remove rows with any NA values
            df_no_na = df.dropna()

            # Sample from the DataFrame without NA values
            sampled_df = df_no_na.sample(n=min(sample_size, len(df_no_na)))

            sampled_df.to_csv(output_path, index=False)
            print(f'{filename} Finished.')
        else:
            print(f"Skipping {filename}: Environment not in allowed list.")
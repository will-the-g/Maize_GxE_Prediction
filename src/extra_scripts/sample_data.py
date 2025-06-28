import os
import pandas as pd

input_folder = 'data/Training Data'
output_folder = 'data/Training Data/sample'

sample_size = 100

os.makedirs(output_folder, exist_ok=True)


for filename in os.listdir(input_folder):
    if filename.endswith('csv'):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        df = pd.read_csv(input_path, encoding='ISO-8859-1')
        sampled_df = df.sample(n=min(sample_size, len(df)))

        sampled_df.to_csv(output_path, index=False)
        print(f'{filename} Finished.')
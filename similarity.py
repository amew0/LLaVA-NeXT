import pandas as pd
import numpy as np

def calculate_loss_statistics(csv_file_path):
    df = pd.read_csv(csv_file_path)
    losses = df['loss'].to_numpy()
    average_loss = np.mean(losses)
    std_loss = np.std(losses)
    return average_loss, std_loss

av, std = calculate_loss_statistics("out/results_test_wz.json_llava-onevision-qwen2-0.5b-ov.csv")
print(f'Average Loss, Standard Deviation {av:.4f}, {std:.4f}')

import pandas as pd
import numpy as np

def calculate_loss_statistics(file_path):
    if file_path.endswith('.csv'):   
        df = pd.read_csv(file_path) 
        losses = df['loss'].to_numpy()
    else:
        import json
        with open(file_path) as f:
            data = json.load(f)
        losses = [d['loss'] for d in data]
    average_loss = np.mean(losses)
    std_loss = np.std(losses)
    return average_loss, std_loss

if __name__ == "__main__":
    import sys
    file_path = sys.argv[1]
    av, std = calculate_loss_statistics(file_path)
    print(f'Average Loss, Standard Deviation {av:.6f}, {std:.6f}')

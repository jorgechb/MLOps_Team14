import logging
import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

class DataLoaderP:
    def __init__(self) -> None:
        pass

    def fetch_dataset(self, save_df=False): 
        print("Fetching dataset...")
        mushroom_dataset = fetch_ucirepo(id=848)
        X = mushroom_dataset.data.features
        y = mushroom_dataset.data.targets

        data = pd.concat([X, y], axis=1)
        print("Data ready")

        # Get the current working directory (project path)
        project_path = os.getcwd()

        # Print the project path
        print(f"Project path: {project_path}")

        if save_df: 
            print("Saving dataset...")
            # Construct the directory path
            raw_data_dir = os.path.join(project_path, "data", "raw")
            # Ensure the directory exists
            os.makedirs(raw_data_dir, exist_ok=True)
            # Construct the file path
            raw_data_path = os.path.join(raw_data_dir, "raw_dataset.csv")
            # Save the CSV file
            data.to_csv(raw_data_path, index=False)
            print(f"Dataset saved to {raw_data_path}")
        else: 
            return data

if __name__ == "__main__": 
    loader = DataLoaderP()
    loader.fetch_dataset(save_df=True)
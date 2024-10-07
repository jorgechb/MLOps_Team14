from ucimlrepo import fetch_ucirepo
import pandas as pd
import logging


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

        if save_df: 
            print("Saving dataset...")
            data.to_csv(r"..\data\raw\raw_dataset2.csv", index=False)
        else: 
            return data


if __name__ == "__main__": 
    
    loader = DataLoaderP()
    loader.fetch_dataset(save_df=True)



import argparse
import os
from kaggle.api.kaggle_api_extended import KaggleApi


def create_folders(folder_list):
    for i in folder_list:
        if not os.path.exists(i):
            os.makedirs(i)
            print(f"folder is created: {i}")
        else:
            print(f"folder already exists: {i}")


def dataset_downloader(dataset,download_path):
    #to authenticate https://www.kaggle.com/docs/api
    api = KaggleApi()
    api.authenticate()  
    print(f"starting download of {dataset}")
    api.dataset_download_files(dataset, path=download_path, unzip=True)



if __name__ =="__main__":
    DIR_RAW = "data/raw"
    DIR_CLEAN = "data/clean"
    folder= [DIR_RAW,DIR_CLEAN]
    create_folders(folder)

    parser = argparse.ArgumentParser(description='data_downloader')

    parser.add_argument('--data_set',default="dgomonov/new-york-city-airbnb-open-data",
                        help='Kaggle dataset download. defaults to "dgomonov/new-york-city-airbnb-open-data"' , type=str)

    args = parser.parse_args()
    params = vars(args)
    dataset=params["data_set"]
    
    dataset_downloader(dataset,DIR_RAW)
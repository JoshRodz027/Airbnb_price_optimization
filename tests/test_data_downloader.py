import pytest
import os
from typing import List
from src.data_downloader import create_folders ,dataset_downloader


@pytest.mark.parametrize("folder_list",[["data/raw","data/clean"]])
def test_create_folders_positive(folder_list:List[str],capfd):
    create_folders(folder_list)
    out1, _ = capfd.readouterr()
    assert (
        "folder already exists" in out1
    )

@pytest.mark.parametrize("folder_list",[["testing/raw","testing/clean"]])
def test_create_folders_negative(folder_list:List[str],capfd,deleted_testing_folder):
    create_folders(folder_list)
    out1, _ = capfd.readouterr()
    assert (
        "folder is created" in out1
    )
    deleted_testing_folder

@pytest.mark.parametrize("data_set",["dgomonov/new-york-city-airbnb-open-data"])
@pytest.mark.parametrize("download_path",["testing/raw"])
def test_dataset_downloader(data_set,download_path,capfd):
    dataset_downloader(data_set,download_path)

    assert (os.path.isfile(r"testing\raw\AB_NYC_2019.csv"))
    assert (os.path.isfile(r"testing\raw\New_York_City_.png"))
    out1, _ = capfd.readouterr()
    assert (
        "starting download of" in out1
    )
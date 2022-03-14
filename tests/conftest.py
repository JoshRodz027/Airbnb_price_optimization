import os
from re import L
import sys
from pathlib import Path
from zipfile import ZipFile
import pytest
import shutil
import yaml

with open(r"C:\Users\rodzi\Documents\My projects\Dyson\Dyson_test\config.yaml") as file:
        params_config = yaml.load(file, Loader=yaml.FullLoader)

delete_test_folder= params_config["delete_test_folder"]

sys.path.append(os.getcwd())

@pytest.fixture(scope="session", autouse=True)
def deleted_testing_folder():
    yield 
    if delete_test_folder == True:
        shutil.rmtree('testing')


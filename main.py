from src.data_downloader import dataset_downloader,create_folders
from src.data_preprocessing import DataPipeline
from src.model import ModelEvaluator
from src.model import Model

import yaml

pl = DataPipeline()
evaluator = ModelEvaluator()
model = Model()

if __name__=="__main__":

    with open(r"C:\Users\rodzi\Documents\My projects\Dyson\Dyson_test\config.yaml") as file:
        params_config = yaml.load(file, Loader=yaml.FullLoader)
    
    print("Unpacking params from config.yaml please wait...")
    use_default_params = params_config["use_default_params"]
    if use_default_params:
        print("Using default model Params")
        model_params = params_config["default_params"]
    else:
        model_params = params_config["params"]
        min_feature = model_params["min_feature"]
        print(f"min_feature for feature_selection using custom params: {min_feature}")
    

    print(f"Model Params: {model_params}")
    
    DIR_RAW = params_config["DIR_RAW"]
    print(f"Raw data directory: {DIR_RAW}")
    DIR_CLEAN = params_config["DIR_CLEAN"]
    print(f"Clean data directory: {DIR_CLEAN}")
    data_path = params_config["data_path"]
    print(f"Data paths to read from {data_path}")
    data_set = params_config["data_set"]
    print(f"Data set to download from {data_set}")
    columns_to_skew = params_config["columns_to_skew"]
    print(f"Columns to skew for data pre-processing {columns_to_skew}")
    test_size = params_config["test_size"]
    print(f"Test size for train test split: {test_size}") 
    to_download = params_config["download_data_set"]
    print(f"Download data set from kaggle set to {to_download}")
    run_pipeline = params_config["run_pipeline"]
    print(f"Run full pipeline set to {run_pipeline}")

    if to_download:
        folder = [DIR_RAW,DIR_CLEAN]
        create_folders(folder)
        dataset_downloader(data_set,folder[0])
        print("Download Completed")

    if run_pipeline:
        print("Starting dataframe ingestion and preprocessing pipline")
        df = pl.full_cleanup(data_path=data_path,columns=columns_to_skew)
        
        print("Starting train_test split")
        train_data, test_data = pl.prepare_train_test(df,test_size)
        x_train, y_train=pl.transform_train_data(train_data)
        x_test, y_test=pl.transform_test_data(test_data)

        print("Starting Model training pipline")
        model.train(model_params,x_train, y_train,min_feature=min_feature)

        print("Starting Model inference pipline")
        model.evaluate(x_test, y_test)

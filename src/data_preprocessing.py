#imports
from lib2to3.pgen2.pgen import DFAState
from typing import List, Tuple
import pandas as pd
from pandas import DataFrame
import numpy as np
import re
import datetime as dt

#Model
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

class DataPipeline():

    def __init__(self):
        self.pl = None
        self.tc =None

    @staticmethod
    def _basic_cleanup(data_path:str)->DataFrame:
        df = pd.read_csv(data_path)

        # cleaning by dropping unused columns
        df.drop_duplicates(inplace=True)
        df.drop(['name','id','host_name','last_review',"host_id"], axis=1, inplace=True)
        df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
          #Clean str columns
        df[df.select_dtypes(['object']).columns] = df.select_dtypes(['object']).applymap(lambda x: x.lower().replace(" ", "_"))

        return df

    @staticmethod
    def _feature_engineering(df:DataFrame)->DataFrame:
        df['all_year_avail'] = df['availability_365']>353
        df['low_avail'] = df['availability_365']< 12
        df['no_reviews'] = df['reviews_per_month']==0
        df.replace({False: 0, True: 1}, inplace=True)
        return df

    @staticmethod
    def _skew_adjustment(df:DataFrame,columns:List[str])->DataFrame:
        # To reverse this use - back = np.expm1(Y)
        for column in columns:
            df[column] = np.log1p(df[column])
        return df


    def full_cleanup(self,data_path:str,columns:List[str],save:bool=True) -> DataFrame:
        df = self._basic_cleanup(data_path)
        df = self._feature_engineering(df)
        df_clean = self._skew_adjustment(df,columns)
        if save:
            df_clean.to_csv("data/clean/full_clean.csv")
        return df_clean

    def _pre_process_train(self,X_train:DataFrame)->DataFrame:
        numeric_features = ["latitude","longitude","availability_365","calculated_host_listings_count","minimum_nights","number_of_reviews","reviews_per_month"]
        ordinal_features = ['room_type']
        nominal_features = [ "neighbourhood_group","neighbourhood"]

        numeric_transformer = Pipeline(
            [
                ('imputer_num', SimpleImputer(strategy = 'median')),
                ('scaler', StandardScaler())
            ]
        )


        ordinal_transformer = Pipeline(steps=[
            ("encoder", OrdinalEncoder())])

        nominal_transformer = Pipeline(
            [
                ('imputer_cat', SimpleImputer(strategy = 'constant',
                  fill_value = 'missing')),
                ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
            ])        
        
        preprocessor = ColumnTransformer(
            transformers=[("numeric", numeric_transformer, numeric_features),
                          ("nominal", nominal_transformer, nominal_features),
                          ("ordinal", ordinal_transformer, ordinal_features)])
        
        pipeline = Pipeline(
        [
            ('preprocessing', preprocessor)

        ]
        )
        

        self.pl = pipeline.fit(X_train)
        X_train = self.pl.transform(X_train).todense()
        
        
        self.tc = (numeric_features + list(preprocessor.named_transformers_['nominal'].named_steps['onehot'].get_feature_names(nominal_features)) +
                            ordinal_features)

        X_train = pd.DataFrame(X_train, columns = self.tc)
        
        
        return X_train

    def _pre_process_test(self,X_test):
    

        X_test = self.pl.transform(X_test).todense()
        X_test = pd.DataFrame(X_test, columns = self.tc)
       
        return X_test
    
    def transform_train_data(self,train_df:DataFrame=None,train_df_path:str=None):
        if train_df_path:
             train_df = pd.read_csv(train_df_path)

        X_train = train_df.drop(["price"], axis=1)
        y_train = train_df['price']

        # transform data
        X_train = self._pre_process_train(X_train)
        
        return X_train, y_train
    

    def transform_test_data(self,test_df:DataFrame=None,test_df_path:str=None):
        if test_df_path:
             test_df = pd.read_csv(test_df_path)
        X_test = test_df.drop(["price"], axis=1)
        y_test = test_df['price']

        # transform data
        X_test = self._pre_process_test(X_test)
      
        return X_test, y_test

    
    def prepare_train_test(self,df:DataFrame,test_size=0.2,random_state=123,shuffle=True,save=True)-> Tuple[DataFrame,DataFrame]:
        train_data, test_data = train_test_split(df, test_size = test_size, random_state=random_state, shuffle=shuffle)
        if save:
            train_data.to_csv("data/clean/train.csv")
            test_data.to_csv("data/clean/test.csv")
        return train_data, test_data
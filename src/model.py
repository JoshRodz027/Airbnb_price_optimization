import pandas as pd
import numpy as np
from pandas import DataFrame
from typing import Dict, List , Union


import scipy.stats as stats
from scipy.stats import norm
from scipy.special import boxcox1p

import statsmodels
import statsmodels.api as sm
#print(statsmodels.__version__)

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, StratifiedKFold, RandomizedSearchCV
from sklearn.feature_selection import RFE,RFECV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, LinearRegression, ElasticNet,  HuberRegressor 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix ,mean_squared_error, r2_score, make_scorer,mean_absolute_error

from xgboost import XGBRegressor



class ModelEvaluator:
    def __init__(self,n_folds:int=None):
        if n_folds:
            self.n_folds = n_folds
            return 
        self.n_folds = 5


    # Try to penalise the model with absolute 
    def rmse_cv(self,model,numerical_features:List[str],X_train:DataFrame,y_train:DataFrame
    )->float:
        kf = KFold(self.n_folds, shuffle=True, random_state = 91).get_n_splits(numerical_features)
        return cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=kf)
    # RMSLE has the meaning of a relative error 1.robust with outliers , 2.gives bigger penality for under prediction --> so for our use case we dont want our customers to feel that they got less rental.
    @staticmethod
    def rmlse(y_test:DataFrame, y_pred:DataFrame,convertExp=True)->float:
        if convertExp:
            y_test = np.exp(y_test),
            y_pred = np.exp(y_pred)
        log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y_test]))
        log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_pred]))
        calc = (log1 - log2) ** 2
        return np.sqrt(np.mean(calc))


    # The proportion of the variance in the dependent variable that is predictable from the independent variable(s)
    @staticmethod
    def r2(y_test:DataFrame,y_pred:DataFrame)->float:
        return r2_score(y_test,y_pred)



class Model:

    def __init__(self):
        # init your model here 
        # Regressor is used to predict continuous values like price, where classifier is used to preduict discret value like gender.
        self.models = {'DecisionTreeRegressor': DecisionTreeRegressor(),
                  'RandomForestRegressor': RandomForestRegressor(),
                  'LinearRegression': LinearRegression(),
                  'KNeighborsRegressor': KNeighborsRegressor(),
                  "GradientBoostingRegressor": GradientBoostingRegressor(),
                    "Lasso": Lasso(),
                    "Ridge": Ridge(),
                    "AdaBoostRegressor": AdaBoostRegressor(), 
                    "XGBRegressor": XGBRegressor(), 
                  }
        
        self.model = None #selected model
        self.rfecv = None
        self.n_fold = 5
        self.model_eval = ModelEvaluator()
            


    def train(self, params:Dict[str,Union[str,int]], X_train:DataFrame, y_train:DataFrame)-> Union[float,float,float,float]:
        self.model = self.models[params.pop("model")] #.pop removes the setting/dict thats left in model 
        
        # extract non model related args
        is_rfecv = (("RFECV" in params.keys()) and (params.pop("RFECV")==1))

        # RFE
        print(f"RFECV: {is_rfecv}")
        if is_rfecv:
            # process data with recursive feature elimination
            X_train = self.rfecv_process(X_train, y_train)  
        # Your implementation goes here
        self.model.set_params(**params)
        self.model.fit(X_train, y_train)
        y_pred_train = self.model.predict(X_train)
        # For our case, this function should train the initialised model and return the train mse
        mse_train = mean_squared_error(y_train, y_pred_train)
        print(f"mse_train score is :{mse_train}")
        # Preparing RMSLE
        rmlse = self.model_eval.rmlse(y_train,y_pred_train)
        print (f"RMSLE Value is :{rmlse} ")
        # Preparing rmsw_w_cv
        numerical_features =['latitude', 'longitude', 'minimum_nights', 'number_of_reviews',
       'reviews_per_month', 'calculated_host_listings_count',
       'availability_365', 'all_year_avail', 'low_avail', 'no_reviews']
        rmse_w_cv = self.model_eval.rmse_cv(self.model,numerical_features,X_train,y_train)
        print(f"rmse_w_cv score is :{rmse_w_cv}")
        # R2 score 
        r_sqr = self.model_eval.r2(y_train,y_pred_train)
        print(f"r2_score score is :{r_sqr}")
        print(f'Initiated model: {self.model} successfully')
        return mse_train , rmse_w_cv , r_sqr, rmlse

    def rfecv_process(self, X_train:DataFrame, y_train:DataFrame)->DataFrame:
        # do rfecv
        print(self.model)
        self.rfecv = RFECV(self.model,cv=KFold(self.n_fold),scoring='r2',step=5)
        print("Fitting rfecv")
        self.rfecv.fit(X_train,y_train)
        print('Optimal number of features:{} selected'.format(self.rfecv.n_features_))
        print("rfecv on training data")
        X_train = self.rfecv.transform(X_train)
        return X_train



    def evaluate(self, X_test:DataFrame, y_test:DataFrame)->Union[float,float,float,float]:
        """
        Description of the function. 
        
        :param X_test: ......
        :param y_test: ......
        :return: ......
        """
        #For RFE
        if self.rfecv!= None:
            print("rfecv done on testing data")
            X_test = self.rfecv.transform(X_test)

        # This should use the trained model to predict the target for the test_data and return the test mse  
        y_pred_test = self.model.predict(X_test)
        mse_test = mean_squared_error(y_test,y_pred_test)
        print(f"mse_test score is :{mse_test}")
        # Preparing rmsw_w_cv
        numerical_features =['latitude', 'longitude', 'minimum_nights', 'number_of_reviews',
       'reviews_per_month', 'calculated_host_listings_count',
       'availability_365', 'all_year_avail', 'low_avail', 'no_reviews']
        rmse_w_cv = self.model_eval.rmse_cv(self.model,numerical_features,X_test,y_test)
        print(f"rmse_w_cv score is :{rmse_w_cv}")

        # Preparing RMSLE
        rmlse = self.model_eval.rmlse(y_test,y_pred_test)
        print (f"RMSLE Value is :{rmlse} ")
        r_sqr = self.model_eval.r2(y_test,y_pred_test)
        print (f"r2_score Value is :{r_sqr} ")
        return mse_test,rmse_w_cv, r_sqr, rmlse
    


import pandas as pd
from pandas import DataFrame
from typing import List

import scipy.stats as stats
from scipy.stats import norm
from scipy.special import boxcox1p

import statsmodels
import statsmodels.api as sm
#print(statsmodels.__version__)

from sklearn.preprocessing import scale, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, LinearRegression, ElasticNet,  HuberRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.utils import resample


from xgboost import XGBRegressor




class ModelEvaluator:
    def __init__(self,n_folds:int=None):
        if n_folds:
            self.n_folds = n_folds
            return 
        self.n_folds = 5


    # squared_loss
    def rmse_cv(self,model,numerical_features:List[str],X_train:DataFrame,y_train:DataFrame
    )->float:
        kf = KFold(self.n_folds, shuffle=True, random_state = 91).get_n_splits(numerical_features)
        return cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=kf)

    def rmse_lv_cv(self,model,numerical_features:List[str],X_train:DataFrame,y_train:DataFrame
    )->float:
        kf = KFold(self.n_folds, shuffle=True, random_state = 91).get_n_splits(numerical_features)
        return cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=kf)
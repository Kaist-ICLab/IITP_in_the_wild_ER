
import os
import pandas as pd
import numpy as np
from datetime import date, timedelta
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, KFold


# Label preprocessing
def remove_unreliable_labels(df):
    df = df.copy()
    start_col = df.columns.get_loc('stress')
    end_col = df.columns.get_loc('valence')
    for i in range(1,6,1):
        df = df.loc[~(df.iloc[:, start_col:end_col+1]==i).all(axis=1)]
    return df

def label_based_binary_encoding(df: pd.DataFrame, label_col: str, threshold:int = 2) -> pd.DataFrame:
    """
    Consider 1 if label is above threshold otherwise 0
    Labeling was used for (Low: 1 / High: 2,3,4,5)
    """
    df_ = df.copy()
    df_[label_col] = [1 if val > threshold else 0 for val in df[label_col].values]
    return df_
def dtr_binary_encoding(df: pd.DataFrame, label_col : str)-> pd.DataFrame:
    """
    TODO: Please refer the correct reference
    """
    df_ = df.copy()
    tmp = df_[[label_col, 'pnum']].groupby('pnum').transform(lambda x: round(x.mean()) - .5)
    df_[label_col] = [ 1 if x > y else 0 for x,y in zip(df_[label_col].values,tmp.values)]
    return df_
def average_based_binary_encoding(df: pd.DataFrame, label_col: str)-> pd.DataFrame:
    """
    Labeling used on https://dl.acm.org/doi/abs/10.1145/3351249
    """
    df_ = df.copy()
    tmp = df_[[label_col, 'pnum']].groupby('pnum').transform(lambda x: x.mean())
    df_[label_col] = [ 1 if x>y else 0 for x,y in zip(df_[label_col].values,tmp.values)]
    return df_

# Data Preprocessing
def min_max(column):
        return (column - column.min()) / (column.max() - column.min() + 1e-10)  # 작은 값을 더해 NaN을 방지

def z_score(column):
    mean = column.mean()
    std = column.std(ddof=0)
    return (column - mean) / std if std != 0 else 0  # NaN이 발생하면 0으로 대체

def normalization(df, normalized_method, numeric_col):
    df = df.copy()
    
    if normalized_method == "no":
        return df 
    elif normalized_method == "min_max":
        df[numeric_col] = df.groupby('pnum')[numeric_col].transform(min_max)
        return df 
    elif normalized_method == "z_norm":
        df[numeric_col] = df.groupby('pnum')[numeric_col].transform(z_score)
        return df 

from sklearn.impute import KNNImputer
def impute_knn_per_person(df,group_col ,numeric_features, n_neighbors=3):
    # 각 개인별로 그룹화하여 결측값을 채움
    imputed_df = df.copy()

    # Step 1: 전역적으로 KNN Imputer 적용
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_df[numeric_features] = imputer.fit_transform(imputed_df[numeric_features])
    
    # Step 2: 그룹별 평균값으로 보정
    for feature in numeric_features:
        imputed_df[feature] = imputed_df.groupby(group_col)[feature].transform(lambda x: x.fillna(x.mean()))
    
    return imputed_df


def one_hot(df,categorical_feature):
    df = df.copy()
    ohe = OneHotEncoder(sparse_output=False,handle_unknown='ignore')
    ohe_data = ohe.fit_transform(df[categorical_feature])
    col_name = ohe.get_feature_names_out(categorical_feature)

    ohe_df = pd.DataFrame(ohe_data,columns=col_name,index = df.index)
    df = df.drop(categorical_feature,axis=1)

    # concatenate
    total_df = pd.concat([df,ohe_df],axis=1)
    return total_df

def select_features_losso(X, y, pids):

    logo = LeaveOneGroupOut()

    # 알파 값들의 범위 설정
    alpha_range = np.logspace(-4, 1, 50)
    param_grid = {'alpha': alpha_range}

    lasso = Lasso(max_iter=1000) # default 값
    grid_search = GridSearchCV(lasso, param_grid, cv=logo.split(X, y, pids), scoring='roc_auc')

    grid_search.fit(X, y)
    best_alpha = grid_search.best_params_['alpha']

    # 최적의 알파 값을 사용하여 라쏘 모델 학습
    lasso_best = Lasso(alpha=best_alpha, max_iter=10000)
    lasso_best.fit(X, y)

    # 중요한 피처의 이름 추출
    feature_names = X.columns
    important_features = feature_names[lasso_best.coef_ != 0]

    return list(important_features), best_alpha

def select_features_lasso_personalized(X, y, k=2):

    # k-fold 교차 검증 설정
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # 알파 값들의 범위 설정
    alpha_range = np.logspace(-4, 1, 50)
    param_grid = {'alpha': alpha_range}

    # 라쏘 모델과 그리드 서치 설정
    lasso = Lasso(max_iter=1000)  # default 값
    grid_search = GridSearchCV(lasso, param_grid, cv=kf, scoring='roc_auc')

    # 그리드 서치로 최적의 알파 값 찾기
    grid_search.fit(X, y)
    best_alpha = grid_search.best_params_['alpha']

    # 최적의 알파 값을 사용하여 라쏘 모델 학습
    lasso_best = Lasso(alpha=best_alpha, max_iter=10000)
    lasso_best.fit(X, y)

    # 중요한 피처의 이름 추출
    feature_names = X.columns
    important_features = feature_names[lasso_best.coef_ != 0]

    return list(important_features), best_alpha

def select_one_high_correlation_column(X_train_df,y_train_df, threshold=0.8):

    X_train = X_train_df.copy()
    # X_train과 y_train 간의 상관계수 계산
    X_y_train = pd.concat([X_train, y_train_df.rename("target")], axis=1)
    target_corr = X_y_train.corr().abs()["target"][:-1]  # y_train과 각 feature 간 상관계수
    
    
    # 상관계수 행렬 계산
    corr_matrix = X_train.corr().abs()
    
    # 상삼각 행렬의 상관계수 값 중 threshold 이상인 값만 선택
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # 상관계수 값이 threshold 이상인 변수 쌍을 찾고 y_train과의 상관관계가 작은 변수 선택
    to_drop = set()
    for column in upper_triangle.columns:
        for index in upper_triangle.index:
            if upper_triangle.loc[index, column] > threshold:
                # y_train과의 상관관계가 작은 feature를 to_drop에 추가
                if target_corr[index] < target_corr[column]:
                    to_drop.add(index)
                else:
                    to_drop.add(column)
    x_feature = X_train.drop(columns=to_drop).columns

    # 상관계수가 높은 변수 중 첫 번째 열을 남기고 나머지 제거
    return list(x_feature)



# Oversampling
# def apply_smote(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, random_state: int): # 이게 더 좋지 않은 성능을 보였음. 
#     smote = SMOTE(random_state= random_state)
#     group = -1
#     try:
#         upsampled_X, upsampled_y, upsampled_groups = [], [], []
#         for group in groups.unique():
#             indices = groups == group
#             upsampled_X_, upsampled_y_ = smote.fit_resample(X.loc[indices], y[indices])
#             upsampled_X.append(upsampled_X_)
#             upsampled_y.append(upsampled_y_)
#             upsampled_groups += [group] * len(upsampled_X_)
#         upsampled_X = pd.concat(upsampled_X, axis = 0)
#         upsampled_y = pd.concat(upsampled_y, axis=0)
#         return upsampled_X, upsampled_y, np.array(upsampled_groups)
#     except Exception as e:
#         print(f"ERROR for group ${group}: {e}")
#         return X,y,groups


def apply_smote_person(X: pd.DataFrame, y: np.ndarray, pid:int, random_state: int):
    smote = SMOTE(random_state= random_state)
    try:
        upsampled_X, upsampled_y = smote.fit_resample(X, y)
        return upsampled_X, upsampled_y
    except Exception as e:
        print(f"ERROR for group ${pid}: {e}")
        return X, y
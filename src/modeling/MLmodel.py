# basic library
import pandas as pd
import numpy as np
from src.modeling.metric import * 
from src.modeling.preprocessing import *   
import matplotlib.pyplot as plt



from sklearn.model_selection import GridSearchCV,LeaveOneGroupOut,StratifiedShuffleSplit
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

import shap

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")



# majority voting
def majority_voting(df: pd.DataFrame,numeric_feature:list,experiment_num: int, random_seed: int,use_smote: bool, result_path: str):

    df_ = df.sort_values(['pnum','start_second']).drop(['start_second'],axis=1)
    for exp_num in range(experiment_num):
        results = []

        for pid in df['pnum'].unique():
            print('start_pid: ',pid)

            # Train / test split
            train_df = df_.loc[df_['pnum'] !=pid]
            test_df =  df_.loc[df_['pnum'] ==pid]

            split_index = int(len(test_df) * 0.5)
            train_df_target_pid = test_df.iloc[:split_index]
            
            test_df = test_df.iloc[split_index:]
            train_df = pd.concat([train_df,train_df_target_pid],axis=0)

            if len(test_df['surface_acting'].unique())< 2:
                print(f"pnum {pid} has only one class.")
                continue

            # Average label encoding
            target_thred = train_df.query('pnum == @pid')['surface_acting'].mean()
            train_df = average_based_binary_encoding(train_df,'surface_acting')
            test_df['surface_acting'] = test_df['surface_acting'].apply(lambda x: 1 if x > target_thred else 0)

             # Predict and evaluate using majority voting
            X_test = test_df.drop(['pnum', 'surface_acting'], axis=1)
            y_test = test_df['surface_acting'].values
            majority_metric = majority_voting(X_test, y_test, test_df['pnum'])
            results.append(majority_metric)

        # Collect results for this experiment
        mean_result = pd.concat(results)

    return  mean_result

### General model ###

def LDA_General(df: pd.DataFrame,numeric_feature:list,experiment_num: int, random_seed: int,use_smote: bool, result_path: str):
    result_df = pd.DataFrame()
    x_feature_df = pd.DataFrame(columns=['experiment_num', 'selected_features'])
    feature_importance_df = pd.DataFrame(columns=['pid', 'abs_feature', 'abs_value', 'pos_feature', 'pos_value', 'neg_feature', 'neg_value'])

    df_ = df.sort_values(['pnum','start_second']).drop(['start_second'],axis=1)
    for exp_num in range(experiment_num):
        for pid in df['pnum'].unique():
            print('start_pid: ',pid)

            all_SHAP_values = []
            all_data = []

            # Train / test split
            train_df = df_.loc[df_['pnum'] !=pid]
            test_df =  df_.loc[df_['pnum'] ==pid]

            split_index = int(len(test_df) * 0.5)
            train_df_target_pid = test_df.iloc[:split_index]
            
            test_df = test_df.iloc[split_index:]
            train_df = pd.concat([train_df,train_df_target_pid],axis=0)

            if len(test_df['surface_acting'].unique())< 2:
                print(f"pnum {pid} has only one class.")
                continue

            # Average label encoding
            train_df = average_based_binary_encoding(train_df,'surface_acting')
            test_df = average_based_binary_encoding(test_df,'surface_acting')

            # Normalization
            if len(numeric_feature) > 0:
                train_df = normalization(train_df, 'z_norm', numeric_feature)
                test_df = normalization(test_df, 'z_norm', numeric_feature)

                # Normalization이 안되는 column 삭제하기
                nan_columns_train = train_df.columns[(np.isinf(train_df) | np.isnan(train_df)).any()].tolist()
                nan_columns_test =  test_df.columns[(np.isinf(test_df) | np.isnan(test_df)).any()].tolist()
                print("normalization이 안되는 column",nan_columns_train,nan_columns_test)

                nan_columns = nan_columns_train + nan_columns_test
                train_df.drop(columns=nan_columns, inplace=True)
                test_df.drop(columns=nan_columns, inplace=True)
            else:
                pass

            # Feature selection
            variance_threshold = 1e-07 # zero variance (basic_feature 제외) / audio feature도 3개 이미 빠짐
            variance_columns = list(train_df.loc[:, train_df.var() > variance_threshold].columns)
            train_df = train_df[variance_columns]
            test_df = test_df[variance_columns]

            x_feature = select_one_high_correlation_column(train_df.drop(['surface_acting'],axis=1),train_df['surface_acting'],0.8) # pairewise
            train_df = train_df[x_feature + ['surface_acting']]
            test_df = test_df[x_feature + ['surface_acting']]

            x_feature, alpha = select_features_losso(
                train_df.drop(['pnum','surface_acting'], axis=1), 
                train_df['surface_acting'],
                train_df['pnum']
            ) # lasso

            if len(x_feature)==0:
                print('no feature selection: ',pid)
                x_feature = list(train_df.drop(['surface_acting'], axis=1).columns)
                new_feature_row = pd.DataFrame({'pnum': [pid], 'experiment_num': [exp_num], 'selected_features': [[]]})
            else:
                new_feature_row = pd.DataFrame({'pnum': [pid], 'experiment_num': [exp_num], 'selected_features': [x_feature]})
                x_feature_df = pd.concat([x_feature_df, new_feature_row], ignore_index=True)

            X_train, y_train = train_df[x_feature], train_df['surface_acting']
            X_test, y_test = test_df[x_feature], test_df['surface_acting']

            # Data 증강
            if use_smote:
                X_train, y_train = apply_smote_person(X_train, y_train, pid, random_state=random_seed)

            # missing_values = X_test.isnull().sum()

            # # 결측값이 있는 열만 출력
            # missing_columns = missing_values[missing_values > 0]
            # if missing_columns.empty:
            #     print("X_train에는 결측값이 없습니다.")
            # else:
            #     print(f"X_train에는 총 {missing_columns.sum()}개의 결측값이 있습니다.")
            #     print("결측값이 있는 열과 개수는 다음과 같습니다:")
            #     print(missing_columns)

            clf = LinearDiscriminantAnalysis()
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)
            y_prod = clf.predict_proba(X_test)[:, 1]
            person_df = pd.DataFrame({"y_pred": y_pred, "y_prod": y_prod, "y_true": y_test, "pid": [pid] * len(y_test)})

            person_metric_df = generate_metrics(person_df)
            result_df = pd.concat([result_df, person_metric_df], axis=0)

    mean_result_df = result_df.groupby(['pid']).mean().reset_index()
    mean_metrics = mean_result_df.mean(axis=0)
    mean_metrics_std = mean_result_df.std(axis=0).round(2)
    mean_metrics['pid'] = 'all'
    mean_metrics_std['pid'] = 'all_std'
    mean_result_df  = pd.concat([mean_result_df , pd.DataFrame(mean_metrics).T,pd.DataFrame(mean_metrics_std).T], ignore_index=True)

    return mean_result_df, x_feature_df


def KNN_General(df: pd.DataFrame,numeric_feature:list,experiment_num: int, random_seed: int,use_smote: bool, result_path: str):
    result_df = pd.DataFrame()
    x_feature_df = pd.DataFrame(columns=['experiment_num', 'selected_features'])
    feature_importance_df = pd.DataFrame(columns=['pid', 'abs_feature', 'abs_value', 'pos_feature', 'pos_value', 'neg_feature', 'neg_value'])

    df_ = df.sort_values(['pnum','start_second']).drop(['start_second'],axis=1)
    for exp_num in range(experiment_num):
        for pid in df['pnum'].unique():
            print('start_pid: ',pid)

            all_SHAP_values = []
            all_data = []

            # Train / test split
            train_df = df_.loc[df_['pnum'] !=pid]
            test_df =  df_.loc[df_['pnum'] ==pid]

            split_index = int(len(test_df) * 0.5)
            train_df_target_pid = test_df.iloc[:split_index]
            
            test_df = test_df.iloc[split_index:]
            train_df = pd.concat([train_df,train_df_target_pid],axis=0)

            if len(test_df['surface_acting'].unique())< 2:
                print(f"pnum {pid} has only one class.")
                continue

            # Average label encoding
            train_df = average_based_binary_encoding(train_df,'surface_acting')
            test_df = average_based_binary_encoding(test_df,'surface_acting')

            # Normalization
            if len(numeric_feature) > 0:
                train_df = normalization(train_df, 'z_norm', numeric_feature)
                test_df = normalization(test_df, 'z_norm', numeric_feature)

                # Normalization이 안되는 column 삭제하기
                nan_columns_train = train_df.columns[(np.isinf(train_df) | np.isnan(train_df)).any()].tolist()
                nan_columns_test =  test_df.columns[(np.isinf(test_df) | np.isnan(test_df)).any()].tolist()
                print("normalization이 안되는 column",nan_columns_train,nan_columns_test)

                nan_columns = nan_columns_train + nan_columns_test
                train_df.drop(columns=nan_columns, inplace=True)
                test_df.drop(columns=nan_columns, inplace=True)
            else:
                pass


            # Feature selection
            variance_threshold = 1e-07 # zero variance (basic_feature 제외) / audio feature도 3개 이미 빠짐
            variance_columns = list(train_df.loc[:, train_df.var() > variance_threshold].columns)
            train_df = train_df[variance_columns]
            test_df = test_df[variance_columns]

            x_feature = select_one_high_correlation_column(train_df.drop(['surface_acting'],axis=1),train_df['surface_acting'],0.8) # pairewise
            train_df = train_df[x_feature + ['surface_acting']]
            test_df = test_df[x_feature + ['surface_acting']]

            x_feature, alpha = select_features_losso(
                train_df.drop(['pnum','surface_acting'], axis=1), 
                train_df['surface_acting'],
                train_df['pnum']
            ) # lasso

            if len(x_feature)==0:
                print('no feature selection: ',pid)
                x_feature = list(train_df.drop(['surface_acting'], axis=1).columns)
                new_feature_row = pd.DataFrame({'pnum': [pid], 'experiment_num': [exp_num], 'selected_features': [[]]})
            else:
                new_feature_row = pd.DataFrame({'pnum': [pid], 'experiment_num': [exp_num], 'selected_features': [x_feature]})
                x_feature_df = pd.concat([x_feature_df, new_feature_row], ignore_index=True)

            X_train, y_train = train_df[x_feature], train_df['surface_acting']
            X_test, y_test = test_df[x_feature], test_df['surface_acting']

            # Data 증강
            if use_smote:
                X_train, y_train = apply_smote_person(X_train, y_train, pid, random_state=random_seed)

            clf = KNeighborsClassifier(n_neighbors=9) # Referred to WESAD
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)
            y_prod = clf.predict_proba(X_test)[:, 1]
            person_df = pd.DataFrame({"y_pred": y_pred, "y_prod": y_prod, "y_true": y_test, "pid": [pid] * len(y_test)})

            person_metric_df = generate_metrics(person_df)
            result_df = pd.concat([result_df, person_metric_df], axis=0)

    mean_result_df = result_df.groupby(['pid']).mean().reset_index()
    mean_metrics = mean_result_df.mean(axis=0)
    mean_metrics_std = mean_result_df.std(axis=0).round(2)
    mean_metrics['pid'] = 'all'
    mean_metrics_std['pid'] = 'all_std'
    mean_result_df  = pd.concat([mean_result_df , pd.DataFrame(mean_metrics).T,pd.DataFrame(mean_metrics_std).T], ignore_index=True)

    return mean_result_df, x_feature_df


def SVM_General(df: pd.DataFrame,numeric_feature:list,experiment_num: int, random_seed: int,use_smote: bool, result_path: str):
    result_df = pd.DataFrame()
    x_feature_df = pd.DataFrame(columns=['experiment_num', 'selected_features'])
    feature_importance_df = pd.DataFrame(columns=['pid', 'abs_feature', 'abs_value', 'pos_feature', 'pos_value', 'neg_feature', 'neg_value'])

    df_ = df.sort_values(['pnum','start_second']).drop(['start_second'],axis=1)
    for exp_num in range(experiment_num):
        for pid in df['pnum'].unique():
            print('start_pid: ',pid)

            all_SHAP_values = []
            all_data = []

            # Train / test split
            train_df = df_.loc[df_['pnum'] !=pid]
            test_df =  df_.loc[df_['pnum'] ==pid]

            split_index = int(len(test_df) * 0.5)
            train_df_target_pid = test_df.iloc[:split_index]
            
            test_df = test_df.iloc[split_index:]
            train_df = pd.concat([train_df,train_df_target_pid],axis=0)

            if len(test_df['surface_acting'].unique())< 2:
                print(f"pnum {pid} has only one class.")
                continue

            # Average label encoding
            train_df = average_based_binary_encoding(train_df,'surface_acting')
            test_df = average_based_binary_encoding(test_df,'surface_acting')

            # Normalization
            if len(numeric_feature) > 0:
                train_df = normalization(train_df, 'z_norm', numeric_feature)
                test_df = normalization(test_df, 'z_norm', numeric_feature)

                # Normalization이 안되는 column 삭제하기
                nan_columns_train = train_df.columns[(np.isinf(train_df) | np.isnan(train_df)).any()].tolist()
                nan_columns_test =  test_df.columns[(np.isinf(test_df) | np.isnan(test_df)).any()].tolist()
                print("normalization이 안되는 column",nan_columns_train,nan_columns_test)

                nan_columns = nan_columns_train + nan_columns_test
                train_df.drop(columns=nan_columns, inplace=True)
                test_df.drop(columns=nan_columns, inplace=True)
            else:
                pass


            # Feature selection
            variance_threshold = 1e-07 # zero variance (basic_feature 제외) / audio feature도 3개 이미 빠짐
            variance_columns = list(train_df.loc[:, train_df.var() > variance_threshold].columns)
            train_df = train_df[variance_columns]
            test_df = test_df[variance_columns]

            x_feature = select_one_high_correlation_column(train_df.drop(['surface_acting'],axis=1),train_df['surface_acting'],0.8) # pairewise
            train_df = train_df[x_feature + ['surface_acting']]
            test_df = test_df[x_feature + ['surface_acting']]

            x_feature, alpha = select_features_losso(
                train_df.drop(['pnum','surface_acting'], axis=1), 
                train_df['surface_acting'],
                train_df['pnum']
            ) # lasso

            if len(x_feature)==0:
                print('no feature selection: ',pid)
                x_feature = list(train_df.drop(['surface_acting'], axis=1).columns)
                new_feature_row = pd.DataFrame({'pnum': [pid], 'experiment_num': [exp_num], 'selected_features': [[]]})
            else:
                new_feature_row = pd.DataFrame({'pnum': [pid], 'experiment_num': [exp_num], 'selected_features': [x_feature]})
                x_feature_df = pd.concat([x_feature_df, new_feature_row], ignore_index=True)

            X_train, y_train = train_df[x_feature], train_df['surface_acting']
            X_test, y_test = test_df[x_feature], test_df['surface_acting']

            # Data 증강
            if use_smote:
                X_train, y_train = apply_smote_person(X_train, y_train, pid, random_state=random_seed)

            clf = SVC(probability=True, kernel='rbf', C=1, gamma=0.03) # Referred to paper https://dl.acm.org/doi/pdf/10.1145/3173574.3174165
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)
            y_prod = clf.predict_proba(X_test)[:, 1]
            person_df = pd.DataFrame({"y_pred": y_pred, "y_prod": y_prod, "y_true": y_test, "pid": [pid] * len(y_test)})

            person_metric_df = generate_metrics(person_df)
            result_df = pd.concat([result_df, person_metric_df], axis=0)

    mean_result_df = result_df.groupby(['pid']).mean().reset_index()
    mean_metrics = mean_result_df.mean(axis=0)
    mean_metrics_std = mean_result_df.std(axis=0).round(2)
    mean_metrics['pid'] = 'all'
    mean_metrics_std['pid'] = 'all_std'
    mean_result_df  = pd.concat([mean_result_df , pd.DataFrame(mean_metrics).T,pd.DataFrame(mean_metrics_std).T], ignore_index=True)

    return mean_result_df, x_feature_df


def AdaBoost_General(df: pd.DataFrame,numeric_feature:list,experiment_num: int, random_seed: int,use_smote: bool, result_path: str):
    result_df = pd.DataFrame()
    x_feature_df = pd.DataFrame(columns=['experiment_num', 'selected_features'])
    feature_importance_df = pd.DataFrame(columns=['pid', 'abs_feature', 'abs_value', 'pos_feature', 'pos_value', 'neg_feature', 'neg_value'])

    df_ = df.sort_values(['pnum','start_second']).drop(['start_second'],axis=1)
    for exp_num in range(experiment_num):
        for pid in df['pnum'].unique():
            print('start_pid: ',pid)

            all_SHAP_values = []
            all_data = []

            # Train / test split
            train_df = df_.loc[df_['pnum'] !=pid]
            test_df =  df_.loc[df_['pnum'] ==pid]

            split_index = int(len(test_df) * 0.5)
            train_df_target_pid = test_df.iloc[:split_index]
            
            test_df = test_df.iloc[split_index:]
            train_df = pd.concat([train_df,train_df_target_pid],axis=0)

            if len(test_df['surface_acting'].unique())< 2:
                print(f"pnum {pid} has only one class.")
                continue

            # Average label encoding
            train_df = average_based_binary_encoding(train_df,'surface_acting')
            test_df = average_based_binary_encoding(test_df,'surface_acting')

            # Normalization
            if len(numeric_feature) > 0:
                train_df = normalization(train_df, 'z_norm', numeric_feature)
                test_df = normalization(test_df, 'z_norm', numeric_feature)

                # Normalization이 안되는 column 삭제하기
                nan_columns_train = train_df.columns[(np.isinf(train_df) | np.isnan(train_df)).any()].tolist()
                nan_columns_test =  test_df.columns[(np.isinf(test_df) | np.isnan(test_df)).any()].tolist()
                print("normalization이 안되는 column",nan_columns_train,nan_columns_test)

                nan_columns = nan_columns_train + nan_columns_test
                train_df.drop(columns=nan_columns, inplace=True)
                test_df.drop(columns=nan_columns, inplace=True)
            else:
                pass


            # Feature selection
            variance_threshold = 1e-07 # zero variance (basic_feature 제외) / audio feature도 3개 이미 빠짐
            variance_columns = list(train_df.loc[:, train_df.var() > variance_threshold].columns)
            train_df = train_df[variance_columns]
            test_df = test_df[variance_columns]

            x_feature = select_one_high_correlation_column(train_df.drop(['surface_acting'],axis=1),train_df['surface_acting'],0.8) # pairewise
            train_df = train_df[x_feature + ['surface_acting']]
            test_df = test_df[x_feature + ['surface_acting']]

            x_feature, alpha = select_features_losso(
                train_df.drop(['pnum','surface_acting'], axis=1), 
                train_df['surface_acting'],
                train_df['pnum']
            ) # lasso

            if len(x_feature)==0:
                print('no feature selection: ',pid)
                x_feature = list(train_df.drop(['surface_acting'], axis=1).columns)
                new_feature_row = pd.DataFrame({'pnum': [pid], 'experiment_num': [exp_num], 'selected_features': [[]]})
            else:
                new_feature_row = pd.DataFrame({'pnum': [pid], 'experiment_num': [exp_num], 'selected_features': [x_feature]})
                x_feature_df = pd.concat([x_feature_df, new_feature_row], ignore_index=True)

            X_train, y_train = train_df[x_feature], train_df['surface_acting']
            X_test, y_test = test_df[x_feature], test_df['surface_acting']

            # Data 증강
            if use_smote:
                X_train, y_train = apply_smote_person(X_train, y_train, pid, random_state=random_seed)


            dt = DecisionTreeClassifier(criterion='entropy', min_samples_split=20) # Referred to WESAD
            clf = AdaBoostClassifier(estimator=dt, n_estimators=100, algorithm='SAMME.R') # Referred to WESAD    
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)
            y_prod = clf.predict_proba(X_test)[:, 1]
            person_df = pd.DataFrame({"y_pred": y_pred, "y_prod": y_prod, "y_true": y_test, "pid": [pid] * len(y_test)})

            person_metric_df = generate_metrics(person_df)
            result_df = pd.concat([result_df, person_metric_df], axis=0)

    mean_result_df = result_df.groupby(['pid']).mean().reset_index()
    mean_metrics = mean_result_df.mean(axis=0)
    mean_metrics_std = mean_result_df.std(axis=0).round(2)
    mean_metrics['pid'] = 'all'
    mean_metrics_std['pid'] = 'all_std'
    mean_result_df  = pd.concat([mean_result_df , pd.DataFrame(mean_metrics).T,pd.DataFrame(mean_metrics_std).T], ignore_index=True)

    return mean_result_df, x_feature_df


def RandomForest_General(df: pd.DataFrame,numeric_feature:list,experiment_num: int, random_seed: int,use_smote: bool, result_path: str):
    result_df = pd.DataFrame()
    x_feature_df = pd.DataFrame(columns=['experiment_num', 'selected_features'])
    feature_importance_df = pd.DataFrame(columns=['pid', 'abs_feature', 'abs_value', 'pos_feature', 'pos_value', 'neg_feature', 'neg_value'])

    df_ = df.sort_values(['pnum','start_second']).drop(['start_second'],axis=1)
    for exp_num in range(experiment_num):
                
        all_SHAP_values =[]
        all_data = []
        all_selected_features = list(df_.drop(['pnum','surface_acting'],axis=1).columns)

        for pid in df['pnum'].unique():
            print('start_pid: ',pid)

            all_SHAP_values = []
            all_data = []

            # Train / test split
            train_df = df_.loc[df_['pnum'] !=pid]
            test_df =  df_.loc[df_['pnum'] ==pid]

            split_index = int(len(test_df) * 0.5)
            train_df_target_pid = test_df.iloc[:split_index]
            
            test_df = test_df.iloc[split_index:]
            # train_df = pd.concat([train_df,train_df_target_pid],axis=0)

            if len(test_df['surface_acting'].unique())< 2:
                print(f"pnum {pid} has only one class.")
                continue

            # Average label encoding
            train_df = average_based_binary_encoding(train_df,'surface_acting')
            test_df = average_based_binary_encoding(test_df,'surface_acting')

            # Normalization
            if len(numeric_feature) > 0:
                train_df = normalization(train_df, 'z_norm', numeric_feature)
                test_df = normalization(test_df, 'z_norm', numeric_feature)

                # Normalization이 안되는 column 삭제하기
                nan_columns_train = train_df.columns[(np.isinf(train_df) | np.isnan(train_df)).any()].tolist()
                nan_columns_test =  test_df.columns[(np.isinf(test_df) | np.isnan(test_df)).any()].tolist()
                print("normalization이 안되는 column",nan_columns_train,nan_columns_test)

                nan_columns = nan_columns_train + nan_columns_test
                train_df.drop(columns=nan_columns, inplace=True)
                test_df.drop(columns=nan_columns, inplace=True)
            else:
                pass

            # Feature selection
            variance_threshold = 1e-07 # zero variance (basic_feature 제외) / audio feature도 3개 이미 빠짐
            variance_columns = list(train_df.loc[:, train_df.var() > variance_threshold].columns)
            train_df = train_df[variance_columns]
            test_df = test_df[variance_columns]

            x_feature = select_one_high_correlation_column(train_df.drop(['surface_acting'],axis=1),train_df['surface_acting'],0.8) # pairewise
            train_df = train_df[x_feature + ['surface_acting']]
            test_df = test_df[x_feature + ['surface_acting']]

            x_feature, alpha = select_features_losso(
                train_df.drop(['pnum','surface_acting'], axis=1), 
                train_df['surface_acting'],
                train_df['pnum']
            ) # lasso

            if len(x_feature)==0:
                print('no feature selection: ',pid)
                x_feature = list(train_df.drop(['surface_acting'], axis=1).columns)
                new_feature_row = pd.DataFrame({'pnum': [pid], 'experiment_num': [exp_num], 'selected_features': [[]]})
            else:
                new_feature_row = pd.DataFrame({'pnum': [pid], 'experiment_num': [exp_num], 'selected_features': [x_feature]})
                x_feature_df = pd.concat([x_feature_df, new_feature_row], ignore_index=True)

            # # Data under sampling
            # train_df_exclude_targe = train_df.query('pnum!=@pid')
            # undersample = RandomUnderSampler(sampling_strategy=0.67, random_state=42)
            # X_resampled, y_resampled = undersample.fit_resample(train_df_exclude_targe.drop(['surface_acting'],axis=1), train_df_exclude_targe['surface_acting'])
            # train_df = pd.concat([X_resampled,y_resampled],axis=1)

            X_train, y_train = train_df[x_feature], train_df['surface_acting']
            X_test, y_test = test_df[x_feature], test_df['surface_acting']

            # Data 증강
            if use_smote:
                X_train, y_train = apply_smote_person(X_train, y_train, pid, random_state=random_seed)

            # clf = RandomForestClassifier(criterion='gini', min_samples_split=20, n_estimators=100)
            clf = RandomForestClassifier(n_estimators=500) # Referred to paper https://dl.acm.org/doi/pdf/10.1145/3173574.3174165
            clf.fit(X_train, y_train)


            explainer = shap.TreeExplainer(clf)
            shap_values = explainer(X_test) # output_margin=True 추가  -> 오류 나서 다시 생략
            # print(shap_values)


            shap_values_class = shap_values.values[:,:, 1]  # 첫 번째 클래스 선택
            data_class = shap_values.data  # 첫 번째 클래스 선택


            # 선택된 feature에 대해 SHAP 값을 채우고 없는 값은 0으로 패딩
            shap_values_padded = np.zeros((len(shap_values_class), len(all_selected_features)))
            shap_values_padded[:, [all_selected_features.index(f) for f in x_feature]] = shap_values_class
            all_SHAP_values.append(shap_values_padded)


            data_padded = np.zeros((len(data_class), len(all_selected_features)))
            data_padded[:, [all_selected_features.index(f) for f in x_feature]] = data_class
            all_data.append(data_padded)

            all_SHAP_values_array = np.concatenate(all_SHAP_values, axis=0) ## for sharply value 
            all_data_array = np.concatenate(all_data, axis=0) ## for feature high low


            y_pred = clf.predict(X_test)  
            y_prod = clf.predict_proba(X_test)[:, 1]
            person_df = pd.DataFrame({"y_pred": y_pred, "y_prod": y_prod, "y_true": y_test, "pid": [pid] * len(y_test)})

            person_metric_df = generate_metrics(person_df)
            result_df = pd.concat([result_df, person_metric_df], axis=0)
    
        #SHAP value graph
        shap.summary_plot(all_SHAP_values_array,all_data_array, feature_names=all_selected_features, plot_type='dot', max_display=20,show=False)
        plt.savefig('{0}/RF_shap_plot.png'.format(result_path))
        plt.show() 

    mean_result_df = result_df.groupby(['pid']).mean().reset_index()
    mean_metrics = mean_result_df.mean(axis=0)
    mean_metrics_std = mean_result_df.std(axis=0).round(2)
    mean_metrics['pid'] = 'all'
    mean_metrics_std['pid'] = 'all_std'
    mean_result_df  = pd.concat([mean_result_df , pd.DataFrame(mean_metrics).T,pd.DataFrame(mean_metrics_std).T], ignore_index=True)

    return mean_result_df, x_feature_df




def XGBoost_General(df: pd.DataFrame,numeric_feature:list,experiment_num: int, random_seed: int,use_smote: bool, result_path: str):
    result_df = pd.DataFrame()
    x_feature_df = pd.DataFrame(columns=['experiment_num', 'selected_features'])
    feature_importance_df = pd.DataFrame(columns=['pid', 'abs_feature', 'abs_value', 'pos_feature', 'pos_value', 'neg_feature', 'neg_value'])

    df_ = df.sort_values(['pnum','start_second']).drop(['start_second'],axis=1)
    for exp_num in range(experiment_num):
        for pid in df['pnum'].unique():
            print('start_pid: ',pid)

            all_SHAP_values = []
            all_data = []

            # Train / test split
            train_df = df_.loc[df_['pnum'] !=pid]
            test_df =  df_.loc[df_['pnum'] ==pid]

            split_index = int(len(test_df) * 0.5)
            train_df_target_pid = test_df.iloc[:split_index]
            
            test_df = test_df.iloc[split_index:]
            # train_df = pd.concat([train_df,train_df_target_pid],axis=0)

            if len(test_df['surface_acting'].unique())< 2:
                print(f"pnum {pid} has only one class.")
                continue

            # Average label encoding
            train_df = average_based_binary_encoding(train_df,'surface_acting')
            test_df = average_based_binary_encoding(test_df,'surface_acting')

            # Normalization

            if len(numeric_feature) > 0:
                train_df = normalization(train_df, 'z_norm', numeric_feature)
                test_df = normalization(test_df, 'z_norm', numeric_feature)

                # Normalization이 안되는 column 삭제하기
                nan_columns_train = train_df.columns[(np.isinf(train_df) | np.isnan(train_df)).any()].tolist()
                nan_columns_test =  test_df.columns[(np.isinf(test_df) | np.isnan(test_df)).any()].tolist()
                print("normalization이 안되는 column",nan_columns_train,nan_columns_test)

                nan_columns = nan_columns_train + nan_columns_test
                train_df.drop(columns=nan_columns, inplace=True)
                test_df.drop(columns=nan_columns, inplace=True)
            else:
                pass


            # Feature selection
            variance_threshold = 1e-07 # zero variance (basic_feature 제외) / audio feature도 3개 이미 빠짐
            variance_columns = list(train_df.loc[:, train_df.var() > variance_threshold].columns)
            train_df = train_df[variance_columns]
            test_df = test_df[variance_columns]

            x_feature = select_one_high_correlation_column(train_df.drop(['surface_acting'],axis=1),train_df['surface_acting'],0.8) # pairewise
            train_df = train_df[x_feature + ['surface_acting']]
            test_df = test_df[x_feature + ['surface_acting']]

            x_feature, alpha = select_features_losso(
                train_df.drop(['pnum','surface_acting'], axis=1), 
                train_df['surface_acting'],
                train_df['pnum']
            ) # lasso

            if len(x_feature)==0:
                print('no feature selection: ',pid)
                x_feature = list(train_df.drop(['surface_acting'], axis=1).columns)
                new_feature_row = pd.DataFrame({'pnum': [pid], 'experiment_num': [exp_num], 'selected_features': [[]]})
            else:
                new_feature_row = pd.DataFrame({'pnum': [pid], 'experiment_num': [exp_num], 'selected_features': [x_feature]})
                x_feature_df = pd.concat([x_feature_df, new_feature_row], ignore_index=True)

            X_train, y_train = train_df[x_feature], train_df['surface_acting']
            X_test, y_test = test_df[x_feature], test_df['surface_acting']

            print(y_test.value_counts())

            # Data 증강
            if use_smote:
                X_train, y_train = apply_smote_person(X_train, y_train, pid, random_state=random_seed)

            clf = XGBClassifier(objective='binary:logistic', max_depth=3, learning_rate=0.1, n_estimators=100,
            tree_method='gpu_hist',  # GPU 사용
            predictor='gpu_predictor',  # GPU 예측기 사용
            random_state=random_seed) #Hyperopt 결과 사용


# # {'colsample_bytree': 0.9189474294432578, 'learning_rate': 0.09313128256061605, 'max_depth': 10.0, 'min_child_weight': 8.0, 'n_estimators': 130.0, 'scale_pos_weight': 3.2939863428010874, 'subsample': 0.7566006866828496}
#             clf  = XGBClassifier(colsample_bytree = 0.9189474294432578,
#                                  learning_rate = 0.09313128256061605,
#                                  max_depth = 10,
#                                  min_child_weight = 8,
#                                  n_estimators  = 130,
#                                  scale_pos_weight = 3.2939863428010874,
#                                  subsample = 0.7566006866828496)

            clf.fit(X_train, y_train )

            y_pred = clf.predict(X_test)
            y_prod = clf.predict_proba(X_test)[:, 1]
            person_df = pd.DataFrame({"y_pred": y_pred, "y_prod": y_prod, "y_true": y_test, "pid": [pid] * len(y_test)})

            person_metric_df = generate_metrics(person_df)
            result_df = pd.concat([result_df, person_metric_df], axis=0)

    mean_result_df = result_df.groupby(['pid']).mean().reset_index()
    mean_metrics = mean_result_df.mean(axis=0)
    mean_metrics_std = mean_result_df.std(axis=0).round(2)
    mean_metrics['pid'] = 'all'
    mean_metrics_std['pid'] = 'all_std'
    mean_result_df  = pd.concat([mean_result_df , pd.DataFrame(mean_metrics).T,pd.DataFrame(mean_metrics_std).T], ignore_index=True)

    return mean_result_df, x_feature_df


def DecisionTree_General(df: pd.DataFrame,numeric_feature:list,experiment_num: int, random_seed: int,use_smote: bool, result_path: str):
    result_df = pd.DataFrame()
    x_feature_df = pd.DataFrame(columns=['experiment_num', 'selected_features'])
    feature_importance_df = pd.DataFrame(columns=['pid', 'abs_feature', 'abs_value', 'pos_feature', 'pos_value', 'neg_feature', 'neg_value'])

    df_ = df.sort_values(['pnum','start_second']).drop(['start_second'],axis=1)
    for exp_num in range(experiment_num):
        for pid in df['pnum'].unique():
            print('start_pid: ',pid)

            all_SHAP_values = []
            all_data = []

            # Train / test split
            train_df = df_.loc[df_['pnum'] !=pid]
            test_df =  df_.loc[df_['pnum'] ==pid]

            split_index = int(len(test_df) * 0.5)
            train_df_target_pid = test_df.iloc[:split_index]
            
            test_df = test_df.iloc[split_index:]
            train_df = pd.concat([train_df,train_df_target_pid],axis=0)

            if len(test_df['surface_acting'].unique())< 2:
                print(f"pnum {pid} has only one class.")
                continue

            # Average label encoding
            train_df = average_based_binary_encoding(train_df,'surface_acting')
            test_df = average_based_binary_encoding(test_df,'surface_acting')

            # Normalization

            if len(numeric_feature) > 0:
                train_df = normalization(train_df, 'z_norm', numeric_feature)
                test_df = normalization(test_df, 'z_norm', numeric_feature)

                # Normalization이 안되는 column 삭제하기
                nan_columns_train = train_df.columns[(np.isinf(train_df) | np.isnan(train_df)).any()].tolist()
                nan_columns_test =  test_df.columns[(np.isinf(test_df) | np.isnan(test_df)).any()].tolist()
                print("normalization이 안되는 column",nan_columns_train,nan_columns_test)

                nan_columns = nan_columns_train + nan_columns_test
                train_df.drop(columns=nan_columns, inplace=True)
                test_df.drop(columns=nan_columns, inplace=True)
            else:
                pass


            # Feature selection
            variance_threshold = 1e-07 # zero variance (basic_feature 제외) / audio feature도 3개 이미 빠짐
            variance_columns = list(train_df.loc[:, train_df.var() > variance_threshold].columns)
            train_df = train_df[variance_columns]
            test_df = test_df[variance_columns]

            x_feature = select_one_high_correlation_column(train_df.drop(['surface_acting'],axis=1),train_df['surface_acting'],0.8) # pairewise
            train_df = train_df[x_feature + ['surface_acting']]
            test_df = test_df[x_feature + ['surface_acting']]

            x_feature, alpha = select_features_losso(
                train_df.drop(['pnum','surface_acting'], axis=1), 
                train_df['surface_acting'],
                train_df['pnum']
            ) # lasso

            if len(x_feature)==0:
                print('no feature selection: ',pid)
                x_feature = list(train_df.drop(['surface_acting'], axis=1).columns)
                new_feature_row = pd.DataFrame({'pnum': [pid], 'experiment_num': [exp_num], 'selected_features': [[]]})
            else:
                new_feature_row = pd.DataFrame({'pnum': [pid], 'experiment_num': [exp_num], 'selected_features': [x_feature]})
                x_feature_df = pd.concat([x_feature_df, new_feature_row], ignore_index=True)

            X_train, y_train = train_df[x_feature], train_df['surface_acting']
            X_test, y_test = test_df[x_feature], test_df['surface_acting']

            # Data 증강
            if use_smote:
                X_train, y_train = apply_smote_person(X_train, y_train, pid, random_state=random_seed)

            clf = DecisionTreeClassifier(criterion='entropy', min_samples_split=20) # Referred to WESAD
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            y_prod = clf.predict_proba(X_test)[:, 1]
            person_df = pd.DataFrame({"y_pred": y_pred, "y_prod": y_prod, "y_true": y_test, "pid": [pid] * len(y_test)})

            person_metric_df = generate_metrics(person_df)
            result_df = pd.concat([result_df, person_metric_df], axis=0)

    mean_result_df = result_df.groupby(['pid']).mean().reset_index()
    mean_metrics = mean_result_df.mean(axis=0)
    mean_metrics_std = mean_result_df.std(axis=0).round(2)
    mean_metrics['pid'] = 'all'
    mean_metrics_std['pid'] = 'all_std'
    mean_result_df  = pd.concat([mean_result_df , pd.DataFrame(mean_metrics).T,pd.DataFrame(mean_metrics_std).T], ignore_index=True)

    return mean_result_df, x_feature_df

### Personal Model ### 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def RandomForest_Personalized(df: pd.DataFrame,numeric_feature:list,experiment_num: int, random_seed: int,use_smote: bool, result_path: str):
    result_df = pd.DataFrame()
    x_feature_df = pd.DataFrame(columns=['pnum','experiment_num', 'selected_features'])
    feature_importance_df = pd.DataFrame(columns=['pid', 'abs_feature', 'abs_value', 'pos_feature', 'pos_value', 'neg_feature', 'neg_value'])


    for exp_num in range(experiment_num):
        for pid in df['pnum'].unique():
            print('start_pid: ',pid)
            pnum_data = df[df['pnum'] == pid].sort_values('start_second').drop(['start_second'],axis=1)

            all_selected_features = list(pnum_data.drop(['pnum','surface_acting'],axis=1).columns)

            all_SHAP_values = []
            all_data = []

            # Train / test split
            split_index = int(len(pnum_data) * 0.5)
            train_df = pnum_data.iloc[:split_index]
            test_df = pnum_data.iloc[split_index:]

            if len(test_df['surface_acting'].unique())< 2:
                print(f"pnum {pid} has only one class.")
                continue

            # Average label encoding
            surface_acting_mean = train_df['surface_acting'].mean()
            train_df['surface_acting'] = train_df['surface_acting'].apply(lambda x: 1 if x > surface_acting_mean else 0)
            test_df['surface_acting'] = test_df['surface_acting'].apply(lambda x: 1 if x > surface_acting_mean else 0)

            # Normalization
            scaler = StandardScaler()
            train_df[numeric_feature] = scaler.fit_transform(train_df[numeric_feature])
            test_df[numeric_feature] = scaler.transform(test_df[numeric_feature])

            # Feature selection
            variance_threshold = 1e-07 # zero variance (basic_feature 제외) / audio feature도 3개 이미 빠짐
            variance_columns = list(train_df.loc[:, train_df.var() > variance_threshold].columns)
            train_df = train_df[variance_columns]
            test_df = test_df[variance_columns]

            # Feature selection
            x_feature = select_one_high_correlation_column(train_df.drop(['surface_acting'],axis=1),train_df['surface_acting'],0.8) # pairewise
            train_df = train_df[x_feature + ['surface_acting']]
            test_df = test_df[x_feature + ['surface_acting']]

            x_feature, alpha = select_features_lasso_personalized(
                train_df.drop(['surface_acting'], axis=1), 
                train_df['surface_acting'],
                k=2
            ) # lasso

            if len(x_feature)==0:
                print('no feature selection: ',pid)
                x_feature = list(train_df.drop(['surface_acting'], axis=1).columns)
                new_feature_row = pd.DataFrame({'pnum': [pid], 'experiment_num': [exp_num], 'selected_features': [[]]})
            else:
                new_feature_row = pd.DataFrame({'pnum': [pid], 'experiment_num': [exp_num], 'selected_features': [x_feature]})
                x_feature_df = pd.concat([x_feature_df, new_feature_row], ignore_index=True)

            X_train, y_train = train_df[x_feature], train_df['surface_acting']
            X_test, y_test = test_df[x_feature], test_df['surface_acting']

            # Data 증강
            if use_smote:
                X_train, y_train = apply_smote_person(X_train, y_train, pid, random_state=random_seed)

            clf = RandomForestClassifier(n_estimators=500) # Referred to paper https://dl.acm.org/doi/pdf/10.1145/3173574.3174165
            clf.fit(X_train,y_train )  


            explainer = shap.TreeExplainer(clf)
            shap_values = explainer(X_test) # output_margin=True 추가  -> 오류 나서 다시 생략
            shap_values_class = shap_values.values[:,:, 1]  # 첫 번째 클래스 선택
            data_class = shap_values.data  # 첫 번째 클래스 선택


            # 선택된 feature에 대해 SHAP 값을 채우고 없는 값은 0으로 패딩
            shap_values_padded = np.zeros((len(shap_values_class), len(all_selected_features)))
            shap_values_padded[:, [all_selected_features.index(f) for f in x_feature]] = shap_values_class
            all_SHAP_values.append(shap_values_padded)


            data_padded = np.zeros((len(data_class), len(all_selected_features)))
            data_padded[:, [all_selected_features.index(f) for f in x_feature]] = data_class
            all_data.append(data_padded)

            all_SHAP_values_array = np.concatenate(all_SHAP_values, axis=0) ## for sharply value 
            all_data_array = np.concatenate(all_data, axis=0) ## for feature high low

            y_pred = clf.predict(X_test)
            y_prod = clf.predict_proba(X_test)[:, 1]
            person_df = pd.DataFrame({"y_pred": y_pred, "y_prod": y_prod, "y_true": y_test, "pid": [pid] * len(y_test)})


            person_metric_df = generate_metrics(person_df)
            result_df = pd.concat([result_df, person_metric_df], axis=0)
        #SHAP value graph
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        font_path = '/home/iclab/HJ/font/NanumGothic.ttf'
        font_prop = font_manager.FontProperties(fname=font_path)

        print()
        shap.summary_plot(all_SHAP_values_array,all_data_array, feature_names=all_selected_features, plot_type='dot', max_display=20,show=False)
        fig = plt.gcf()  # 현재 플롯 객체 가져오기
        for text in fig.findobj(plt.Text):  # 모든 텍스트 객체에 대해
            text.set_fontproperties(font_prop)
        plt.savefig('{0}/RF_shap_personalized_plot.png'.format(result_path))
        plt.show() 

    mean_result_df = result_df.groupby(['pid']).mean().reset_index()
    mean_metrics = mean_result_df.mean(axis=0)
    mean_metrics_std = mean_result_df.std(axis=0).round(3)
    mean_metrics['pid'] = 'all'
    mean_metrics_std['pid'] = 'all_std'
    mean_result_df  = pd.concat([mean_result_df , pd.DataFrame(mean_metrics).T,pd.DataFrame(mean_metrics_std).T], ignore_index=True)

    return mean_result_df, x_feature_df



def XGBoost_Personalized(df: pd.DataFrame,numeric_feature:list,experiment_num: int, random_seed: int,use_smote: bool, result_path: str):
    result_df = pd.DataFrame()
    x_feature_df = pd.DataFrame(columns=['experiment_num', 'selected_features'])
    feature_importance_df = pd.DataFrame(columns=['pid', 'abs_feature', 'abs_value', 'pos_feature', 'pos_value', 'neg_feature', 'neg_value'])

    for exp_num in range(experiment_num):
        for pid in df['pnum'].unique():
            print('start_pid: ',pid)
            pnum_data = df[df['pnum'] == pid].sort_values('start_second').drop(['start_second'],axis=1)

            all_SHAP_values = []
            all_data = []

            # Train / test split
            split_index = int(len(pnum_data) * 0.5)
            train_df = pnum_data.iloc[:split_index]
            test_df = pnum_data.iloc[split_index:]

            if len(test_df['surface_acting'].unique())< 2:
                print(f"pnum {pid} has only one class.")
                continue

            # Average label encoding
            surface_acting_mean = train_df['surface_acting'].mean()
            train_df['surface_acting'] = train_df['surface_acting'].apply(lambda x: 1 if x > surface_acting_mean else 0)
            test_df['surface_acting'] = test_df['surface_acting'].apply(lambda x: 1 if x > surface_acting_mean else 0)

            # Normalization
            scaler = StandardScaler()
            train_df[numeric_feature] = scaler.fit_transform(train_df[numeric_feature])
            test_df[numeric_feature] = scaler.transform(test_df[numeric_feature])

            # Feature selection
            variance_threshold = 1e-07 # zero variance (basic_feature 제외) / audio feature도 3개 이미 빠짐
            variance_columns = list(train_df.loc[:, train_df.var() > variance_threshold].columns)
            train_df = train_df[variance_columns]
            test_df = test_df[variance_columns]

            # Feature selection
            x_feature = select_one_high_correlation_column(train_df.drop(['surface_acting'],axis=1),train_df['surface_acting'],0.8) # pairewise
            train_df = train_df[x_feature + ['surface_acting']]
            test_df = test_df[x_feature + ['surface_acting']]

            x_feature, alpha = select_features_lasso_personalized(
                train_df.drop(['surface_acting'], axis=1), 
                train_df['surface_acting'],
                k=2
            ) # lasso


            if len(x_feature)==0:
                print('no feature selection: ',pid)
                x_feature = list(train_df.drop(['surface_acting'], axis=1).columns)
                new_feature_row = pd.DataFrame({'pnum': [pid], 'experiment_num': [exp_num], 'selected_features': [[]]})
            else:
                new_feature_row = pd.DataFrame({'pnum': [pid], 'experiment_num': [exp_num], 'selected_features': [x_feature]})
                x_feature_df = pd.concat([x_feature_df, new_feature_row], ignore_index=True)

            X_train, y_train = train_df[x_feature], train_df['surface_acting']
            X_test, y_test = test_df[x_feature], test_df['surface_acting']

            # Data 증강
            if use_smote:
                X_train, y_train = apply_smote_person(X_train, y_train, pid, random_state=random_seed)

            clf = XGBClassifier(objective='binary:logistic', max_depth=3, learning_rate=0.1, n_estimators=100,
            tree_method='gpu_hist',  # GPU 사용
            predictor='gpu_predictor',  # GPU 예측기 사용
            random_state=random_seed) #Hyperopt 결과 사용
            
            clf.fit(X_train,y_train )  

            y_pred = clf.predict(X_test)
            y_prod = clf.predict_proba(X_test)[:, 1]
            person_df = pd.DataFrame({"y_pred": y_pred, "y_prod": y_prod, "y_true": y_test, "pid": [pid] * len(y_test)})


            person_metric_df = generate_metrics(person_df)
            result_df = pd.concat([result_df, person_metric_df], axis=0)

    mean_result_df = result_df.groupby(['pid']).mean().reset_index()
    mean_metrics = mean_result_df.mean(axis=0)
    mean_metrics_std = mean_result_df.std(axis=0).round(3)
    mean_metrics['pid'] = 'all'
    mean_metrics_std['pid'] = 'all_std'
    mean_result_df  = pd.concat([mean_result_df , pd.DataFrame(mean_metrics).T,pd.DataFrame(mean_metrics_std).T], ignore_index=True)

    return mean_result_df, x_feature_df




def DecisionTree_Personalized(df: pd.DataFrame,numeric_feature:list,experiment_num: int, random_seed: int,use_smote: bool, result_path: str):
    result_df = pd.DataFrame()
    x_feature_df = pd.DataFrame(columns=['experiment_num', 'selected_features'])
    feature_importance_df = pd.DataFrame(columns=['pid', 'abs_feature', 'abs_value', 'pos_feature', 'pos_value', 'neg_feature', 'neg_value'])

    for exp_num in range(experiment_num):
        for pid in df['pnum'].unique():
            print('start_pid: ',pid)
            pnum_data = df[df['pnum'] == pid].sort_values('start_second').drop(['start_second'],axis=1)

            all_SHAP_values = []
            all_data = []

            # Train / test split
            split_index = int(len(pnum_data) * 0.5)
            train_df = pnum_data.iloc[:split_index]
            test_df = pnum_data.iloc[split_index:]

            if len(test_df['surface_acting'].unique())< 2:
                print(f"pnum {pid} has only one class.")
                continue

            # Average label encoding
            surface_acting_mean = train_df['surface_acting'].mean()
            train_df['surface_acting'] = train_df['surface_acting'].apply(lambda x: 1 if x > surface_acting_mean else 0)
            test_df['surface_acting'] = test_df['surface_acting'].apply(lambda x: 1 if x > surface_acting_mean else 0)

            # Normalization
            scaler = StandardScaler()
            train_df[numeric_feature] = scaler.fit_transform(train_df[numeric_feature])
            test_df[numeric_feature] = scaler.transform(test_df[numeric_feature])

            # Feature selection
            variance_threshold = 1e-07 # zero variance (basic_feature 제외) / audio feature도 3개 이미 빠짐
            variance_columns = list(train_df.loc[:, train_df.var() > variance_threshold].columns)
            train_df = train_df[variance_columns]
            test_df = test_df[variance_columns]

            # Feature selection
            x_feature = select_one_high_correlation_column(train_df.drop(['surface_acting'],axis=1),train_df['surface_acting'],0.8) # pairewise
            train_df = train_df[x_feature + ['surface_acting']]
            test_df = test_df[x_feature + ['surface_acting']]

            x_feature, alpha = select_features_lasso_personalized(
                train_df.drop(['surface_acting'], axis=1), 
                train_df['surface_acting'],
                k=2
            ) # lasso

            if len(x_feature)==0:
                print('no feature selection: ',pid)
                x_feature = list(train_df.drop(['surface_acting'], axis=1).columns)
                new_feature_row = pd.DataFrame({'pnum': [pid], 'experiment_num': [exp_num], 'selected_features': [[]]})
            else:
                new_feature_row = pd.DataFrame({'pnum': [pid], 'experiment_num': [exp_num], 'selected_features': [x_feature]})
                x_feature_df = pd.concat([x_feature_df, new_feature_row], ignore_index=True)

            X_train, y_train = train_df[x_feature], train_df['surface_acting']
            X_test, y_test = test_df[x_feature], test_df['surface_acting']

            # Data 증강
            if use_smote:
                X_train, y_train = apply_smote_person(X_train, y_train, pid, random_state=random_seed)

            clf = DecisionTreeClassifier(criterion='entropy', min_samples_split=20) # Referred to WESAD
            clf.fit(X_train,y_train )             

            y_pred = clf.predict(X_test)
            y_prod = clf.predict_proba(X_test)[:, 1]
            person_df = pd.DataFrame({"y_pred": y_pred, "y_prod": y_prod, "y_true": y_test, "pid": [pid] * len(y_test)})


            person_metric_df = generate_metrics(person_df)
            result_df = pd.concat([result_df, person_metric_df], axis=0)

    mean_result_df = result_df.groupby(['pid']).mean().reset_index()
    mean_metrics = mean_result_df.mean(axis=0)
    mean_metrics_std = mean_result_df.std(axis=0).round(3)
    mean_metrics['pid'] = 'all'
    mean_metrics_std['pid'] = 'all_std'
    mean_result_df  = pd.concat([mean_result_df , pd.DataFrame(mean_metrics).T,pd.DataFrame(mean_metrics_std).T], ignore_index=True)

    return mean_result_df, x_feature_df



def AdaBoost_Personalized(df: pd.DataFrame,numeric_feature:list,experiment_num: int, random_seed: int,use_smote: bool, result_path: str):
    result_df = pd.DataFrame()
    x_feature_df = pd.DataFrame(columns=['experiment_num', 'selected_features'])
    feature_importance_df = pd.DataFrame(columns=['pid', 'abs_feature', 'abs_value', 'pos_feature', 'pos_value', 'neg_feature', 'neg_value'])

    for exp_num in range(experiment_num):
        for pid in df['pnum'].unique():
            print('start_pid: ',pid)
            pnum_data = df[df['pnum'] == pid].sort_values('start_second').drop(['start_second'],axis=1)

            all_SHAP_values = []
            all_data = []

            # Train / test split
            split_index = int(len(pnum_data) * 0.5)
            train_df = pnum_data.iloc[:split_index]
            test_df = pnum_data.iloc[split_index:]

            if len(test_df['surface_acting'].unique())< 2:
                print(f"pnum {pid} has only one class.")
                continue

            # Average label encoding
            surface_acting_mean = train_df['surface_acting'].mean()
            train_df['surface_acting'] = train_df['surface_acting'].apply(lambda x: 1 if x > surface_acting_mean else 0)
            test_df['surface_acting'] = test_df['surface_acting'].apply(lambda x: 1 if x > surface_acting_mean else 0)

            # Normalization
            scaler = StandardScaler()
            train_df[numeric_feature] = scaler.fit_transform(train_df[numeric_feature])
            test_df[numeric_feature] = scaler.transform(test_df[numeric_feature])

            # Feature selection
            variance_threshold = 1e-07 # zero variance (basic_feature 제외) / audio feature도 3개 이미 빠짐
            variance_columns = list(train_df.loc[:, train_df.var() > variance_threshold].columns)
            train_df = train_df[variance_columns]
            test_df = test_df[variance_columns]

            # Feature selection
            x_feature = select_one_high_correlation_column(train_df.drop(['surface_acting'],axis=1),train_df['surface_acting'],0.8) # pairewise
            train_df = train_df[x_feature + ['surface_acting']]
            test_df = test_df[x_feature + ['surface_acting']]

            x_feature, alpha = select_features_lasso_personalized(
                train_df.drop(['surface_acting'], axis=1), 
                train_df['surface_acting'],
                k=3
            ) # lasso

            if len(x_feature)==0:
                print('no feature selection: ',pid)
                x_feature = list(train_df.drop(['surface_acting'], axis=1).columns)
                new_feature_row = pd.DataFrame({'pnum': [pid], 'experiment_num': [exp_num], 'selected_features': [[]]})
            else:
                new_feature_row = pd.DataFrame({'pnum': [pid], 'experiment_num': [exp_num], 'selected_features': [x_feature]})
                x_feature_df = pd.concat([x_feature_df, new_feature_row], ignore_index=True)

            X_train, y_train = train_df[x_feature], train_df['surface_acting']
            X_test, y_test = test_df[x_feature], test_df['surface_acting']

            # Data 증강
            if use_smote:
                X_train, y_train = apply_smote_person(X_train, y_train, pid, random_state=random_seed)

            dt = DecisionTreeClassifier(criterion='entropy', min_samples_split=20) # Referred to WESAD
            clf = AdaBoostClassifier(estimator=dt, n_estimators=100, algorithm='SAMME.R') # Referred to WESAD    
            clf.fit(X_train,y_train )           

            y_pred = clf.predict(X_test)
            y_prod = clf.predict_proba(X_test)[:, 1]
            person_df = pd.DataFrame({"y_pred": y_pred, "y_prod": y_prod, "y_true": y_test, "pid": [pid] * len(y_test)})


            person_metric_df = generate_metrics(person_df)
            result_df = pd.concat([result_df, person_metric_df], axis=0)

    mean_result_df = result_df.groupby(['pid']).mean().reset_index()
    mean_metrics = mean_result_df.mean(axis=0)
    mean_metrics_std = mean_result_df.std(axis=0).round(3)
    mean_metrics['pid'] = 'all'
    mean_metrics_std['pid'] = 'all_std'
    mean_result_df  = pd.concat([mean_result_df , pd.DataFrame(mean_metrics).T,pd.DataFrame(mean_metrics_std).T], ignore_index=True)

    return mean_result_df, x_feature_df



def LDA_Personalized(df: pd.DataFrame,numeric_feature:list,experiment_num: int, random_seed: int,use_smote: bool, result_path: str):
    result_df = pd.DataFrame()
    x_feature_df = pd.DataFrame(columns=['experiment_num', 'selected_features'])
    feature_importance_df = pd.DataFrame(columns=['pid', 'abs_feature', 'abs_value', 'pos_feature', 'pos_value', 'neg_feature', 'neg_value'])

    for exp_num in range(experiment_num):
        for pid in df['pnum'].unique():
            print('start_pid: ',pid)
            pnum_data = df[df['pnum'] == pid].sort_values('start_second').drop(['start_second'],axis=1)

            all_SHAP_values = []
            all_data = []

            # Train / test split
            split_index = int(len(pnum_data) * 0.5)
            train_df = pnum_data.iloc[:split_index]
            test_df = pnum_data.iloc[split_index:]

            if len(test_df['surface_acting'].unique())< 2:
                print(f"pnum {pid} has only one class.")
                continue

            # Average label encoding
            surface_acting_mean = train_df['surface_acting'].mean()
            train_df['surface_acting'] = train_df['surface_acting'].apply(lambda x: 1 if x > surface_acting_mean else 0)
            test_df['surface_acting'] = test_df['surface_acting'].apply(lambda x: 1 if x > surface_acting_mean else 0)

            # Normalization
            scaler = StandardScaler()
            train_df[numeric_feature] = scaler.fit_transform(train_df[numeric_feature])
            test_df[numeric_feature] = scaler.transform(test_df[numeric_feature])

            # Feature selection
            variance_threshold = 1e-07 # zero variance (basic_feature 제외) / audio feature도 3개 이미 빠짐
            variance_columns = list(train_df.loc[:, train_df.var() > variance_threshold].columns)
            train_df = train_df[variance_columns]
            test_df = test_df[variance_columns]

            # Feature selection
            x_feature = select_one_high_correlation_column(train_df.drop(['surface_acting'],axis=1),train_df['surface_acting'],0.8) # pairewise
            train_df = train_df[x_feature + ['surface_acting']]
            test_df = test_df[x_feature + ['surface_acting']]

            x_feature, alpha = select_features_lasso_personalized(
                train_df.drop(['surface_acting'], axis=1), 
                train_df['surface_acting'],
                k=3
            ) # lasso

            if len(x_feature)==0:
                print('no feature selection: ',pid)
                x_feature = list(train_df.drop(['surface_acting'], axis=1).columns)
                new_feature_row = pd.DataFrame({'pnum': [pid], 'experiment_num': [exp_num], 'selected_features': [[]]})
            else:
                new_feature_row = pd.DataFrame({'pnum': [pid], 'experiment_num': [exp_num], 'selected_features': [x_feature]})
                x_feature_df = pd.concat([x_feature_df, new_feature_row], ignore_index=True)

            X_train, y_train = train_df[x_feature], train_df['surface_acting']
            X_test, y_test = test_df[x_feature], test_df['surface_acting']

            # Data 증강
            if use_smote:
                X_train, y_train = apply_smote_person(X_train, y_train, pid, random_state=random_seed)

            clf = LinearDiscriminantAnalysis()
            clf.fit(X_train,y_train )        

            y_pred = clf.predict(X_test)
            y_prod = clf.predict_proba(X_test)[:, 1]
            person_df = pd.DataFrame({"y_pred": y_pred, "y_prod": y_prod, "y_true": y_test, "pid": [pid] * len(y_test)})


            person_metric_df = generate_metrics(person_df)
            result_df = pd.concat([result_df, person_metric_df], axis=0)

    mean_result_df = result_df.groupby(['pid']).mean().reset_index()
    mean_metrics = mean_result_df.mean(axis=0)
    mean_metrics_std = mean_result_df.std(axis=0).round(3)
    mean_metrics['pid'] = 'all'
    mean_metrics_std['pid'] = 'all_std'
    mean_result_df  = pd.concat([mean_result_df , pd.DataFrame(mean_metrics).T,pd.DataFrame(mean_metrics_std).T], ignore_index=True)

    return mean_result_df, x_feature_df



def KNN_Personalized(df: pd.DataFrame,numeric_feature:list,experiment_num: int, random_seed: int,use_smote: bool, result_path: str):
    result_df = pd.DataFrame()
    x_feature_df = pd.DataFrame(columns=['experiment_num', 'selected_features'])
    feature_importance_df = pd.DataFrame(columns=['pid', 'abs_feature', 'abs_value', 'pos_feature', 'pos_value', 'neg_feature', 'neg_value'])

    for exp_num in range(experiment_num):
        for pid in df['pnum'].unique():
            print('start_pid: ',pid)
            pnum_data = df[df['pnum'] == pid].sort_values('start_second').drop(['start_second'],axis=1)

            all_SHAP_values = []
            all_data = []

            # Train / test split
            split_index = int(len(pnum_data) * 0.5)
            train_df = pnum_data.iloc[:split_index]
            test_df = pnum_data.iloc[split_index:]

            if len(test_df['surface_acting'].unique())< 2:
                print(f"pnum {pid} has only one class.")
                continue

            # Average label encoding
            surface_acting_mean = train_df['surface_acting'].mean()
            train_df['surface_acting'] = train_df['surface_acting'].apply(lambda x: 1 if x > surface_acting_mean else 0)
            test_df['surface_acting'] = test_df['surface_acting'].apply(lambda x: 1 if x > surface_acting_mean else 0)

            # Normalization
            scaler = StandardScaler()
            train_df[numeric_feature] = scaler.fit_transform(train_df[numeric_feature])
            test_df[numeric_feature] = scaler.transform(test_df[numeric_feature])

            # Feature selection
            variance_threshold = 1e-07 # zero variance (basic_feature 제외) / audio feature도 3개 이미 빠짐
            variance_columns = list(train_df.loc[:, train_df.var() > variance_threshold].columns)
            train_df = train_df[variance_columns]
            test_df = test_df[variance_columns]

            # Feature selection
            x_feature = select_one_high_correlation_column(train_df.drop(['surface_acting'],axis=1),train_df['surface_acting'],0.8) # pairewise
            train_df = train_df[x_feature + ['surface_acting']]
            test_df = test_df[x_feature + ['surface_acting']]

            x_feature, alpha = select_features_lasso_personalized(
                train_df.drop(['surface_acting'], axis=1), 
                train_df['surface_acting'],
                k=3
            ) # lasso

            if len(x_feature)==0:
                print('no feature selection: ',pid)
                x_feature = list(train_df.drop(['surface_acting'], axis=1).columns)
                new_feature_row = pd.DataFrame({'pnum': [pid], 'experiment_num': [exp_num], 'selected_features': [[]]})
            else:
                new_feature_row = pd.DataFrame({'pnum': [pid], 'experiment_num': [exp_num], 'selected_features': [x_feature]})
                x_feature_df = pd.concat([x_feature_df, new_feature_row], ignore_index=True)

            X_train, y_train = train_df[x_feature], train_df['surface_acting']
            X_test, y_test = test_df[x_feature], test_df['surface_acting']

            # Data 증강
            if use_smote:
                X_train, y_train = apply_smote_person(X_train, y_train, pid, random_state=random_seed)

            clf = KNeighborsClassifier(n_neighbors=9) # Referred to WESAD
            clf.fit(X_train,y_train )  

            y_pred = clf.predict(X_test)
            y_prod = clf.predict_proba(X_test)[:, 1]
            person_df = pd.DataFrame({"y_pred": y_pred, "y_prod": y_prod, "y_true": y_test, "pid": [pid] * len(y_test)})


            person_metric_df = generate_metrics(person_df)
            result_df = pd.concat([result_df, person_metric_df], axis=0)

    mean_result_df = result_df.groupby(['pid']).mean().reset_index()
    mean_metrics = mean_result_df.mean(axis=0)
    mean_metrics_std = mean_result_df.std(axis=0).round(3)
    mean_metrics['pid'] = 'all'
    mean_metrics_std['pid'] = 'all_std'
    mean_result_df  = pd.concat([mean_result_df , pd.DataFrame(mean_metrics).T,pd.DataFrame(mean_metrics_std).T], ignore_index=True)

    return mean_result_df, x_feature_df



def SVM_Personalized(df: pd.DataFrame,numeric_feature:list,experiment_num: int, random_seed: int,use_smote: bool, result_path: str):
    result_df = pd.DataFrame()
    x_feature_df = pd.DataFrame(columns=['experiment_num', 'selected_features'])
    feature_importance_df = pd.DataFrame(columns=['pid', 'abs_feature', 'abs_value', 'pos_feature', 'pos_value', 'neg_feature', 'neg_value'])

    for exp_num in range(experiment_num):
        for pid in df['pnum'].unique():
            print('start_pid: ',pid)
            pnum_data = df[df['pnum'] == pid].sort_values('start_second').drop(['start_second'],axis=1)

            all_SHAP_values = []
            all_data = []

            # Train / test split
            split_index = int(len(pnum_data) * 0.5)
            train_df = pnum_data.iloc[:split_index]
            test_df = pnum_data.iloc[split_index:]

            if len(test_df['surface_acting'].unique())< 2:
                print(f"pnum {pid} has only one class.")
                continue

            # Average label encoding
            surface_acting_mean = train_df['surface_acting'].mean()
            train_df['surface_acting'] = train_df['surface_acting'].apply(lambda x: 1 if x > surface_acting_mean else 0)
            test_df['surface_acting'] = test_df['surface_acting'].apply(lambda x: 1 if x > surface_acting_mean else 0)

            # Normalization
            scaler = StandardScaler()
            train_df[numeric_feature] = scaler.fit_transform(train_df[numeric_feature])
            test_df[numeric_feature] = scaler.transform(test_df[numeric_feature])

            # Feature selection
            variance_threshold = 1e-07 # zero variance (basic_feature 제외) / audio feature도 3개 이미 빠짐
            variance_columns = list(train_df.loc[:, train_df.var() > variance_threshold].columns)
            train_df = train_df[variance_columns]
            test_df = test_df[variance_columns]

            # Feature selection
            x_feature = select_one_high_correlation_column(train_df.drop(['surface_acting'],axis=1),train_df['surface_acting'],0.8) # pairewise
            train_df = train_df[x_feature + ['surface_acting']]
            test_df = test_df[x_feature + ['surface_acting']]

            x_feature, alpha = select_features_lasso_personalized(
                train_df.drop(['surface_acting'], axis=1), 
                train_df['surface_acting'],
                k=3
            ) # lasso

            if len(x_feature)==0:
                print('no feature selection: ',pid)
                x_feature = list(train_df.drop(['surface_acting'], axis=1).columns)
                new_feature_row = pd.DataFrame({'pnum': [pid], 'experiment_num': [exp_num], 'selected_features': [[]]})
            else:
                new_feature_row = pd.DataFrame({'pnum': [pid], 'experiment_num': [exp_num], 'selected_features': [x_feature]})
                x_feature_df = pd.concat([x_feature_df, new_feature_row], ignore_index=True)

            X_train, y_train = train_df[x_feature], train_df['surface_acting']
            X_test, y_test = test_df[x_feature], test_df['surface_acting']

            # Data 증강
            if use_smote:
                X_train, y_train = apply_smote_person(X_train, y_train, pid, random_state=random_seed)

            clf = SVC(probability=True, kernel='rbf', C=1, gamma=0.03) # Referred to paper https://dl.acm.org/doi/pdf/10.1145/3173574.3174165
            clf.fit(X_train,y_train )

            y_pred = clf.predict(X_test)
            y_prod = clf.predict_proba(X_test)[:, 1]
            person_df = pd.DataFrame({"y_pred": y_pred, "y_prod": y_prod, "y_true": y_test, "pid": [pid] * len(y_test)})

            person_metric_df = generate_metrics(person_df)
            result_df = pd.concat([result_df, person_metric_df], axis=0)

    mean_result_df = result_df.groupby(['pid']).mean().reset_index()
    mean_metrics = mean_result_df.mean(axis=0)
    mean_metrics_std = mean_result_df.std(axis=0).round(3)
    mean_metrics['pid'] = 'all'
    mean_metrics_std['pid'] = 'all_std'
    mean_result_df  = pd.concat([mean_result_df , pd.DataFrame(mean_metrics).T,pd.DataFrame(mean_metrics_std).T], ignore_index=True)

    return mean_result_df, x_feature_df

# from imblearn.under_sampling import RandomUnderSampler
# import matplotlib.pyplot as plt
# import matplotlib
# import matplotlib.font_manager as fm

# font_path = '/usr/share/fonts/truetype/nanum/NanumGothicCodingBold.ttf'
# font_prop = fm.FontProperties(fname=font_path)

# # matplotlib에 폰트 설정
# plt.rcParams['font.family'] = font_prop.get_name()
# plt.rcParams['axes.unicode_minus'] = False  # 마이너스 부호 깨짐 방지

# Hybrid
def RandomForest_Hybrid(df: pd.DataFrame,numeric_feature:list,experiment_num: int, random_seed: int,use_smote: bool, result_path: str):
    result_df = pd.DataFrame()
    x_feature_df = pd.DataFrame(columns=['experiment_num', 'selected_features'])
    feature_importance_df = pd.DataFrame(columns=['pid', 'abs_feature', 'abs_value', 'pos_feature', 'pos_value', 'neg_feature', 'neg_value'])

    df_ = df.sort_values(['pnum','start_second']).drop(['start_second'],axis=1)
    for exp_num in range(experiment_num):
                
        all_SHAP_values =[]
        all_data = []
        all_selected_features = list(df_.drop(['pnum','surface_acting'],axis=1).columns)

        for pid in df['pnum'].unique():
            print('start_pid: ',pid)

            all_SHAP_values = []
            all_data = []

            # Train / test split
            train_df = df_.loc[df_['pnum'] !=pid]
            test_df =  df_.loc[df_['pnum'] ==pid]

            split_index = int(len(test_df) * 0.5)
            train_df_target_pid = test_df.iloc[:split_index]
            
            test_df = test_df.iloc[split_index:]
            train_df = pd.concat([train_df,train_df_target_pid],axis=0)

            if len(test_df['surface_acting'].unique())< 2:
                print(f"pnum {pid} has only one class.")
                continue

            # Average label encoding
            target_thred = train_df.query('pnum == @pid')['surface_acting'].mean()
            train_df = average_based_binary_encoding(train_df,'surface_acting')
            test_df['surface_acting'] = test_df['surface_acting'].apply(lambda x: 1 if x > target_thred else 0)

            # Normalization
            means = train_df.query('pnum == @pid')[numeric_feature].mean()
            stds = train_df.query('pnum == @pid')[numeric_feature].std()

            if len(numeric_feature) > 0:

                train_df = normalization(train_df, 'z_norm', numeric_feature)
                test_df[numeric_feature] = (test_df[numeric_feature] - means) / stds

                # Normalization이 안되는 column 삭제하기 ==> Train에 필요 없기 때문에
                nan_columns_train = train_df.columns[(np.isinf(train_df) | np.isnan(train_df)).any()].tolist()
                nan_columns_test =  test_df.columns[(np.isinf(test_df) | np.isnan(test_df)).any()].tolist()
                print("normalization이 안되는 column",nan_columns_train,nan_columns_test)

                nan_columns = nan_columns_train + nan_columns_test
                train_df.drop(columns=nan_columns, inplace=True)
                test_df.drop(columns=nan_columns, inplace=True)
            else:
                pass

            # Feature selection
            variance_threshold = 1e-07 # zero variance (basic_feature 제외) / audio feature도 3개 이미 빠짐
            variance_columns = list(train_df.loc[:, train_df.var() > variance_threshold].columns)
            train_df = train_df[variance_columns]
            test_df = test_df[variance_columns]

            x_feature = select_one_high_correlation_column(train_df.drop(['surface_acting'],axis=1),train_df['surface_acting'],0.8) # pairewise
            train_df = train_df[x_feature + ['surface_acting']]
            test_df = test_df[x_feature + ['surface_acting']]

            x_feature, alpha = select_features_losso(
                train_df.drop(['pnum','surface_acting'], axis=1), 
                train_df['surface_acting'],
                train_df['pnum']
            ) # lasso

            if len(x_feature)==0:
                print('no feature selection: ',pid)
                x_feature = list(train_df.drop(['surface_acting'], axis=1).columns)
                new_feature_row = pd.DataFrame({'pnum': [pid], 'experiment_num': [exp_num], 'selected_features': [[]]})
            else:
                new_feature_row = pd.DataFrame({'pnum': [pid], 'experiment_num': [exp_num], 'selected_features': [x_feature]})
                x_feature_df = pd.concat([x_feature_df, new_feature_row], ignore_index=True)

            # # Data under sampling
            # train_df_exclude_targe = train_df.query('pnum!=@pid')
            # undersample = RandomUnderSampler(sampling_strategy=0.67, random_state=42)
            # X_resampled, y_resampled = undersample.fit_resample(train_df_exclude_targe.drop(['surface_acting'],axis=1), train_df_exclude_targe['surface_acting'])
            # train_df = pd.concat([X_resampled,y_resampled],axis=1)

            X_train, y_train = train_df[x_feature], train_df['surface_acting']
            X_test, y_test = test_df[x_feature], test_df['surface_acting']

            # Data 증강
            if use_smote:
                X_train, y_train = apply_smote_person(X_train, y_train, pid, random_state=random_seed)

            # clf = RandomForestClassifier(criterion='gini', min_samples_split=20, n_estimators=100)
            clf = RandomForestClassifier(n_estimators=500) # Referred to paper https://dl.acm.org/doi/pdf/10.1145/3173574.3174165
            clf.fit(X_train, y_train)


            explainer = shap.TreeExplainer(clf)
            shap_values = explainer(X_test) # output_margin=True 추가  -> 오류 나서 다시 생략
            # print(shap_values)


            shap_values_class = shap_values.values[:,:, 1]  # 첫 번째 클래스 선택
            data_class = shap_values.data  # 첫 번째 클래스 선택


            # 선택된 feature에 대해 SHAP 값을 채우고 없는 값은 0으로 패딩
            shap_values_padded = np.zeros((len(shap_values_class), len(all_selected_features)))
            shap_values_padded[:, [all_selected_features.index(f) for f in x_feature]] = shap_values_class
            all_SHAP_values.append(shap_values_padded)


            data_padded = np.zeros((len(data_class), len(all_selected_features)))
            data_padded[:, [all_selected_features.index(f) for f in x_feature]] = data_class
            all_data.append(data_padded)

            all_SHAP_values_array = np.concatenate(all_SHAP_values, axis=0) ## for sharply value 
            all_data_array = np.concatenate(all_data, axis=0) ## for feature high low


            y_pred = clf.predict(X_test)  
            y_prod = clf.predict_proba(X_test)[:, 1]
            person_df = pd.DataFrame({"y_pred": y_pred, "y_prod": y_prod, "y_true": y_test, "pid": [pid] * len(y_test)})


            # from sklearn.metrics import roc_curve

            # # roc_auc curve
            # fpr,tpr, thresholds = roc_curve(y_test,y_prod)
            # plt.figure(figsize=(15,5))
            # plt.figure(figsize=(15,5))
            # plt.plot([0,1],[0,1],label='STR')
            # plt.plot(fpr,tpr,label='ROC')
            # plt.title(f'pid : {pid}')
            # plt.xlabel('FPR')
            # plt.ylabel('TPR')
            # plt.legend()
            # plt.grid()
            # plt.show()

            person_metric_df = generate_metrics(person_df)
            result_df = pd.concat([result_df, person_metric_df], axis=0)
    
        #SHAP value graph
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        font_path = '/home/iclab/HJ/font/NanumGothic.ttf'
        font_prop = font_manager.FontProperties(fname=font_path)

        print()
        shap.summary_plot(all_SHAP_values_array,all_data_array, feature_names=all_selected_features, plot_type='dot', max_display=20,show=False)
        fig = plt.gcf()  # 현재 플롯 객체 가져오기
        for text in fig.findobj(plt.Text):  # 모든 텍스트 객체에 대해
            text.set_fontproperties(font_prop)
        plt.savefig('{0}/RF_shap_plot.png'.format(result_path))
        plt.show() 

    mean_result_df = result_df.groupby(['pid']).mean().reset_index()
    mean_metrics = mean_result_df.mean(axis=0)
    mean_metrics_std = mean_result_df.std(axis=0).round(2)
    mean_metrics['pid'] = 'all'
    mean_metrics_std['pid'] = 'all_std'
    mean_result_df  = pd.concat([mean_result_df , pd.DataFrame(mean_metrics).T,pd.DataFrame(mean_metrics_std).T], ignore_index=True)

    return mean_result_df, x_feature_df



import matplotlib.pyplot as plt
plt.rc('font', family='NanumGothic')

def XGBoost_Hybrid(df: pd.DataFrame,numeric_feature:list,experiment_num: int, random_seed: int,use_smote: bool, result_path: str):
    result_df = pd.DataFrame()
    x_feature_df = pd.DataFrame(columns=['experiment_num', 'selected_features'])
    feature_importance_df = pd.DataFrame(columns=['pid', 'abs_feature', 'abs_value', 'pos_feature', 'pos_value', 'neg_feature', 'neg_value'])

    df_ = df.sort_values(['pnum','start_second']).drop(['start_second'],axis=1)
    for exp_num in range(experiment_num):
        for pid in df['pnum'].unique():
            print('start_pid: ',pid)

            all_SHAP_values = []
            all_data = []

            # Train / test split
            train_df = df_.loc[df_['pnum'] !=pid]
            test_df =  df_.loc[df_['pnum'] ==pid]

            split_index = int(len(test_df) * 0.5)
            train_df_target_pid = test_df.iloc[:split_index]
            
            test_df = test_df.iloc[split_index:]
            train_df = pd.concat([train_df,train_df_target_pid],axis=0)

            if len(test_df['surface_acting'].unique())< 2:
                print(f"pnum {pid} has only one class.")
                continue

            # Average label encoding
            target_thred = train_df.query('pnum == @pid')['surface_acting'].mean()
            train_df = average_based_binary_encoding(train_df,'surface_acting')
            test_df['surface_acting'] = test_df['surface_acting'].apply(lambda x: 1 if x > target_thred else 0)

            if len(numeric_feature) > 0:
                # Normalization
                means = train_df.query('pnum == @pid')[numeric_feature].mean()
                stds = train_df.query('pnum == @pid')[numeric_feature].std()

                train_df = normalization(train_df, 'z_norm', numeric_feature)
                test_df[numeric_feature] = (test_df[numeric_feature] - means) / stds

                # Normalization이 안되는 column 삭제하기
                nan_columns_train = train_df.columns[(np.isinf(train_df) | np.isnan(train_df)).any()].tolist()
                nan_columns_test =  test_df.columns[(np.isinf(test_df) | np.isnan(test_df)).any()].tolist()
                print("normalization이 안되는 column",nan_columns_train,nan_columns_test)

                nan_columns = nan_columns_train + nan_columns_test
                train_df.drop(columns=nan_columns, inplace=True)
                test_df.drop(columns=nan_columns, inplace=True)
            else:
                pass


            # Feature selection
            variance_threshold = 1e-07 # zero variance (basic_feature 제외) / audio feature도 3개 이미 빠짐
            variance_columns = list(train_df.loc[:, train_df.var() > variance_threshold].columns)
            train_df = train_df[variance_columns]
            test_df = test_df[variance_columns]

            x_feature = select_one_high_correlation_column(train_df.drop(['surface_acting'],axis=1),train_df['surface_acting'],0.8) # pairewise
            train_df = train_df[x_feature + ['surface_acting']]
            test_df = test_df[x_feature + ['surface_acting']]

            x_feature, alpha = select_features_losso(
                train_df.drop(['pnum','surface_acting'], axis=1), 
                train_df['surface_acting'],
                train_df['pnum']
            ) # lasso

            if len(x_feature)==0:
                print('no feature selection: ',pid)
                x_feature = list(train_df.drop(['surface_acting'], axis=1).columns)
                new_feature_row = pd.DataFrame({'pnum': [pid], 'experiment_num': [exp_num], 'selected_features': [[]]})
            else:
                new_feature_row = pd.DataFrame({'pnum': [pid], 'experiment_num': [exp_num], 'selected_features': [x_feature]})
                x_feature_df = pd.concat([x_feature_df, new_feature_row], ignore_index=True)

            X_train, y_train = train_df[x_feature], train_df['surface_acting']
            X_test, y_test = test_df[x_feature], test_df['surface_acting']

            # Data 증강
            if use_smote:
                X_train, y_train = apply_smote_person(X_train, y_train, pid, random_state=random_seed)

            clf = XGBClassifier(objective='binary:logistic', max_depth=3, learning_rate=0.1, n_estimators=100,
            tree_method='gpu_hist',  # GPU 사용
            predictor='gpu_predictor',  # GPU 예측기 사용
            random_state=random_seed) #Hyperopt 결과 사용


# # {'colsample_bytree': 0.9189474294432578, 'learning_rate': 0.09313128256061605, 'max_depth': 10.0, 'min_child_weight': 8.0, 'n_estimators': 130.0, 'scale_pos_weight': 3.2939863428010874, 'subsample': 0.7566006866828496}
#             clf  = XGBClassifier(colsample_bytree = 0.9189474294432578,
#                                  learning_rate = 0.09313128256061605,
#                                  max_depth = 10,
#                                  min_child_weight = 8,
#                                  n_estimators  = 130,
#                                  scale_pos_weight = 3.2939863428010874,
#                                  subsample = 0.7566006866828496)

            clf.fit(X_train, y_train )

            y_pred = clf.predict(X_test)
            y_prod = clf.predict_proba(X_test)[:, 1]
            person_df = pd.DataFrame({"y_pred": y_pred, "y_prod": y_prod, "y_true": y_test, "pid": [pid] * len(y_test)})

            person_metric_df = generate_metrics(person_df)
            result_df = pd.concat([result_df, person_metric_df], axis=0)

    mean_result_df = result_df.groupby(['pid']).mean().reset_index()
    mean_metrics = mean_result_df.mean(axis=0)
    mean_metrics_std = mean_result_df.std(axis=0).round(2)
    mean_metrics['pid'] = 'all'
    mean_metrics_std['pid'] = 'all_std'
    mean_result_df  = pd.concat([mean_result_df , pd.DataFrame(mean_metrics).T,pd.DataFrame(mean_metrics_std).T], ignore_index=True)

    return mean_result_df, x_feature_df



def DecisionTree_Hybrid(df: pd.DataFrame,numeric_feature:list,experiment_num: int, random_seed: int,use_smote: bool, result_path: str):
    result_df = pd.DataFrame()
    x_feature_df = pd.DataFrame(columns=['experiment_num', 'selected_features'])
    feature_importance_df = pd.DataFrame(columns=['pid', 'abs_feature', 'abs_value', 'pos_feature', 'pos_value', 'neg_feature', 'neg_value'])

    df_ = df.sort_values(['pnum','start_second']).drop(['start_second'],axis=1)
    for exp_num in range(experiment_num):
        for pid in df['pnum'].unique():
            print('start_pid: ',pid)

            all_SHAP_values = []
            all_data = []

            # Train / test split
            train_df = df_.loc[df_['pnum'] !=pid]
            test_df =  df_.loc[df_['pnum'] ==pid]

            split_index = int(len(test_df) * 0.5)
            train_df_target_pid = test_df.iloc[:split_index]
            
            test_df = test_df.iloc[split_index:]
            train_df = pd.concat([train_df,train_df_target_pid],axis=0)

            if len(test_df['surface_acting'].unique())< 2:
                print(f"pnum {pid} has only one class.")
                continue

            # Average label encoding
            target_thred = train_df.query('pnum == @pid')['surface_acting'].mean()
            train_df = average_based_binary_encoding(train_df,'surface_acting')
            test_df['surface_acting'] = test_df['surface_acting'].apply(lambda x: 1 if x > target_thred else 0)

            # Normalization
            means = train_df.query('pnum == @pid')[numeric_feature].mean()
            stds = train_df.query('pnum == @pid')[numeric_feature].std()

            train_df = normalization(train_df, 'z_norm', numeric_feature)
            test_df[numeric_feature] = (test_df[numeric_feature] - means) / stds

            # Normalization이 안되는 column 삭제하기
            nan_columns_train = train_df.columns[np.isinf(train_df).any()].tolist()
            nan_columns_test = test_df.columns[np.isinf(test_df).any()].tolist()
            print("normalization이 안되는 column",nan_columns_train,nan_columns_test)

            nan_columns = nan_columns_train + nan_columns_test
            train_df.drop(columns=nan_columns, inplace=True)
            test_df.drop(columns=nan_columns, inplace=True)


            # Feature selection
            variance_threshold = 1e-07 # zero variance (basic_feature 제외) / audio feature도 3개 이미 빠짐
            variance_columns = list(train_df.loc[:, train_df.var() > variance_threshold].columns)
            train_df = train_df[variance_columns]
            test_df = test_df[variance_columns]

            x_feature = select_one_high_correlation_column(train_df.drop(['surface_acting'],axis=1),train_df['surface_acting'],0.8) # pairewise
            train_df = train_df[x_feature + ['surface_acting']]
            test_df = test_df[x_feature + ['surface_acting']]

            x_feature, alpha = select_features_losso(
                train_df.drop(['pnum','surface_acting'], axis=1), 
                train_df['surface_acting'],
                train_df['pnum']
            ) # lasso

            if len(x_feature)==0:
                print('no feature selection: ',pid)
                x_feature = list(train_df.drop(['surface_acting'], axis=1).columns)
                new_feature_row = pd.DataFrame({'pnum': [pid], 'experiment_num': [exp_num], 'selected_features': [[]]})
            else:
                new_feature_row = pd.DataFrame({'pnum': [pid], 'experiment_num': [exp_num], 'selected_features': [x_feature]})
                x_feature_df = pd.concat([x_feature_df, new_feature_row], ignore_index=True)

            X_train, y_train = train_df[x_feature], train_df['surface_acting']
            X_test, y_test = test_df[x_feature], test_df['surface_acting']

            # Data 증강
            if use_smote:
                X_train, y_train = apply_smote_person(X_train, y_train, pid, random_state=random_seed)


            clf = DecisionTreeClassifier(criterion='entropy', min_samples_split=20) # Referred to WESAD
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            y_prod = clf.predict_proba(X_test)[:, 1]
            person_df = pd.DataFrame({"y_pred": y_pred, "y_prod": y_prod, "y_true": y_test, "pid": [pid] * len(y_test)})

            person_metric_df = generate_metrics(person_df)
            result_df = pd.concat([result_df, person_metric_df], axis=0)

    mean_result_df = result_df.groupby(['pid']).mean().reset_index()
    mean_metrics = mean_result_df.mean(axis=0)
    mean_metrics_std = mean_result_df.std(axis=0).round(2)
    mean_metrics['pid'] = 'all'
    mean_metrics_std['pid'] = 'all_std'
    mean_result_df  = pd.concat([mean_result_df , pd.DataFrame(mean_metrics).T,pd.DataFrame(mean_metrics_std).T], ignore_index=True)

    return mean_result_df, x_feature_df



def AdaBoost_Hybrid(df: pd.DataFrame,numeric_feature:list,experiment_num: int, random_seed: int,use_smote: bool, result_path: str):
    result_df = pd.DataFrame()
    x_feature_df = pd.DataFrame(columns=['experiment_num', 'selected_features'])
    feature_importance_df = pd.DataFrame(columns=['pid', 'abs_feature', 'abs_value', 'pos_feature', 'pos_value', 'neg_feature', 'neg_value'])

    df_ = df.sort_values(['pnum','start_second']).drop(['start_second'],axis=1)
    for exp_num in range(experiment_num):
        for pid in df['pnum'].unique():
            print('start_pid: ',pid)

            all_SHAP_values = []
            all_data = []

            # Train / test split
            train_df = df_.loc[df_['pnum'] !=pid]
            test_df =  df_.loc[df_['pnum'] ==pid]

            split_index = int(len(test_df) * 0.5)
            train_df_target_pid = test_df.iloc[:split_index]
            
            test_df = test_df.iloc[split_index:]
            train_df = pd.concat([train_df,train_df_target_pid],axis=0)

            if len(test_df['surface_acting'].unique())< 2:
                print(f"pnum {pid} has only one class.")
                continue

            # Average label encoding
            target_thred = train_df.query('pnum == @pid')['surface_acting'].mean()
            train_df = average_based_binary_encoding(train_df,'surface_acting')
            test_df['surface_acting'] = test_df['surface_acting'].apply(lambda x: 1 if x > target_thred else 0)

            # Normalization
            means = train_df.query('pnum == @pid')[numeric_feature].mean()
            stds = train_df.query('pnum == @pid')[numeric_feature].std()

            train_df = normalization(train_df, 'z_norm', numeric_feature)
            test_df[numeric_feature] = (test_df[numeric_feature] - means) / stds

            # Normalization이 안되는 column 삭제하기
            nan_columns_train = train_df.columns[(np.isinf(train_df) | np.isnan(train_df)).any()].tolist()
            nan_columns_test =  test_df.columns[(np.isinf(test_df) | np.isnan(test_df)).any()].tolist()
            print("normalization이 안되는 column",nan_columns_train,nan_columns_test)

            nan_columns = nan_columns_train + nan_columns_test
            train_df.drop(columns=nan_columns, inplace=True)
            test_df.drop(columns=nan_columns, inplace=True)

            # Feature selection
            variance_threshold = 1e-07 # zero variance (basic_feature 제외) / audio feature도 3개 이미 빠짐
            variance_columns = list(train_df.loc[:, train_df.var() > variance_threshold].columns)
            train_df = train_df[variance_columns]
            test_df = test_df[variance_columns]

            x_feature = select_one_high_correlation_column(train_df.drop(['surface_acting'],axis=1),train_df['surface_acting'],0.8) # pairewise
            train_df = train_df[x_feature + ['surface_acting']]
            test_df = test_df[x_feature + ['surface_acting']]

            x_feature, alpha = select_features_losso(
                train_df.drop(['pnum','surface_acting'], axis=1), 
                train_df['surface_acting'],
                train_df['pnum']
            ) # lasso

            if len(x_feature)==0:
                print('no feature selection: ',pid)
                x_feature = list(train_df.drop(['surface_acting'], axis=1).columns)
                new_feature_row = pd.DataFrame({'pnum': [pid], 'experiment_num': [exp_num], 'selected_features': [[]]})
            else:
                new_feature_row = pd.DataFrame({'pnum': [pid], 'experiment_num': [exp_num], 'selected_features': [x_feature]})
                x_feature_df = pd.concat([x_feature_df, new_feature_row], ignore_index=True)

            X_train, y_train = train_df[x_feature], train_df['surface_acting']
            X_test, y_test = test_df[x_feature], test_df['surface_acting']

            # Data 증강
            if use_smote:
                X_train, y_train = apply_smote_person(X_train, y_train, pid, random_state=random_seed)


            dt = DecisionTreeClassifier(criterion='entropy', min_samples_split=20) # Referred to WESAD
            clf = AdaBoostClassifier(estimator=dt, n_estimators=100, algorithm='SAMME.R') # Referred to WESAD    
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)
            y_prod = clf.predict_proba(X_test)[:, 1]
            person_df = pd.DataFrame({"y_pred": y_pred, "y_prod": y_prod, "y_true": y_test, "pid": [pid] * len(y_test)})

            person_metric_df = generate_metrics(person_df)
            result_df = pd.concat([result_df, person_metric_df], axis=0)

    mean_result_df = result_df.groupby(['pid']).mean().reset_index()
    mean_metrics = mean_result_df.mean(axis=0)
    mean_metrics_std = mean_result_df.std(axis=0).round(2)
    mean_metrics['pid'] = 'all'
    mean_metrics_std['pid'] = 'all_std'
    mean_result_df  = pd.concat([mean_result_df , pd.DataFrame(mean_metrics).T,pd.DataFrame(mean_metrics_std).T], ignore_index=True)

    return mean_result_df, x_feature_df




def LDA_Hybrid(df: pd.DataFrame,numeric_feature:list,experiment_num: int, random_seed: int,use_smote: bool, result_path: str):
    result_df = pd.DataFrame()
    x_feature_df = pd.DataFrame(columns=['experiment_num', 'selected_features'])
    feature_importance_df = pd.DataFrame(columns=['pid', 'abs_feature', 'abs_value', 'pos_feature', 'pos_value', 'neg_feature', 'neg_value'])

    df_ = df.sort_values(['pnum','start_second']).drop(['start_second'],axis=1)
    for exp_num in range(experiment_num):
        for pid in df['pnum'].unique():
            print('start_pid: ',pid)

            all_SHAP_values = []
            all_data = []

            # Train / test split
            train_df = df_.loc[df_['pnum'] !=pid]
            test_df =  df_.loc[df_['pnum'] ==pid]

            split_index = int(len(test_df) * 0.5)
            train_df_target_pid = test_df.iloc[:split_index]
            
            test_df = test_df.iloc[split_index:]
            train_df = pd.concat([train_df,train_df_target_pid],axis=0)

            if len(test_df['surface_acting'].unique())< 2:
                print(f"pnum {pid} has only one class.")
                continue

            # Average label encoding
            target_thred = train_df.query('pnum == @pid')['surface_acting'].mean()
            train_df = average_based_binary_encoding(train_df,'surface_acting')
            test_df['surface_acting'] = test_df['surface_acting'].apply(lambda x: 1 if x > target_thred else 0)

            # Normalization
            means = train_df.query('pnum == @pid')[numeric_feature].mean()
            stds = train_df.query('pnum == @pid')[numeric_feature].std()

            train_df = normalization(train_df, 'z_norm', numeric_feature)
            test_df[numeric_feature] = (test_df[numeric_feature] - means) / stds
            

            # Normalization이 안되는 column 삭제하기
            nan_columns_train = train_df.columns[(np.isinf(train_df) | np.isnan(train_df)).any()].tolist()
            nan_columns_test =  test_df.columns[(np.isinf(test_df) | np.isnan(test_df)).any()].tolist()
            print("normalization이 안되는 column",nan_columns_train,nan_columns_test)

            nan_columns = nan_columns_train + nan_columns_test
            train_df.drop(columns=nan_columns, inplace=True)
            test_df.drop(columns=nan_columns, inplace=True)


            # Feature selection
            variance_threshold = 1e-07 # zero variance (basic_feature 제외) / audio feature도 3개 이미 빠짐
            variance_columns = list(train_df.loc[:, train_df.var() > variance_threshold].columns)
            train_df = train_df[variance_columns]
            test_df = test_df[variance_columns]

            x_feature = select_one_high_correlation_column(train_df.drop(['surface_acting'],axis=1),train_df['surface_acting'],0.8) # pairewise
            train_df = train_df[x_feature + ['surface_acting']]
            test_df = test_df[x_feature + ['surface_acting']]

            x_feature, alpha = select_features_losso(
                train_df.drop(['pnum','surface_acting'], axis=1), 
                train_df['surface_acting'],
                train_df['pnum']
            ) # lasso

            if len(x_feature)==0:
                print('no feature selection: ',pid)
                x_feature = list(train_df.drop(['surface_acting'], axis=1).columns)
                new_feature_row = pd.DataFrame({'pnum': [pid], 'experiment_num': [exp_num], 'selected_features': [[]]})
            else:
                new_feature_row = pd.DataFrame({'pnum': [pid], 'experiment_num': [exp_num], 'selected_features': [x_feature]})
                x_feature_df = pd.concat([x_feature_df, new_feature_row], ignore_index=True)

            X_train, y_train = train_df[x_feature], train_df['surface_acting']
            X_test, y_test = test_df[x_feature], test_df['surface_acting']

            # Data 증강
            if use_smote:
                X_train, y_train = apply_smote_person(X_train, y_train, pid, random_state=random_seed)

            # missing_values = X_test.isnull().sum()

            # # 결측값이 있는 열만 출력
            # missing_columns = missing_values[missing_values > 0]
            # if missing_columns.empty:
            #     print("X_train에는 결측값이 없습니다.")
            # else:
            #     print(f"X_train에는 총 {missing_columns.sum()}개의 결측값이 있습니다.")
            #     print("결측값이 있는 열과 개수는 다음과 같습니다:")
            #     print(missing_columns)

            clf = LinearDiscriminantAnalysis()
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)
            y_prod = clf.predict_proba(X_test)[:, 1]
            person_df = pd.DataFrame({"y_pred": y_pred, "y_prod": y_prod, "y_true": y_test, "pid": [pid] * len(y_test)})

            person_metric_df = generate_metrics(person_df)
            result_df = pd.concat([result_df, person_metric_df], axis=0)

    mean_result_df = result_df.groupby(['pid']).mean().reset_index()
    mean_metrics = mean_result_df.mean(axis=0)
    mean_metrics_std = mean_result_df.std(axis=0).round(2)
    mean_metrics['pid'] = 'all'
    mean_metrics_std['pid'] = 'all_std'
    mean_result_df  = pd.concat([mean_result_df , pd.DataFrame(mean_metrics).T,pd.DataFrame(mean_metrics_std).T], ignore_index=True)

    return mean_result_df, x_feature_df



def KNN_Hybrid(df: pd.DataFrame,numeric_feature:list,experiment_num: int, random_seed: int,use_smote: bool, result_path: str):
    result_df = pd.DataFrame()
    x_feature_df = pd.DataFrame(columns=['experiment_num', 'selected_features'])
    feature_importance_df = pd.DataFrame(columns=['pid', 'abs_feature', 'abs_value', 'pos_feature', 'pos_value', 'neg_feature', 'neg_value'])

    df_ = df.sort_values(['pnum','start_second']).drop(['start_second'],axis=1)
    for exp_num in range(experiment_num):
        for pid in df['pnum'].unique():
            print('start_pid: ',pid)

            all_SHAP_values = []
            all_data = []

            # Train / test split
            train_df = df_.loc[df_['pnum'] !=pid]
            test_df =  df_.loc[df_['pnum'] ==pid]

            split_index = int(len(test_df) * 0.5)
            train_df_target_pid = test_df.iloc[:split_index]
            
            test_df = test_df.iloc[split_index:]
            train_df = pd.concat([train_df,train_df_target_pid],axis=0)

            if len(test_df['surface_acting'].unique())< 2:
                print(f"pnum {pid} has only one class.")
                continue

            # Average label encoding
            target_thred = train_df.query('pnum == @pid')['surface_acting'].mean()
            train_df = average_based_binary_encoding(train_df,'surface_acting')
            test_df['surface_acting'] = test_df['surface_acting'].apply(lambda x: 1 if x > target_thred else 0)

            # Normalization
            means = train_df.query('pnum == @pid')[numeric_feature].mean()
            stds = train_df.query('pnum == @pid')[numeric_feature].std()

            train_df = normalization(train_df, 'z_norm', numeric_feature)
            test_df[numeric_feature] = (test_df[numeric_feature] - means) / stds

            # Normalization이 안되는 column 삭제하기
            nan_columns_train = train_df.columns[(np.isinf(train_df) | np.isnan(train_df)).any()].tolist()
            nan_columns_test =  test_df.columns[(np.isinf(test_df) | np.isnan(test_df)).any()].tolist()
            print("normalization이 안되는 column",nan_columns_train,nan_columns_test)

            nan_columns = nan_columns_train + nan_columns_test
            train_df.drop(columns=nan_columns, inplace=True)
            test_df.drop(columns=nan_columns, inplace=True)


            # Feature selection
            variance_threshold = 1e-07 # zero variance (basic_feature 제외) / audio feature도 3개 이미 빠짐
            variance_columns = list(train_df.loc[:, train_df.var() > variance_threshold].columns)
            train_df = train_df[variance_columns]
            test_df = test_df[variance_columns]

            x_feature = select_one_high_correlation_column(train_df.drop(['surface_acting'],axis=1),train_df['surface_acting'],0.8) # pairewise
            train_df = train_df[x_feature + ['surface_acting']]
            test_df = test_df[x_feature + ['surface_acting']]

            x_feature, alpha = select_features_losso(
                train_df.drop(['pnum','surface_acting'], axis=1), 
                train_df['surface_acting'],
                train_df['pnum']
            ) # lasso

            if len(x_feature)==0:
                print('no feature selection: ',pid)
                x_feature = list(train_df.drop(['surface_acting'], axis=1).columns)
                new_feature_row = pd.DataFrame({'pnum': [pid], 'experiment_num': [exp_num], 'selected_features': [[]]})
            else:
                new_feature_row = pd.DataFrame({'pnum': [pid], 'experiment_num': [exp_num], 'selected_features': [x_feature]})
                x_feature_df = pd.concat([x_feature_df, new_feature_row], ignore_index=True)

            X_train, y_train = train_df[x_feature], train_df['surface_acting']
            X_test, y_test = test_df[x_feature], test_df['surface_acting']

            # Data 증강
            if use_smote:
                X_train, y_train = apply_smote_person(X_train, y_train, pid, random_state=random_seed)

            clf = KNeighborsClassifier(n_neighbors=9) # Referred to WESAD
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)
            y_prod = clf.predict_proba(X_test)[:, 1]
            person_df = pd.DataFrame({"y_pred": y_pred, "y_prod": y_prod, "y_true": y_test, "pid": [pid] * len(y_test)})

            person_metric_df = generate_metrics(person_df)
            result_df = pd.concat([result_df, person_metric_df], axis=0)

    mean_result_df = result_df.groupby(['pid']).mean().reset_index()
    mean_metrics = mean_result_df.mean(axis=0)
    mean_metrics_std = mean_result_df.std(axis=0).round(2)
    mean_metrics['pid'] = 'all'
    mean_metrics_std['pid'] = 'all_std'
    mean_result_df  = pd.concat([mean_result_df , pd.DataFrame(mean_metrics).T,pd.DataFrame(mean_metrics_std).T], ignore_index=True)

    return mean_result_df, x_feature_df




def SVM_Hybrid(df: pd.DataFrame,numeric_feature:list,experiment_num: int, random_seed: int,use_smote: bool, result_path: str):
    result_df = pd.DataFrame()
    x_feature_df = pd.DataFrame(columns=['experiment_num', 'selected_features'])
    feature_importance_df = pd.DataFrame(columns=['pid', 'abs_feature', 'abs_value', 'pos_feature', 'pos_value', 'neg_feature', 'neg_value'])

    df_ = df.sort_values(['pnum','start_second']).drop(['start_second'],axis=1)
    for exp_num in range(experiment_num):
        for pid in df['pnum'].unique():
            print('start_pid: ',pid)

            all_SHAP_values = []
            all_data = []

            # Train / test split
            train_df = df_.loc[df_['pnum'] !=pid]
            test_df =  df_.loc[df_['pnum'] ==pid]

            split_index = int(len(test_df) * 0.5)
            train_df_target_pid = test_df.iloc[:split_index]
            
            test_df = test_df.iloc[split_index:]
            train_df = pd.concat([train_df,train_df_target_pid],axis=0)

            if len(test_df['surface_acting'].unique())< 2:
                print(f"pnum {pid} has only one class.")
                continue

            # Average label encoding
            train_df = average_based_binary_encoding(train_df,'surface_acting')
            test_df = average_based_binary_encoding(test_df,'surface_acting')

            # Normalization
            if len(numeric_feature) > 0:
                train_df = normalization(train_df, 'z_norm', numeric_feature)
                test_df = normalization(test_df, 'z_norm', numeric_feature)

                # Normalization이 안되는 column 삭제하기
                nan_columns_train = train_df.columns[(np.isinf(train_df) | np.isnan(train_df)).any()].tolist()
                nan_columns_test =  test_df.columns[(np.isinf(test_df) | np.isnan(test_df)).any()].tolist()
                print("normalization이 안되는 column",nan_columns_train,nan_columns_test)

                nan_columns = nan_columns_train + nan_columns_test
                train_df.drop(columns=nan_columns, inplace=True)
                test_df.drop(columns=nan_columns, inplace=True)
            else:
                pass


            # Feature selection
            variance_threshold = 1e-07 # zero variance (basic_feature 제외) / audio feature도 3개 이미 빠짐
            variance_columns = list(train_df.loc[:, train_df.var() > variance_threshold].columns)
            train_df = train_df[variance_columns]
            test_df = test_df[variance_columns]

            x_feature = select_one_high_correlation_column(train_df.drop(['surface_acting'],axis=1),train_df['surface_acting'],0.8) # pairewise
            train_df = train_df[x_feature + ['surface_acting']]
            test_df = test_df[x_feature + ['surface_acting']]

            x_feature, alpha = select_features_losso(
                train_df.drop(['pnum','surface_acting'], axis=1), 
                train_df['surface_acting'],
                train_df['pnum']
            ) # lasso

            if len(x_feature)==0:
                print('no feature selection: ',pid)
                x_feature = list(train_df.drop(['surface_acting'], axis=1).columns)
                new_feature_row = pd.DataFrame({'pnum': [pid], 'experiment_num': [exp_num], 'selected_features': [[]]})
            else:
                new_feature_row = pd.DataFrame({'pnum': [pid], 'experiment_num': [exp_num], 'selected_features': [x_feature]})
                x_feature_df = pd.concat([x_feature_df, new_feature_row], ignore_index=True)

            X_train, y_train = train_df[x_feature], train_df['surface_acting']
            X_test, y_test = test_df[x_feature], test_df['surface_acting']

            # Data 증강
            if use_smote:
                X_train, y_train = apply_smote_person(X_train, y_train, pid, random_state=random_seed)

            clf = SVC(probability=True, kernel='rbf', C=1, gamma=0.03) # Referred to paper https://dl.acm.org/doi/pdf/10.1145/3173574.3174165
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)
            y_prod = clf.predict_proba(X_test)[:, 1]
            person_df = pd.DataFrame({"y_pred": y_pred, "y_prod": y_prod, "y_true": y_test, "pid": [pid] * len(y_test)})

            person_metric_df = generate_metrics(person_df)
            result_df = pd.concat([result_df, person_metric_df], axis=0)

    mean_result_df = result_df.groupby(['pid']).mean().reset_index()
    mean_metrics = mean_result_df.mean(axis=0)
    mean_metrics_std = mean_result_df.std(axis=0).round(2)
    mean_metrics['pid'] = 'all'
    mean_metrics_std['pid'] = 'all_std'
    mean_result_df  = pd.concat([mean_result_df , pd.DataFrame(mean_metrics).T,pd.DataFrame(mean_metrics_std).T], ignore_index=True)

    return mean_result_df, x_feature_df


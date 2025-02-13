
import pandas as pd
import numpy as np
import json
import os
import re
import time
import multiprocessing
from datetime import datetime
from datetime import timedelta
import shutil
import zipfile

with open("env.json") as f: # input your env file
    envs = json.load(f)


## 1. Tablet data: Acc, Env, Daily_survey, AfterCall_survey
# 1.1. Unzip tablet file
def make_unzip_file(args):
    zip_file, destination_folder = args
    match = re.search(r'_p(\d+)_', zip_file) #pnum 있는파일만 대상 
    if match:
        folder_number = match.group(1) # 숫자부분만  
        folder_name =  os.path.join(destination_folder, f"{folder_number}")
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        # 압축 해제
        with zipfile.ZipFile(zip_file, 'r') as zf:
            zf.extractall(folder_name)
    else: 
        print('zip_file')
    return None 

start = time.time()
cnt = 0
if __name__ == '__main__':
    directory_path = os.path.join(envs['DATA_PATH'],"0_raw","Tablet")  # current directory
    destination_folder = os.path.join(envs['DATA_PATH'], "0_raw" ,"tablet")  # destination_directory
    zip_files = [os.path.join(directory_path, filename) for filename in os.listdir(directory_path) if filename.endswith('.zip')] 
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)
    pool.map(make_unzip_file, [(zip_file, destination_folder) for zip_file in zip_files]) # pnum 이 포함된 폴더만 make_unzip_file def 로 실행 
    pool.close()
    pool.join()

end = time.time()
print("Unzip raw table data file: ",end - start)


#  1.2. Reorganize table data("CALL_LABEL", "DAILY", "ENV", "ACC")
source_directory = os.path.join(envs['DATA_PATH'],"0_raw","tablet")
destination_directory =  os.path.join(envs['DATA_PATH'],"1_raw")

# Standardize folder names
for folder in os.listdir(source_directory): 
    if folder.startswith('0'):
        # Make a path
        source_directory_folder_path = os.path.join(source_directory, folder) 
        new_folder_name = folder.replace('0', '', 1) 
        new_folder_path = os.path.join(source_directory, new_folder_name)
        # Rename folder
        os.rename(source_directory_folder_path, new_folder_path) 

# Generate folder for data modality. 
folder_names = ["CALL_LABEL", "SURVEY_DAILY", "ENV", "ACC"]
for folder in folder_names:    
    if not os.path.exists(os.path.join(destination_directory,folder)):
        os.makedirs(os.path.join(destination_directory,folder))
    for p in range(1,24):
        p = str(p)
        p_folder = os.path.join(destination_directory,folder,p)    
        if not os.path.exists(p_folder):
            os.makedirs(p_folder)


# Reorganaized data according to the data category
for participant_folder in os.listdir(source_directory):
    participant_path = os.path.join(source_directory, participant_folder)
    
    if not os.path.isdir(participant_path):
        continue
    
    participant_number = participant_folder
    
    emotion_path = os.path.join(participant_path, "EmotionESM")
    
    for sub_folder_emotion in os.listdir(emotion_path):
        emotion_sub_path = os.path.join(emotion_path, sub_folder_emotion)
        
        if sub_folder_emotion == 'Sensors':
            sensor_path = os.path.join(emotion_sub_path, 'Sensors')
            
            for sub_sub_folder in os.listdir(sensor_path):
                if sub_sub_folder == 'Acc':
                    acc_path = os.path.join(sensor_path, "Acc")
                    destination_acc = os.path.join(destination_directory, "ACC", participant_number)
                    
                    for file_acc in os.listdir(acc_path):
                        src_acc_filepath = os.path.join(acc_path, file_acc)
                        
                        if os.path.isfile(src_acc_filepath):
                            dst_acc_filepath = os.path.join(destination_acc, file_acc)
                            shutil.copy(src_acc_filepath, dst_acc_filepath)
                
                elif sub_sub_folder == 'EnvSensor':
                    env_sensor_path = os.path.join(sensor_path, "EnvSensor")
                    destination_env_sensor = os.path.join(destination_directory, "ENV", participant_number)
                    
                    for file_env_sensor in os.listdir(env_sensor_path):
                        src_env_sensor_filepath = os.path.join(env_sensor_path, file_env_sensor)
                        
                        if os.path.isfile(src_env_sensor_filepath):
                            dst_env_sensor_filepath = os.path.join(destination_env_sensor, file_env_sensor)
                            shutil.copy(src_env_sensor_filepath, dst_env_sensor_filepath)
        
        if sub_folder_emotion == 'Survey':
            survey_path = os.path.join(emotion_path, 'Survey')
            
            for sub_sub_survey in os.listdir(survey_path):
                if sub_sub_survey == 'AfterCall':
                    after_call_path = os.path.join(survey_path, "AfterCall")
                    destination_after_call = os.path.join(destination_directory, "CALL_LABEL", participant_number)
                    
                    for file_after_call in os.listdir(after_call_path):
                        src_after_call_filepath = os.path.join(after_call_path, file_after_call)
                        
                        if os.path.isfile(src_after_call_filepath):
                            dst_after_call_filepath = os.path.join(destination_after_call, file_after_call)
                            shutil.copy(src_after_call_filepath, dst_after_call_filepath)
                
                elif sub_sub_survey == 'AfterWork':
                    after_work_path = os.path.join(survey_path, "AfterWork")
                    destination_after_work = os.path.join(destination_directory, "DAILY", participant_number)
                    
                    for file_after_work in os.listdir(after_work_path):
                        src_after_work_filepath = os.path.join(after_work_path, file_after_work)
                        
                        if os.path.isfile(src_after_work_filepath):
                            dst_after_work_filepath = os.path.join(destination_after_work, file_after_work)
                            shutil.copy(src_after_work_filepath, dst_after_work_filepath)
                
                elif sub_sub_survey == 'BeforeWork':
                    before_work_path = os.path.join(survey_path, "BeforeWork")
                    destination_before_work = os.path.join(destination_directory, "DAILY", participant_number)
                    
                    for file_before_work in os.listdir(before_work_path):
                        src_before_work_filepath = os.path.join(before_work_path, file_before_work)
                        
                        if os.path.isfile(src_before_work_filepath):
                            dst_before_work_filepath = os.path.join(destination_before_work, file_before_work)
                            shutil.copy(src_before_work_filepath, dst_before_work_filepath)


# 2. Fitbit data

# 2.1. Reorganize Fitbit data

# Fitbit_ICLab data: Step, Heart rate
source_directory = os.path.join(envs['DATA_PATH'],"0_raw","Fitbit_ICLab") # Fitbit_ICLab

STEPS = pd.DataFrame()
HR = pd.DataFrame()
Fitbit_df_1 = pd.DataFrame()

def process_heart_data(data):
    
    heart_data = data.get('heart', {})
    heartRateZones = heart_data.get('heartRateZones', [])
    restingHeartRate = heart_data.get('restingHeartRate', None)
    
    if not heartRateZones:
        return None
    
    heart_dict = {}
    for zone in heartRateZones:
        prefix = zone['name'].replace(' ', '_')
        for key, value in zone.items():
            if key != 'name':
                new_key = f"{prefix}_{key}"
                heart_dict[new_key] = value

    if restingHeartRate is not None:
        heart_dict['restingHeartRate'] = restingHeartRate
    
    return pd.DataFrame([heart_dict])


for root, dirs, files in os.walk(source_directory):
    for file in files:
        # JSON 파일만 처리
        if file.endswith('.json'):
            # 파일 경로
            file_path = os.path.join(root, file)

            # 파일 이름에서 pnum 추출
            match = re.search(r'.p(\d{2}).', file_path)
            pnum = match.group(1)
            with open(file_path, 'r') as f:
                    data = json.load(f)

            ## INTRA Day Data
            # Step 
            try:
                steps_intraday = pd.DataFrame(data['steps-intraday'])
                steps_intraday.columns = ['steps-intraday_time', 'steps-intraday_value']
                steps_intraday['date'] = data['date']
                steps_intraday['pnum'] = pnum
                steps_intraday['pnum'] = steps_intraday['pnum'].astype(int)

                STEPS = pd.concat([STEPS, steps_intraday], ignore_index=True)
            except KeyError as e:
                print(f"Step-Key Error in file {file_path}: {e}")
                continue
            except Exception as e:
                print(f"Step-Error reading file {file_path}: {e}")
                continue

            # Heart rate
            try:    
                heart_intraday = pd.DataFrame(data['heart-intraday'])
                heart_intraday.columns = ['heart-intraday_time', 'heart-intraday_value']
                heart_intraday['date'] = data['date']
                heart_intraday['pnum'] = pnum
                heart_intraday['pnum'] = heart_intraday['pnum'].astype(int)

                HR = pd.concat([HR, heart_intraday], ignore_index=True)
            except KeyError as e:
                print(f"Heart-Key Error in file {file_path}: {e}")
                continue
            except Exception as e:
                print(f"Heart-Error reading file {file_path}: {e}")
                continue
            
            ## Daily Data
            try:
                fitbit_data = process_heart_data(data)
                fitbit_data['total_steps']= data['steps']
                fitbit_data['date']= data['date']
                fitbit_data['pnum']= pnum
            
                Fitbit_df_1 = pd.concat([Fitbit_df_1, fitbit_data],axis=0)
            except KeyError as e:
                print(f"Else-Key Error in file {file_path}: {e}")
                continue
            except Exception as e:
                print(f"Else-Error reading file {file_path}: {e}")
                continue


#  Fitbit_Share data: Calories, distance, floors, elevation, sleep
source_directory = os.path.join(envs['DATA_PATH'],"0_raw","Fitbit_Share") # Fitbit_ICLab

Calories = pd.DataFrame()
Distance = pd.DataFrame()
Floors =  pd.DataFrame()
Elevation = pd.DataFrame()
Fitbit_df_2 = pd.DataFrame()

def get_value_or_nan(data, key):
    value = data.get(key, np.nan)
    if isinstance(value, list) and not value:
        return np.nan
    return value



for root, dirs, files in os.walk(source_directory):
    for file in files:
        # JSON 파일만 처리
        if file.endswith('.json'):
            # 파일 경로
            file_path = os.path.join(root, file)

            # 파일 이름에서 pnum 추출
            match = re.search(r'.p(\d{2}).', file_path)
            pnum = match.group(1)
            with open(file_path, 'r') as f:
                    data = json.load(f)

            ## INTRA Day Data
            # calories
            try:
                calories_intraday = pd.DataFrame(data['calories-intraday'])
                calories_intraday.columns = ['calories-level','calories-mets','calories-intraday_time', 'calories-intraday_value']
                calories_intraday['date'] = data['date']
                calories_intraday['pnum'] = pnum
                calories_intraday['pnum'] = calories_intraday['pnum'].astype(int)

                Calories = pd.concat([Calories, calories_intraday], ignore_index=True)
            except KeyError as e:
                print(f"Step-Key Error in file {file_path}: {e}")
                continue
            except Exception as e:
                print(f"Step-Error reading file {file_path}: {e}")
                continue

            # distance
            try: 
                distance_intraday = pd.DataFrame(data['distance-intraday'])
                distance_intraday.columns = ['distance-intraday_time', 'distance-intraday_value']
                distance_intraday['date'] = data['date']
                distance_intraday['pnum'] = pnum
                distance_intraday['pnum'] = distance_intraday['pnum'].astype(int)

                Distance = pd.concat([Distance, distance_intraday], ignore_index=True)
            except KeyError as e:
                print(f"distance-Key Error in file {file_path}: {e}")
                continue
            except Exception as e:
                print(f"distance-Error reading file {file_path}: {e}")
                continue

            # floors
            try: 
                floors_intraday = pd.DataFrame(data['floors-intraday'])
                floors_intraday.columns = ['floors-intraday_time', 'floors-intraday_value']
                floors_intraday['date'] = data['date']
                floors_intraday['pnum'] = pnum
                floors_intraday['pnum'] = floors_intraday['pnum'].astype(int)

                Floors = pd.concat([Floors, floors_intraday], ignore_index=True)
            except KeyError as e:
                print(f"floors-Key Error in file {file_path}: {e}")
                continue
            except Exception as e:
                print(f"floors-Error reading file {file_path}: {e}")
                continue

            # elevation 
            try:
                elevation_intraday = pd.DataFrame(data['elevation-intraday'])
                elevation_intraday.columns = ['elevation-intraday_time', 'elevation-intraday_value']
                elevation_intraday['date'] = data['date']
                elevation_intraday['pnum'] = pnum
                elevation_intraday['pnum'] = elevation_intraday['pnum'].astype(int)

                Elevation = pd.concat([Elevation, elevation_intraday], ignore_index=True)
            except KeyError as e:
                print(f"elevation-Key Error in file {file_path}: {e}")
                continue
            except Exception as e:
                print(f"elevation-Error reading file {file_path}: {e}")
                continue
            
            try:
                ## Daily Data
                sleep_data = data.get('sleep')
                if isinstance(sleep_data, list):
                    sleep_data = {'deep': np.nan, 'light': np.nan, 'rem': np.nan, 'wake': np.nan}

                fitbit_data = pd.DataFrame([sleep_data])
                fitbit_data['minutesSedentary'] = get_value_or_nan(data, 'minutesSedentary')
                fitbit_data['minutesLightlyActive'] = get_value_or_nan(data, 'minutesLightlyActive')
                fitbit_data['minutesFairlyActive'] = get_value_or_nan(data, 'minutesFairlyActive')
                fitbit_data['minutesVeryActive'] = get_value_or_nan(data, 'minutesVeryActive')
                fitbit_data['activityCalories'] = get_value_or_nan(data, 'activityCalories')
                fitbit_data['total_calories'] = get_value_or_nan(data, 'calories')
                fitbit_data['total_distance'] = get_value_or_nan(data, 'distance')
                fitbit_data['total_floors'] = get_value_or_nan(data, 'floors')
                fitbit_data['total_elevation'] = get_value_or_nan(data, 'elevation')

                fitbit_data['date']= data['date']
                fitbit_data['pnum']= pnum
            
                Fitbit_df_2 = pd.concat([Fitbit_df_2, fitbit_data],axis=0)
            except KeyError as e:
                print(f"Else-Key Error in file {file_path}: {e}")
                continue
            except Exception as e:
                print(f"Else-Error reading file {file_path}: {e}")
                continue


# 2.2 Save the data

destination_directory =  os.path.join(envs['DATA_PATH'],"1_raw","FITBIT")

# intraday data 
Calories.to_csv(os.path.join(destination_directory,"Fitbit_calories.csv")) # 1 min freq
Distance.to_csv(os.path.join(destination_directory,"Fitbit_distance.csv")) # 1 min freq
Floors.to_csv(os.path.join(destination_directory,"Fitbit_floors.csv")) # 1 min freq
Elevation.to_csv(os.path.join(destination_directory,"Fitbit_elevation.csv")) # 1 min freq
STEPS.to_csv(os.path.join(destination_directory,"Fitbit_step.csv")) # 1 min freq
HR.to_csv(os.path.join(destination_directory,"Fitbit_hr.csv")) # 1 sec freq, but not regular

Final_fitbit_df = pd.concat([Fitbit_df_1.set_index(['pnum', 'date']), Fitbit_df_2.set_index(['pnum', 'date'])],axis=1).reset_index()
Final_fitbit_df.to_csv(os.path.join(destination_directory,"Fitbit_daily.csv"))


# 3. CALL LOG 
call_log_content = os.path.join(envs['DATA_PATH'],"0_raw","Callcenter",'content')
call_log_time = os.path.join(envs['DATA_PATH'],"0_raw","Callcenter",'cityhall_audio_based_call_info.xlsx')

# 3.1 server1 - contents of call 
# Concatenate all call log data
call_log_content_df = [] 

for filename in os.listdir(call_log_content):
    file_path = os.path.join(call_log_content, filename)
    if filename.endswith(('.xls', '.xlsx', '.xlsm')):
        xls = pd.ExcelFile(file_path)
        for sheet_name in xls.sheet_names:
            sheet_data = pd.read_excel(file_path, sheet_name=sheet_name)
            call_log_content_df.append(sheet_data)

call_log_content_df = pd.concat(call_log_content_df, ignore_index=True)
call_log_content_df = call_log_content_df.drop(" ", axis=1)

destination_directory =  os.path.join(envs['DATA_PATH'],"1_raw","CALL_LOG")
call_log_content_df.to_csv(os.path.join(destination_directory,"Call_log_contents.csv"),index=False)

# 3.2 server2 - time of call 
# Rename the file name
call_log_time_df = pd.read_excel(call_log_time)
call_log_time_df.to_csv(os.path.join(destination_directory,"Call_log_time.csv"),index=False)


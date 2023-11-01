import time

import numpy as np
from LocalModel import LocalModel
import yaml
import pandas as pd
import pickle
import os
from pathlib import Path
import sys
from datetime import datetime


class MIAConstructor:
    def __init__(self, session_name) -> None:
        self.session_name = session_name
        with open("config.yaml", "r") as yamlfile:
            self.params = yaml.load(yamlfile, Loader=yaml.FullLoader)
        self.params["defence"] = 'none'
        

    def _read_model_states(self):
        shadow_model_states = []
        target_model_states = []
        self.rounds = self.params["global"]["rounds"]
        for i in range(1, self.rounds + 1):
            try:
                with open(f'sessions\\{self.session_name}\\weights\\shadow_{i}.weights', 'rb') as f:
                    shadow_model_states.append(pickle.load(f))
                with open(f'sessions\\{self.session_name}\\weights\\target_{i}.weights', 'rb') as f:
                    target_model_states.append(pickle.load(f))
            except FileNotFoundError:
                print('file not found')
                sys.exit(0)
        return target_model_states, shadow_model_states

    def create_mia_dataset(self):
        print(f'Start of creating MIA datasets...')
        start_time = datetime.now()
        target_model_states, shadow_model_states = self._read_model_states()

        shadow_train_df = pd.read_csv(f'sessions\\{self.session_name}\\datasets\\shadow.csv')
        target_train_df = pd.read_csv(f'sessions\\{self.session_name}\\datasets\\target.csv')
        test_df = pd.read_csv(f'sessions\\{self.session_name}\\datasets\\test.csv')

        


        mia_train = self._get_mia_dataframe(shadow_model_states, shadow_train_df, test_df)        
        mia_train.to_csv(f'sessions\\{self.session_name}\\train_mia.csv', index=False)
        print(f'Creating MIA train dataset ended in {datetime.now() - start_time}.')

        start_time = datetime.now()
        
        mia_test = self._get_mia_dataframe(target_model_states, target_train_df, test_df)        
        mia_test.to_csv(f'sessions\\{self.session_name}\\test_mia.csv', index=False)
        print(f'Creating MIA test dataset ended in {datetime.now() - start_time}.')
        
        print("Creating Diffs")
        start_time = datetime.now()
        mia_train_diffs = self._get_diffs(mia_train)        
        mia_train_diffs.to_csv(f'sessions\\{self.session_name}\\train_mia_diffs.csv', index=False)
        mia_test_diffs = self._get_diffs(mia_test)        
        mia_test_diffs.to_csv(f'sessions\\{self.session_name}\\test_mia_diffs.csv', index=False)
        print(f"Creating diffs ended in {datetime.now() - start_time}.")

    def _get_diffs(self, df):
        all_vals = []
        for i in range(len(df) - 1):
            vals = []
            ndvals = df.iloc[i:i+1].values[0]
            #vals = df.iloc[i:i+1].values
            for j in range(ndvals.shape[0] - 2): # last is Presented
                vals.append(ndvals[j+1] - ndvals[j])
            

            # комбинирование признаков
            diff_sum = sum(vals)
            diff_abs_sum = sum(map(abs, vals))

            vals.append(diff_sum)
            vals.append(diff_abs_sum)

            # добавление Presented стоблца
            vals.append(ndvals[-1])

            #vals = np.delete(vals, 0)
            all_vals.append(vals)
        col_names= [str(i+1) for i in range(len(all_vals[0]) - 3)]

        col_names.append("Total Sum")
        col_names.append("Total Absolute Sum")
        col_names.append("Presented")

        res_df = pd.DataFrame(all_vals, columns=col_names)
        return res_df
        

    def _get_mia_dataframe(self, model_states, presented_df, not_presented_df, representation=True):
        test_model = LocalModel(presented_df, self.params)
        
        # подготовка датасетов        
        conf_dataframe  = presented_df.drop(self.params["dataset"]["labels"], axis=1) 
        conf_test_dataframe  = not_presented_df.drop(self.params["dataset"]["labels"], axis=1)

        count_presented = conf_dataframe.shape[0]
        count_not_presented = conf_test_dataframe.shape[0]

        if representation:
            # сокращаем тестовый датасет до размера обучающего датасета
            if count_not_presented > count_presented * 2:
                conf_test_dataframe = conf_test_dataframe.sample(n=count_presented)
            elif count_presented > count_not_presented*2:
                conf_dataframe = conf_dataframe.sample(n=count_not_presented)

        count_presented = conf_dataframe.shape[0]
        count_not_presented = conf_test_dataframe.shape[0]

        # конкатенация строк
        conf_dataframe = pd.concat([conf_dataframe, conf_test_dataframe], axis=0)

        # получение confidence series для присутствующих в выборке записей
        test_model.set_weights(model_states[0])
        mia_train_df = pd.DataFrame(test_model.get_confidence_score(conf_dataframe), columns=['1'])

        for state_num in range(1, len(model_states)):
            test_model.set_weights(model_states[state_num])
            mia_train_df = pd.concat([mia_train_df, pd.DataFrame(test_model.get_confidence_score(conf_dataframe), columns=[str(state_num+1)])], axis=1)

        labels_list = [True for _ in range(count_presented)]
        labels_list += [False for _ in range(count_not_presented)]
        labels_df = pd.DataFrame(labels_list, columns=['Presented'])
        
        # конкатенация столбцов
        mia_train_df = pd.concat([mia_train_df, labels_df ], axis=1)

        mia_train_df = mia_train_df.dropna()
        return mia_train_df


if __name__ == "__main__":
    session_name = "Texas100 (2023-10-31 14-01-06)"
    miaconstructor = MIAConstructor(session_name)
    miaconstructor.create_mia_dataset()
    sys.exit(0)
    mia_train = pd.read_csv(f'sessions\\{session_name}\\train_mia.csv')
    mia_test = pd.read_csv(f'sessions\\{session_name}\\test_mia.csv')

    mia_train_diffs = miaconstructor._get_diffs(mia_train)
    mia_test_diffs = miaconstructor._get_diffs(mia_test)
    
    print(mia_test_diffs.head())

    mia_train_diffs.to_csv(f'sessions\\{session_name}\\train_mia_diffs.csv', index=False)
    mia_test_diffs.to_csv(f'sessions\\{session_name}\\test_mia_diffs.csv', index=False)

    #miaconstructor.create_mia_dataset()
import time
from LocalModel import LocalModel
import yaml
import pandas as pd
import pickle
import os
from pathlib import Path
from MIAConstructor import MIAConstructor
from datetime import datetime

class FL:
    def __init__(self, df_train, df_test, params=dict()) -> None:
        #self.current_session = str(time.strftime("%Y-%m-%d %H-%M-%S", time.gmtime()))
        self.current_session = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
        Path("sessions").mkdir(parents=True, exist_ok=True)        
        os.mkdir(f'sessions\\{self.current_session}')
        with open(f'sessions\\{self.current_session}\\config.yaml', 'w') as yaml_file:
            yaml.dump(params, yaml_file, default_flow_style=False)      

        self.target_model_num = 0
        self.target_model_states = []
        self.shadow_model_states = []

        self.params = params.copy()
        self.parse_params(params) 
        self.datasets = []       
        self.df_train_split(df_train)
        self.df_test = df_test

        os.mkdir(f'sessions\\{self.current_session}\\datasets')
        self.df_shadow.to_csv(f'sessions\\{self.current_session}\\datasets\\shadow.csv', index=False)
        self.df_test.to_csv(f'sessions\\{self.current_session}\\datasets\\test.csv', index=False)
        self.df_target.to_csv(f'sessions\\{self.current_session}\\datasets\\target.csv', index=False)

        self.models = []
        for _ in range(self.MODEL_COUNT):  
            self.models.append(LocalModel(self.datasets[0][0], params=params))
        self.shadow_model = LocalModel(self.shadow_dataset[0], params=params)
        
        
    
    def parse_params(self, params:dict):
        try:
            self.MODEL_COUNT = params["global"]["model_count"]
        except KeyError:
            self.MODEL_COUNT = 5
        try:
            self.rounds = params["global"]["rounds"]
        except KeyError:
            self.rounds = 20


    def _splitDataframe(df, parts):
        row_count = df.shape[0]
        frac_size = int(row_count//parts)

        df_parts = []
        for i in range(parts):
            df_parts.append(df.iloc[i*frac_size:(i+1)*frac_size])
        return df_parts

    def df_train_split(self, df_train):
        df_train.copy()
        full_dfs = FL._splitDataframe(df_train, self.MODEL_COUNT+1)
        self.df_target = full_dfs[self.target_model_num]
        for i in range(self.MODEL_COUNT):
            self.datasets.append(FL._splitDataframe(full_dfs[i], self.rounds))
        self.df_shadow = full_dfs[self.MODEL_COUNT]
        self.shadow_dataset = FL._splitDataframe(full_dfs[self.MODEL_COUNT], self.rounds)

    

       
    def _save_weights(self):
        os.mkdir(f'sessions\\{self.current_session}\\weights')
        i = 1
        for target_state in self.target_model_states:
            with open(f'sessions\\{self.current_session}\\weights\\target_{i}.weights', 'wb') as f:
                pickle.dump(target_state, f)
            i += 1

        i = 1
        for shadow_state in self.shadow_model_states:
            with open(f'sessions\\{self.current_session}\\weights\\shadow_{i}.weights', 'wb') as f:
                pickle.dump(shadow_state, f)
            i += 1

    def work(self):
        metric_results = []
        total_time_start = time.time()
        for i in range(self.rounds):
            print(f"ROUND # {i+1}")
            total_seconds = self.round(i)
            global_model = self.aggregate()
            
            print('GLOBAL MODEL EVALUATION:')
            metric_result  = global_model.evaluate(self.df_test)
            metric_results.append(metric_result)

            for metric in range(len(global_model.metrics)):
                print(f'Global model {global_model.metrics[metric]}: { round(metric_result[metric], 2)}')
            print(f'Round ended in {total_seconds} seconds.')
            print("-----------------------")
        print('ALL METRICS:')
        for i in range(len(metric_results)):
            for j in range(len(metric_results[i])):
                print(i+1, f'{global_model.metrics[j]}:{round(metric_results[i][j], 4)}')
                
        print(f'TOTAL TIME: {time.time() - total_time_start} seconds.')
        print('Saving session...')
        self._save_weights()

    def round(self, round):
        total_seconds = 0        
        for i in range(self.MODEL_COUNT):
            seconds = self.models[i].learn(self.datasets[i][round])
            total_seconds += seconds
        total_seconds += self.shadow_model.learn(self.shadow_dataset[round])

        self.target_model_states.append(self.models[self.target_model_num].extract_weights())
        self.shadow_model_states.append(self.shadow_model.extract_weights())
        
        return total_seconds
    

    def aggregate(self):
        global_model_params = self.params.copy()
        global_model_params["defence"] = "none"
        global_model = LocalModel(self.datasets[0][0], params=global_model_params)
        
        all_weigths = []
        for model in self.models:
            all_weigths.append(model.extract_weights())
        new_weights = []
        for tuple_num in range(len(all_weigths[0])):
            l_w, l_bias = all_weigths[0][tuple_num]
            
            for model_num in range(1, self.MODEL_COUNT):
                m_w = all_weigths[model_num]
                l_w += m_w[tuple_num][0]
                l_bias += m_w[tuple_num][1]
            l_w /= self.MODEL_COUNT
            l_bias /= self.MODEL_COUNT
            new_weights.append((l_w, l_bias))
        global_model.set_weights(new_weights)
        for model in self.models:
            model.set_weights(new_weights)
        
        self.shadow_model.set_weights(new_weights)
        return global_model
           
if __name__ == "__main__":

    with open("config.yaml", "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    df_train = pd.read_csv('prepared_datasets\\train.csv')
    df_test = pd.read_csv('prepared_datasets\\test.csv')
    fl = FL(df_train, df_test, config)
    fl.work()
    miaconstructor = MIAConstructor(fl.current_session)
    miaconstructor.create_mia_dataset()
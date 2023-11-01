# Класс FL
import time
from LocalModel import LocalModel
import yaml
import pandas as pd
import pickle
import os
from pathlib import Path
from MIAConstructor import MIAConstructor
from datetime import datetime
import logging
class FL:
    def __init__(self, df_train, df_test, params=dict()) -> None:       
             

        self.target_model_num = 0
        self.target_model_states = []
        self.shadow_model_states = []

        self.params = params.copy()

        self.parse_params(params)
        #self.current_session = str(time.strftime("%Y-%m-%d %H-%M-%S", time.gmtime()))
        self.current_session = f'{self.short_name} ({ str(datetime.now().strftime("%Y-%m-%d %H-%M-%S")) })'
        Path("sessions").mkdir(parents=True, exist_ok=True)        
        os.mkdir(f'sessions\\{self.current_session}')
        with open(f'sessions\\{self.current_session}\\config.yaml', 'w') as yaml_file:
            yaml.dump(params, yaml_file, default_flow_style=False)

        logging.basicConfig(filename=f'sessions\\{self.current_session}\\FL.log',
                    format='%(asctime)s %(message)s',
                    filemode='a')
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO) 


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
        
        print('SHADOW MODEL SUMMARY:')
        print(self.shadow_model.model.summary())
    
    def parse_params(self, params:dict):
        try: self.MODEL_COUNT = params["global"]["model_count"]
        except KeyError: self.MODEL_COUNT = 5
        
        try: self.rounds = params["global"]["rounds"]
        except KeyError: self.rounds = 20
        
        try: self.decription = params["global"]["description"]
        except KeyError: self.decription = "no description"
        
        try:self.short_name = params["global"]["short_name"]
        except KeyError: self.short_name = "unnamed"
        
        try: self.round_train_data_mode = params["global"]["round_train_data_mode"]
        except KeyError: self.round_train_data_mode = "each_round_add_new"

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

    

       
    def _save_round_weights(self, round):        
        Path(f"sessions").mkdir(parents=True, exist_ok=True)
        Path(f"sessions\\{self.current_session}").mkdir(parents=True, exist_ok=True)
        Path(f"sessions\\{self.current_session}\\weights").mkdir(parents=True, exist_ok=True)     

        with open(f'sessions\\{self.current_session}\\weights\\target_{round+1}.weights', 'wb') as f:
            target_state = self.target_model_states[round]
            pickle.dump(target_state, f)
        with open(f'sessions\\{self.current_session}\\weights\\shadow_{round+1}.weights', 'wb') as f:
            shadow_state = self.shadow_model_states[round]
            pickle.dump(shadow_state, f)
        

       
    def _save_weights(self):
        for i in range(self.rounds):
            self._save_round_weights(i)

    def work(self):
        try:
            metric_results = []
            total_time_start = datetime.now()
            for i in range(self.rounds):
                print(f"ROUND # {i+1}")
                self.logger.info(f"ROUND # {i+1}")
                total_seconds = self.round(i)
                global_model = self.aggregate()
                
                print('GLOBAL MODEL EVALUATION:')
                metric_result  = global_model.evaluate(self.df_test)
                metric_results.append(metric_result)
        
                for metric in range(len(global_model.metrics)):
                    print(f'Global model {global_model.metrics[metric]}: { round(metric_result[metric], 2)}')
                    self.logger.info(f'Global model {global_model.metrics[metric]}: { round(metric_result[metric], 2)}')
                print(f'Round ended in {total_seconds} seconds.')
                print("-----------------------")

                self.logger.info(f'Round ended in {total_seconds} seconds.')
                self.logger.info("-----------------------")

            print('ALL GLOBAL METRICS:')
            self.logger.info("ALL GLOBAL METRICS:")
            for i in range(len(metric_results)):
                for j in range(len(metric_results[i])):
                    m_s =f'{str(i+1)}: {global_model.metrics[j]}: {round(metric_results[i][j], 4)}'
                    print(m_s)
                    self.logger.info(m_s)
                    
            print(f'TOTAL TIME: {datetime.now() - total_time_start} seconds.')
            self.logger.info(f'TOTAL TIME: {datetime.now() - total_time_start} seconds.')
            self.logger.info("Work ended successfully.")
        except Exception as e:
            self.logger.exception(e)
            print(e)

    def round(self, round):
        total_seconds = 0 
        # обучение клиентов       
        for i in range(self.MODEL_COUNT):
            # создание обучающей раундовой выборки
            if self.round_train_data_mode == "each_round_add_new":
                round_dataset = None
                for j in range(round+1):
                    if round_dataset is None: round_dataset = self.datasets[i][j].copy()
                    else: round_dataset = pd.concat([round_dataset, self.datasets[i][j]], axis=0)
            elif self.round_train_data_mode == "each_round_new":
                round_dataset = self.datasets[i][round]     

            seconds, target_history = self.models[i].learn(round_dataset)
            target_metric = target_history.history[self.models[i].metrics[0]][-1]
            total_seconds += seconds
            self.logger.info(f"Target model # {i+1} {self.models[i].metrics[0]}: {str(target_metric)}")

        # создание обучающей раундовой выборки
        if self.round_train_data_mode == "each_round_add_new":
            round_dataset = None
            for j in range(round+1):
                if round_dataset is None: round_dataset = self.shadow_dataset[j].copy()
                else: round_dataset = pd.concat([round_dataset, self.shadow_dataset[j]], axis=0)
        elif self.round_train_data_mode == "each_round_new":
            round_dataset = self.shadow_dataset[j]
        # обучение теневой модели
        shadow_seconds, shadow_history = self.shadow_model.learn(round_dataset)
        shadow_history = shadow_history.history[self.shadow_model.metrics[0]][-1]
        total_seconds += shadow_seconds
        self.logger.info(f"Shadow model {self.shadow_model.metrics[0]}: {str(shadow_history)}")

        self.target_model_states.append(self.models[self.target_model_num].extract_weights())
        self.shadow_model_states.append(self.shadow_model.extract_weights())
        
        self._save_round_weights(round)
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
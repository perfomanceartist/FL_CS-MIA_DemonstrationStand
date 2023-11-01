# Класс LocalModel
import tensorflow as tf
import pandas as pd
import numpy as np

import time
import yaml

class LocalModel:
    def __init__(self, df_train, params=dict()) -> None:
        self.parse_params(params)       
        self.y_length = len(self.label_names)         
        #self.x_length = len(df_train.columns) - self.y_length
        self.construct_model()

    def parse_params(self, params):
        self.plain_params = params.copy()
        try:
            self.label_names = params["dataset"]["labels"]
        except KeyError:
            self.label_names = ["label_0", "label_1", "label_2", "label_3", "label_4", "label_5", "label_6", "label_7", "label_8", "label_9"]
        try:
            self.structure = params["local_model"]["structure"]["type"]
        except:
            self.structure = "fcn"
        if self.structure == 'conv':
            self.param_layers = params["local_model"]["structure"]["layers"]
        try:
            self.input_shape = tuple(params["local_model"]["structure"]["input_shape"])
        except:
            print("INPUT SHAPE IS NOT SPECIFIED!")
            self.input_shape = (784,)
        try:
            self.hidden_layers = params["local_model"]["structure"]["fcn_hidden_layers"]
        except KeyError:
            self.hidden_layers = 2
        try:
            self.hidden_layer_neurons = params["local_model"]["structure"]["fcn_hidden_layer_neurons"]
            if type(self.hidden_layer_neurons) == type(150):
                self.hidden_layer_neurons = [self.hidden_layer_neurons for i in range(self.hidden_layers)]
            while len(self.hidden_layer_neurons) != self.hidden_layers:
                self.hidden_layer_neurons.append(self.hidden_layer_neurons[:-1])
        except KeyError:
            self.hidden_layer_neurons = [150 for i in range(self.hidden_layers)]
        try:
            self.batch_size = params["local_model"]['batch_size'] 
        except KeyError:
            self.batch_size = 10
        try:
            self.validation_split = params["local_model"]['validation_split'] 
        except KeyError:
            self.validation_split = 0.1
        try:
            self.epochs = params["local_model"]["epochs"]
        except KeyError:
            self.epochs = 3
        try:
            self.debug = params["local_model"]["debug"]
        except KeyError:
            self.debug = True


        try:
            self.loss = params["local_model"]["loss"]
        except KeyError:
            self.loss = 'categorical_crossentropy'
        try:
            self.optimizer = params["local_model"]['optimizer']
        except KeyError:
            self.optimizer = 'adam'
        try:
            self.metrics = params["local_model"]['metrics']
        except KeyError:
            self.metrics = ['categorical_accuracy']
        try:
            pass
        except KeyError:
            pass

        try:
            self.defence = params["defence"] # none, dropout, pckd, data_spoof, noise
        except KeyError:
            self.defence = 'none'
        
        if self.defence == 'dropout':
            try:
                self.dropout_rate = params['dropout']['rate']
            except KeyError:
                self.dropout_rate = 0.5
        elif self.defence == 'pckd':
            try:
                self.pckd_K  = params["pckd"]["K"]
            except KeyError:
                self.pckd_K = 4
            try:
                self.pckd_teacher_epochs = params['pckd']['teacher_epochs']
            except KeyError:
                self.pckd_teacher_epochs = 3
        elif self.defence == 'noise':
            try:
                self.noise_scale  = params["noise"]["scale"]
            except KeyError:
                self.noise_scale = 0.1
        elif self.defence =='with_data_spoof':
            try:
                self.corr_limit = params['data_spoof']["corr_limit"]
            except KeyError:
                self.corr_limit = 0.1
            try:
                self.truth_limit = params['data_spoof']["truth_limit"]
            except KeyError:
                self.truth_limit = 0.5
            try:
                self.set_student_layer_num = params["data_spoof"]['set_student_layer_num']
            except KeyError:
                self.set_student_layer_num = 0.0
            try:
                self.set_student_prob = params["data_spoof"]['set_student_prob']
            except KeyError:
                self.set_student_prob = 1.0

    def prepare_dataset(self, df_train:pd.DataFrame, with_y=True):         
        data_x = df_train.copy()
        if self.defence == 'pckd':
            data_x = self.pckd(data_x)        
        if with_y:
            data_y = data_x[self.label_names].copy().to_numpy()
            data_x  = data_x.drop(self.label_names, axis=1).to_numpy()
            # стандартизация входных данных
            data_x = data_x / 255
        else:
            data_x = data_x.to_numpy()

        if self.structure == "conv":  
            data_x = data_x.reshape(data_x.shape[0],3,32,32).transpose(0,2,3,1)
        
        if with_y:
            return data_x, data_y
        else:
            return data_x

    def _construct_conv_model(self, params) -> (list, list):
        layers = []
        layers_have_weights = []
        for i in range(len(self.param_layers)):
            param = self.param_layers[i]["layer"]
            if param["type"] == 'conv2d':
                if i == 0:
                    layers.append(
                        tf.keras.layers.Conv2D( param["filters"], 
                                               tuple(param["kernel"]),  
                                               tuple(param["strides"]), 
                                               padding=param["padding"], 
                                               input_shape=self.input_shape)
                    )
                else:
                    layers.append(
                        tf.keras.layers.Conv2D( param["filters"], 
                                               tuple(param["kernel"]),  
                                               tuple(param["strides"]), 
                                               padding=param["padding"])
                    )
                layers_have_weights.append(True)
            elif param["type"] == "maxpooling2d":
                layers.append(
                tf.keras.layers.MaxPooling2D( tuple(param["kernel"]),  param["strides"], padding="same")
                )
                layers_have_weights.append(False)
        
        layers.append(tf.keras.layers.Flatten())
        layers_have_weights.append(False)

        fcn_layers, fcn_layers_have_weights = self._construct_fcn_model(params)
        return layers + fcn_layers, layers_have_weights + fcn_layers_have_weights       


    def _construct_fcn_model(self, params) -> (list, list):
        layers = []
        layers_have_weights = []
                
        defence = params["defence"]
        dropout_rate = params["dropout"]["rate"]
        params = params["local_model"]["structure"]
        for i in range(params["fcn_hidden_layers"]):
            if True:
                neuron_number = self.hidden_layer_neurons[i]
            else:
                if type(params["fcn_hidden_layer_neurons"]) == 'list':
                    neuron_number = params["fcn_hidden_layer_neurons"][i]
                else:
                    neuron_number = params["fcn_hidden_layer_neurons"]
            if self.structure == 'fcn' and i == 0:
                layers.append(tf.keras.layers.Dense(neuron_number, input_shape=self.input_shape, activation="relu"))
            else:
                layers.append(tf.keras.layers.Dense(neuron_number, activation="relu"))            
            layers_have_weights.append(True)
        if defence == 'dropout':
            dropout_place = len(layers) - 1
            layers.insert(dropout_place,  tf.keras.layers.Dropout(dropout_rate))
            layers_have_weights.insert(dropout_place, False)
        
        layers.append(tf.keras.layers.Dense(self.y_length, activation='softmax'))
        layers_have_weights.append(True)        
        
        return layers, layers_have_weights


    def construct_model(self):
        
        if self.structure == "fcn":
            layers, self.layers_have_weights = self._construct_fcn_model(self.plain_params)
        elif self.structure == "conv":
            layers, self.layers_have_weights = self._construct_conv_model(self.plain_params)
        self.model = tf.keras.models.Sequential(layers)
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        #print(self.model.summary())
        return
    
        layers = []
        self.layers_have_weights = []
        if self.structure == 'conv':
            for param in self.param_layers:
                if param["type"] == 'conv2d':
                    layers.append(
                        tf.keras.layers.Conv2D(param["filters"],
                                                tuple(param["kernel"]),
                                                tuple(param["strides"])),
                                                padding=param["padding"])
                    self.layers_have_weights.append(True)
                elif param["type"] == "maxpooling2d":
                    layers.append(
                        tf.keras.layers.MaxPooling2D( tuple(param["kernel"]),  tuple(param["strides"]))
                    )
                    self.layers_have_weights.append(False)
            layers.append(tf.keras.layers.Flatten())
        for i in range(self.hidden_layers):
            if self.structure == 'fcn' and i == 0:
                layers.append(tf.keras.layers.Dense(self.hidden_layer_neurons[i], input_shape=(self.x_length,), activation="relu"))
            else:
                layers.append(tf.keras.layers.Dense(self.hidden_layer_neurons[i], activation="relu"))            
            self.layers_have_weights.append(True)
        if self.defence == 'dropout':
            dropout_place = len(layers) - 1
            layers.insert(dropout_place,  tf.keras.layers.Dropout(self.dropout_rate))
            self.layers_have_weights.insert(dropout_place, False)
        
        layers.append(tf.keras.layers.Dense(self.y_length, activation='softmax'))
        self.layers_have_weights.append(True)
        
       

        if self.defence == 'data_spoof':
            self.teacher = tf.keras.models.Sequential(layers)
            self.teacher.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)


    def get_correlations(df, corr_type='dict'):
        if corr_type == 'list':
            correlations = []
            for col in df.columns:
                if col == 'label':
                    continue
                corr = (col, df[col].corr(df['label']))            
                if np.isnan(corr[1]):
                    continue
                correlations.append(corr)
            correlations = sorted(correlations, key=lambda x: x[1], reverse=True)            
        elif corr_type == 'dict':
            correlations = dict()
            for col in df.columns:
                if col == 'label':
                    continue
                corr = df[col].corr(df['label'])        
                if np.isnan(corr):
                    corr = 0
                correlations[col] = corr
        return correlations

    def get_distributions(df, ranging=2):
        distr = dict()
        distr['count'] = len(df) #df['label'].count()
        for col in df.columns:
            values = df[col].unique()
            min_value = values.min()
            max_value = values.max()
            width = (max_value - min_value)//ranging
            ranges = []

            if width == 0:
                ranges.append({"start" : min_value, "end" : max_value, "freq" : 60000, "values":[min_value, max_value]})
                distr[col] = ranges
                continue

            for i in range(ranging):
                start = + i*width + min_value
                if i == ranging - 1:
                    end = max_value
                else:
                    end = (i+1)*width + min_value
                ranges.append({"start" : start, "end" : end, "freq" : 0, "values":[]})        
            distr[col] = []
            freqs = df[col].value_counts()        
            for value in values:
                freq = freqs[value]
                
                # найти диапазон, в который входит value
                for i in range(len(ranges)):
                    r = ranges[i]
                    if i == ranging - 1:
                        if value >= r["start"] and value <= r["end"]:
                            r["freq"] += freq
                            r["values"].append(value)
                    else:
                        if value >= r["start"] and value < r["end"]:
                            r["freq"] += freq
                            r["values"].append(value)
            distr[col] = ranges
        return distr

    
    
    def spoof_dataset(df, truth_limit=0.6, fraction=0.5, corr_limit=0.1, ranging=16):        
        correlations = LocalModel.get_correlations(df)    
        groups = []
        spoofed = None
        # разделение данных по классам
        for label_name, group in df.groupby('label'):
            groups.append(group)
        for df_class in groups: # работаем с каждым классом отдельно
            df_class = df_class.iloc[np.random.permutation(len(df_class))]
            #print(correlations)
            distr = LocalModel.get_distributions(df_class, ranging=ranging) #print(get_distributions(df_class)['4x25'])
            #print(distr)
            df_class_size = distr.pop('count')

            generated = [] # поддельные записи
            for i in range(int(df_class_size*fraction)): # генерируем запрашиваемое количество записей
                generated_el = dict()
                for col in distr.keys():  # перебираем по всем колонкам
                    # попробовать на count = 980 и (0, 978), (36, 1), (80, 1)
                    if col == 'label':
                        val = df_class[col].iloc[i] # оставляем метку
                    elif abs(correlations[col]) > corr_limit:
                        val = df_class[col].iloc[i] # оставляем как есть
                    elif np.random.random() < truth_limit:
                        val = df_class[col].iloc[i] # оставляем как есть
                    else:
                        r = np.random.randint(1, df_class_size+1)
                        for rang in distr[col]:
                            if rang['freq'] == 0:
                                continue
                            if r <= rang['freq']:
                                #print(col, rang)
                                if rang['end'] == rang['start']:
                                    val = rang['values'][0]
                                else:
                                    #val = np.random.randint(rang['start'], rang['end'])
                                    val = rang['values'][np.random.randint(len(rang['values']))]
                                break
                            r -= rang['freq']
                    generated_el[col] = val
                generated.append(generated_el)
            gen_df = pd.DataFrame(generated)
            if spoofed is None:
                spoofed = gen_df.copy()
            else:
                spoofed = pd.concat([spoofed, gen_df], axis=0)
        return spoofed


    def pckd(self, data:pd.DataFrame) -> pd.DataFrame:
        "Применяя метод PCKD, возвращает новый датафрейм с измененными метками"
        if self.debug:
            print('Starting PCKD dataset change')
        start_time = time.time()
        teacher_train_data = LocalModel._splitDataframe(data, self.pckd_K)
        # создание моделей-учителей
        teacher_models = []
        for i in range(self.pckd_K):
            layers = []
            for i in range(self.hidden_layers):
                if i == 0:
                    layers.append(tf.keras.layers.Dense(self.hidden_layer_neurons, input_shape=(self.x_length,), activation="relu"))
                else:
                    layers.append(tf.keras.layers.Dense(self.hidden_layer_neurons, activation="relu"))            
                self.layers_have_weights.append(True)            
            layers.append(tf.keras.layers.Dense(self.y_length, activation='softmax'))

            teacher = tf.keras.models.Sequential(layers)
            teacher.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
            teacher_models.append(teacher)
                
        # обучение моделей-учителей
        for i in range(self.pckd_K):
            data_x = teacher_train_data[i].copy()
            data_y = data_x[self.label_names].copy()
            data_x  = data_x.drop(self.label_names, axis=1)

            teacher_models[i].fit(data_x.to_numpy(), data_y.to_numpy(), 
                                  epochs=self.pckd_teacher_epochs, batch_size=self.batch_size, verbose=0)
        
        # получение предсказаний и формирование новой выборки
        new_data_x = None
        new_data_y = None
        for i in range(self.pckd_K):
            data_x = teacher_train_data[i].copy()
            data_x  = data_x.drop(self.label_names, axis=1)
            
            vector_count = data_x.shape[0] # rows

            labels_for_merge = []
            for j in range(self.pckd_K):
                if i == j:
                    continue
                pred_labels = teacher_models[i].predict(data_x, verbose=0)
                labels_for_merge.append(pred_labels)
            
            new_labels = []
            for v_num in range(vector_count):
                vecs = []
                for t_m in labels_for_merge:
                    vecs.append(t_m[v_num])
                sum_vector= (1/self.pckd_K)*np.add.reduce(vecs)
                new_labels.append(np.round(sum_vector, 4))
            if new_data_x is None:
                new_data_x = data_x.copy()
            else:
                new_data_x = pd.concat([new_data_x, data_x.copy()], axis=0)            
            
            if new_data_y is None:
                new_data_y = pd.DataFrame(new_labels, columns=self.label_names).copy()
            else:
                new_data_y = pd.concat([new_data_y, pd.DataFrame(new_labels, columns=self.label_names)], axis=0)        
        
        new_data_x = new_data_x.reset_index(drop=True)
        new_data_y = new_data_y.reset_index(drop=True)
        new_df = pd.concat([new_data_x, new_data_y], axis=1)

        seconds = time.time() - start_time
        if self.debug:
            print(f'PCKD dataset change ended in {seconds} seconds.')

        return new_df

    def _splitDataframe(df, parts):
        row_count = df.shape[0]
        frac_size = int(row_count//parts)

        df_parts = []
        for i in range(parts):
            df_parts.append(df.iloc[i*frac_size:(i+1)*frac_size])
        return df_parts


    def learn(self, df_train:pd.DataFrame):
        start_time = time.time()
        data_x, data_y = self.prepare_dataset(df_train)

        if self.defence == 'data_spoof':
            raise NotImplemented
            if with_data_spoof:
                self.spoofed_x = LocalModel.spoof_dataset(df_train.copy(), fraction=1.0, corr_limit=self.corr_limit, truth_limit=self.truth_limit)
                self.spoofed_y = tf.one_hot(self.spoofed_x.pop('label'), 10)  
        else:
            history = self.model.fit(data_x, data_y, 
                            epochs=self.epochs, batch_size=self.batch_size, 
                            validation_split=self.validation_split)

        seconds = time.time() - start_time
        if self.debug:
            print(f'Learning was ended in {seconds} seconds.')
        return seconds, history

        
        if self.with_data_spoof:
            student_reset = False
            spoofed_data_reset = False
            print("TEACHER LEARNING")
            self.teacher.fit(data_x.to_numpy(), data_y, epochs=epochs*3, batch_size=10,  validation_split=0.1)
            if spoofed_data_reset:
                self.spoofed_x = LocalModel.spoof_dataset(self.df_train.copy(), fraction=1.0, corr_limit=self.corr_limit, truth_limit=self.truth_limit)
                self.spoofed_y = self.spoofed_x.pop('label')
            self.spoofed_y = self.teacher.predict(self.spoofed_x.to_numpy(), verbose=0)            
            
            if student_reset:
                print("MODEL LEARNING (WITH WEIGHTS RESET)")
                self.model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(150, input_shape=(784,),activation="relu"), 
                tf.keras.layers.Dense(150, activation="relu"),    
                tf.keras.layers.Dense(10, activation='softmax'), 
                ])
                self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
            else:
                print("MODEL LEARNING (NO WEIGHTS RESET)")
            self.model.fit(self.spoofed_x.to_numpy(), self.spoofed_y, epochs=epochs, batch_size=10,  validation_split=0.1)
        
        
    def extract_weights(self):
        weights = []
        for i in range(len(self.model.layers)):
            layer = self.model.layers[i]
            if self.layers_have_weights[i]:
                try:
                    l_w = layer.get_weights()[0]
                    l_bias = layer.get_weights()[1]
                    if self.defence == 'noise':
                        for neuron_num in range(len(l_w)):
                            for w_num in range(len(l_w[neuron_num])):
                                l_w[neuron_num][w_num] += np.random.normal(loc=0, scale=self.noise_scale)
                        for bias_num in range(len(l_bias)):
                            l_bias[bias_num] += np.random.normal(loc=0, scale=self.noise_scale)
                        
                    weights.append((l_w, l_bias))
                except Exception as e:
                    print(e)
                    print(layer.get_weights())
                    
        return weights

    def set_weights(self, weights):
        
        i = 0
        j= 0
        while i < len(weights):
            l_w, l_bias = weights[i]
            if not self.layers_have_weights[j]:
                j+=1                
                continue
            try:
                # Кому ставить веса?
                if self.defence == 'data_spoof':
                    self.teacher.layers[j].set_weights([l_w, l_bias])
                    if self.set_student_layer_num != 0 and i < self.set_student_layer_num:
                        self.model.layers[j].set_weights([l_w, l_bias])
                    elif np.random.rand() < self.set_student_prob:
                        self.model.layers[j].set_weights([l_w, l_bias])
                else:
                    self.model.layers[j].set_weights([l_w, l_bias])
            except Exception as e:
                print(e)
                print(weights[i])
            i += 1
            j += 1   

    def evaluate(self, df_test:pd.DataFrame):
        data_x, data_y = self.prepare_dataset(df_test)
            
        res = self.model.evaluate(data_x, data_y)
        res.pop(0)
        if self.debug:
            print(res)
        return res

    def get_confidence_score(self, data):

        predictions = self.model.predict(self.prepare_dataset(data, with_y=False), verbose=0)
        
        confs = []
        for pred in predictions:
            conf = 0
            for pred_value in pred:
                #pred_value = abs(pred_value - 0.1) 
                pred_value = round(pred_value, 5) # *10
                if pred_value == 0:
                    continue
                
                conf -= pred_value*np.log(pred_value) # минус потому что log отрицательный на всех значениях
            confs.append(conf)
        return confs

    def predict(self, data):
        return self.model.predict(data) 


if __name__ == '__main__':
    #dataframes = []
    #for i in range(1, 6):
        #dataframes.append(pd.read_csv(f'datasets\\train_mnist_{i}.csv'))
    #shadow_dataframe = pd.read_csv('datasets\\train_mnist_shadow.csv')
    #test_dataframe = pd.read_csv('datasets\\test_mnist.csv')
    with open("config.yaml", "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)

    df = pd.read_csv('prepared_datasets\\train_purchase100.csv').iloc[0:10]
    df_test = pd.read_csv('prepared_datasets\\test_purchase100.csv')
    t_m = LocalModel(df, params=config)
    t_m.learn(df)
    t_m.evaluate(df_test)
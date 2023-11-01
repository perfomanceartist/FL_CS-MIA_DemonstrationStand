import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression 
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import yaml 

class MIAClassifier:
    def __init__(self, params=dict()) -> None:
        self.methodList = params["methods"]
        self.y_label = params["y_label"]
        self.mode = params["mode"]
        self.target_metric = params["target_metric"]
        try: self.rounds = params["rounds"]
        except: self.rounds = None


        #self.methodList = ['LogisticRegression', 'DecisionTrees', 'MLP', 'SVC', 'NaiveBayes', 'KNN']
        Path("results").mkdir(parents=True, exist_ok=True) 
        logging.basicConfig(filename="attacker.log",
                    format='%(asctime)s %(message)s',
                    filemode='a')
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

    
    def _save_confusion_matrix(self, cm, method):
        #accuracy = round(accuracy, 3)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #confusion_matrix_name = f'{method} ({self.mode})\nAccuracy: {accuracy}'
        confusion_matrix_name = f'{method}'
        sns.heatmap(cm, 
                        cmap="Blues",
                        annot=True,
                        fmt='g', 
                        xticklabels=['Not Presented','Presented'],
                        yticklabels=['Not Presented','Presented'])        
        plt.xlabel('Predicted',fontsize=12)
        plt.ylabel('Actual',fontsize=12)
        plt.title(confusion_matrix_name, fontsize=16)
        #plt.show()
        plt.savefig(f'results\\{method} ({self.mode}).png')
        plt.clf()
        plt.cla()
        plt.close()

    def attack(self):
        if self.mode == "entropy":
            attacker_df = pd.read_csv('train_mia.csv')
            attacker_df = attacker_df.iloc[np.random.permutation(len(attacker_df))]
            labels = attacker_df.pop(self.y_label)

            attacker_eval_df = pd.read_csv('test_mia.csv')
            eval_labels = attacker_eval_df.pop(self.y_label)
        elif self.mode == "entropy_diffs":
            attacker_df = pd.read_csv('train_mia_diffs.csv')
            attacker_df = attacker_df.iloc[np.random.permutation(len(attacker_df))]
            labels = attacker_df.pop(self.y_label)

            attacker_eval_df = pd.read_csv('test_mia_diffs.csv')
            eval_labels = attacker_eval_df.pop(self.y_label)

        if not self.rounds is None:
            col_names = [str(round) for round in self.rounds]
            attacker_df = attacker_df[col_names]
            attacker_eval_df = attacker_eval_df[col_names]

        print(f'MODE: {self.mode}')
        infos = []
        for method in self.methodList:            

            if method == 'LogisticRegression':    
                classifier = LogisticRegression() 
            elif method == 'DecisionTrees':
                classifier =  DecisionTreeClassifier()
            elif method == 'MLP':
                classifier = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(64, ), n_iter_no_change=50)
            elif method == 'SVC':
                classifier = SVC(kernel='poly', degree=4)  
            elif method == 'NaiveBayes':
                classifier =  GaussianNB()
            elif method == 'KNN':
                classifier =  KNeighborsClassifier(n_neighbors=2, algorithm='ball_tree')
            elif method == "SGD":
                classifier = SGDClassifier()
            elif method == "RF":
                classifier = RandomForestClassifier()
            
            classifier.fit(attacker_df, labels)  
            y_pred = classifier.predict(attacker_eval_df)

            accuracy = accuracy_score(eval_labels, y_pred)
            presicion = precision_score(eval_labels, y_pred)
            recall = recall_score(eval_labels, y_pred)
            f1 = f1_score(eval_labels, y_pred)
            try:
                roc_auc = roc_auc_score(eval_labels, classifier.predict_proba(attacker_eval_df)[:,1])          
            except:
                roc_auc = 'undefined'  
            cm = confusion_matrix(eval_labels, y_pred)


            info = dict()
            info["method"] = method
            info["mode"] = self.mode
            info["metrics"] = dict()
            info["metrics"]["accuracy"] = float(round(accuracy, 3))
            info["metrics"]["precision"] = float(round(presicion, 3))
            info["metrics"]["recall"] = float(round(recall, 3))
            info["metrics"]["f1_score"] = float(round(f1, 3))
            if roc_auc != 'undefined':
                info["metrics"]["roc_auc"] = float(round(roc_auc, 3))
            else:
                info["metrics"]["roc_auc"] = roc_auc
            infos.append(info)

            yaml_string = yaml.dump(info, default_flow_style=False)
            print(yaml_string)
            self.logger.info(yaml_string)
            #print (f"{method} accuracy: {accuracy}")
            #self.logger.info(f"{method} accuracy: {accuracy_score(eval_labels, y_pred)}")
            #tn, fp, fn, tp = cm.ravel()
            #print(f"{method} confusion matrix: {cm}; TN: {tn}; TP: {tp}") 
            self._save_confusion_matrix(cm, method)
            continue

            sns.heatmap(cm, 
                        cmap="Blues",
                        annot=True,
                        fmt='g', 
                        xticklabels=['Presented','Not Presented'],
                        yticklabels=['Presented','Not Presented'])
        

        infos = sorted(infos, key=lambda info: -info["metrics"][self.target_metric])
        for info in infos:
            print(f"{info['method']} {self.target_metric}: {info['metrics'][self.target_metric]}")
        
        max_metric = 0
        max_metric_method = None
        for info in infos:
            if info["metrics"][self.target_metric] > max_metric:
                max_metric = info["metrics"][self.target_metric]
                max_metric_method = info["method"]
        print(f"Best method by {self.target_metric} is {max_metric_method} with {self.target_metric} = {max_metric}")
        self.logger.info(f"Best method by {self.target_metric} is {max_metric_method} with {self.target_metric} = {max_metric}")


            

if __name__ == "__main__":
    with open("config.yaml", "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    classifier = MIAClassifier(params=config)
    classifier.attack()
import pandas as pd
import numpy as np
import pathlib

data = np.load('..\\original_datasets\\purchase100.npz')
features = data['features']
labels = data['labels']
column_names = []
for i in range(features.shape[1]):
    column_names.append(f"feature_{i}")
features_df = pd.DataFrame(features, columns=column_names)
labels_df = pd.DataFrame(labels, columns=['label'])
labels_df = pd.get_dummies(labels, prefix='label')
#print(labels_df.head())
df = pd.concat([features_df, labels_df], axis=1)
df = df.astype('float32')
print(df.info())
print(df.head())

row_count = df.shape[0]
train_df =  df.iloc[:int(row_count*0.9)]
test_df = df.iloc[int(row_count*0.9):]

pathlib.Path("..\\prepared_datasets").mkdir(parents=True, exist_ok=True)    
train_df.to_csv(f'..\\prepared_datasets\\train_purchase100.csv', index=False)
test_df.to_csv(f'..\\prepared_datasets\\test_purchase100.csv', index=False)
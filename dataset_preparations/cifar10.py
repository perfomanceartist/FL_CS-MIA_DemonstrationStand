import pandas as pd
import yaml
import pathlib
from sklearn.model_selection import train_test_split


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

def cifar10_to_df():
    batches_names = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
    full_df = None
    for batch_name in batches_names:        
        data_batch_1 = unpickle(f"..\\original_datasets\\cifar10\\{batch_name}")
        x_train = data_batch_1['data']
        y_train = data_batch_1['labels']
        column_names = [f'pixel_{str(i+1)}' for i in range(x_train.shape[1])]
        
        df_images = pd.DataFrame(x_train, columns=column_names)
        df_labels = pd.DataFrame(y_train, columns=["label"])
        df = pd.concat([df_images, df_labels], axis=1)
        if full_df is None:
            full_df = df.copy()
        else:
            full_df = pd.concat([full_df, df], axis=0)
    full_df.to_csv("..\\original_datasets\\cifar10.csv", index=False)

    #meta_file = '..\\original_datasets\\cifar10\\batches.meta'
    #meta_data = unpickle(meta_file)
    #label_names = meta_data['label_names']
    



def _splitDataframe(df, parts):
    row_count = df.shape[0]
    frac_size = int(row_count//parts)

    df_parts = []
    for i in range(parts):
        df_parts.append(df.iloc[i*frac_size:(i+1)*frac_size])
    return df_parts

def preprocess(filename):
    data = pd.read_csv(filename)

    
    labels_df = pd.get_dummies(data['label'], prefix='label')
    labels_df = labels_df.astype('float32')
    data.pop('label')
    df = pd.concat([data, labels_df], axis=1)
    print(df.head())

    row_count = df.shape[0]
    train_df =  df.iloc[:int(row_count*0.9)]
    test_df = df.iloc[int(row_count*0.9):]

    pathlib.Path("..\\prepared_datasets").mkdir(parents=True, exist_ok=True)    
    train_df.to_csv(f'..\\prepared_datasets\\train_cifar10.csv', index=False)
    test_df.to_csv(f'..\\prepared_datasets\\test_cifar10.csv', index=False)

if __name__ == "__main__":
    cifar10_to_df()
    preprocess("..\\original_datasets\\cifar10.csv")
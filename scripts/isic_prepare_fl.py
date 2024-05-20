import os
import sys
import pandas as pd
import random

from glob import glob

import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split

current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from utils.sample_dirichlet import clients_indices

filepath = Path(__file__).resolve().parent
config = yaml.safe_load(open(filepath.joinpath("../configs/isic/scripts_conf.yaml")))

data_folder = config.get("data_folder", "~/data/isic2018")
raw_data_folder = config.get("raw_data_folder", f"{data_folder}/raw")
client_num = config.get("client_num", 6)
dirichlet_alpha = config.get("dirichlet_alpha", 0.5) 
random.seed(config.get("seed", 0))

class_names =  ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

target_folder = f'{data_folder}/fedmu'
Path(target_folder).mkdir(parents=True, exist_ok=True)

all_xray_df = pd.read_csv(f'{raw_data_folder}/ISIC2018_Task3_Training_GroundTruth.csv')
all_image_paths = {Path(x).stem: x for x in glob(os.path.join(raw_data_folder, 'ISIC*', '*', 'resized', '*.jpg'))}

all_xray_df['path'] = all_xray_df['image'].map(all_image_paths.get)

class_num = len(class_names)

train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2

train, test = train_test_split(all_xray_df,test_size=test_ratio, random_state=config.get("seed", 0))
train, val = train_test_split(train,test_size=val_ratio/(train_ratio+val_ratio), random_state=config.get("seed", 0))

test.to_csv(f'{target_folder}/test.csv', index=False)
val.to_csv(f'{target_folder}/val.csv', index=False)

# split the training data into clients by the dirichlet distribution
list_label2indices = []
for class_name in class_names:
    indices = train[train[class_name] == 1].index.tolist()
    list_label2indices.append(indices)

client_indices = clients_indices(
    list_label2indices=list_label2indices,
    num_classes=class_num,
    num_clients=client_num,
    non_iid_alpha=dirichlet_alpha,
    seed=config.get("seed", 0)
)

for client_id, indices in enumerate(client_indices):
    client_df = all_xray_df.loc[indices]
    Path(target_folder).joinpath(f'client_{client_id}').mkdir(parents=True, exist_ok=True)
    client_df.to_csv(f'{target_folder}/client_{client_id}/train.csv', index=False)

# show the distribution 
distribution = {client_id: {class_name: 0 for class_name in class_names} for client_id in range(client_num)}

for client_id, indices in enumerate(client_indices):
    client_df = all_xray_df.loc[indices]
    for class_name in class_names:
        distribution[client_id][class_name] = client_df[class_name].sum()

def get_distribution(df, class_names):
    distribution = {class_name: 0 for class_name in class_names}
    for class_name in class_names:
        distribution[class_name] = df[class_name].sum()
    return distribution

print(pd.DataFrame(distribution).T)

print('val distribution', get_distribution(val, class_names), "total", len(val))
print('test distribution', get_distribution(test, class_names), "total", len(test))

unlearn_client = config.get("unlearn_client", 0)
forgotten_set = pd.read_csv(f'{target_folder}/client_{unlearn_client}/train.csv')
forgotten_set.to_csv(f'{target_folder}/forgotten.csv', index=False)
remembered_set = []
for client_id in range(client_num):
    if client_id != unlearn_client:
        remembered_set.append(pd.read_csv(f'{target_folder}/client_{client_id}/train.csv'))
remembered_set = pd.concat(remembered_set)
remembered_set.to_csv(f'{target_folder}/remembered.csv', index=False)
print('forgotton distribution',get_distribution(forgotten_set, class_names), "total", len(forgotten_set))
print('remembered distribution',get_distribution(remembered_set, class_names), "total", len(remembered_set))
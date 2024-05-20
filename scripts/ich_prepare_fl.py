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
config = yaml.safe_load(open(filepath.joinpath("../configs/ich/scripts_conf.yaml")))

data_folder = config.get("data_folder", "~/data/ich")
raw_data_folder = config.get("raw_data_folder", f"{data_folder}/raw")
client_num = config.get("client_num", 6)
dirichlet_alpha = config.get("dirichlet_alpha", 0.5) 
random.seed(config.get("seed", 0))

class_names =  ["healthy"]

target_folder = f'{data_folder}/fedmu'
Path(target_folder).mkdir(parents=True, exist_ok=True)

all_xray_df = pd.read_csv(f'{raw_data_folder}/balanced_sampled.csv')

class_num = len(class_names) * 2

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
    healthy_indices = train[train[class_name] == 1].index.tolist()
    list_label2indices.append(healthy_indices)
    
    unhealthy_indices = train[train[class_name] == 0].index.tolist()
    list_label2indices.append(unhealthy_indices)

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


def get_distribution(df):
    healthy_count = df['healthy'].sum()
    total_count = len(df)
    unhealthy_count = total_count - healthy_count  
    
    distribution = {
        "healthy": healthy_count,
        "unhealthy": unhealthy_count
    }
    
    return distribution


# Assuming client_indices is a list of lists, where each inner list contains the indices for a client
distribution = {}

for client_id, indices in enumerate(client_indices):
    client_df = all_xray_df.loc[indices]
    # Use the get_distribution function to get the counts of healthy and unhealthy samples
    client_distribution = get_distribution(client_df)
    distribution[client_id] = client_distribution

# Convert the distribution dictionary to a DataFrame and print it
distribution_df = pd.DataFrame(distribution).T  # Transpose to have clients as rows
print(distribution_df)

print('val distribution', get_distribution(val), "total", len(val))
print('test distribution', get_distribution(test), "total", len(test))

unlearn_client = config.get("unlearn_client", 0)
forgotten_set = pd.read_csv(f'{target_folder}/client_{unlearn_client}/train.csv')
forgotten_set.to_csv(f'{target_folder}/forgotten.csv', index=False)
remembered_set = []
for client_id in range(client_num):
    if client_id != unlearn_client:
        remembered_set.append(pd.read_csv(f'{target_folder}/client_{client_id}/train.csv'))
remembered_set = pd.concat(remembered_set)
remembered_set.to_csv(f'{target_folder}/remembered.csv', index=False)
print('forgotten distribution',get_distribution(forgotten_set), "total", len(forgotten_set))
print('remembered distribution',get_distribution(remembered_set), "total", len(remembered_set))
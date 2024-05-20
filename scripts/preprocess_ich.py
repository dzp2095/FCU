import pydicom
from PIL import Image
import numpy as np

import os
import sys
import pandas as pd
import random

from glob import glob
import yaml
from pathlib import Path

current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

filepath = Path(__file__).resolve().parent
config = yaml.safe_load(open(filepath.joinpath("../configs/ich/scripts_conf.yaml")))

data_folder = config.get("data_folder", "~/data/ich")
raw_data_folder = config.get("raw_data_folder", f"{data_folder}/raw")
client_num = config.get("client_num", 6)
dirichlet_alpha = config.get("dirichlet_alpha", 0.5) 
random.seed(config.get("seed", 0))

all_xray_df = pd.read_csv(f'{raw_data_folder}/rsna-intracranial-hemorrhage-detection/stage_2_train.csv')


label = all_xray_df.Label.values
all_xray_df = all_xray_df.ID.str.rsplit("_", n=1, expand=True)
all_xray_df.loc[:, "label"] = label
all_xray_df = all_xray_df.rename({0: "id", 1: "subtype"}, axis=1)

all_xray_df = pd.pivot_table(all_xray_df, index="id", columns="subtype", values="label").reset_index()

all_xray_df['healthy'] = (all_xray_df['any'] == 0).astype(int)
all_xray_df.drop('any', axis=1, inplace=True)


# randomly sample 25000 images
healthy_sample = all_xray_df[all_xray_df['healthy'] == 1].sample(n=12500, random_state=1)
unhealthy_sample = all_xray_df[all_xray_df['healthy'] == 0].sample(n=12500, random_state=1)

balanced_sampled_df = pd.concat([healthy_sample, unhealthy_sample])

healthy_count = balanced_sampled_df['healthy'].sum()
unhealthy_count = balanced_sampled_df.shape[0] - healthy_count

print(f"Healthy: {healthy_count}")
print(f"Unhealthy: {unhealthy_count}")

all_image_paths = {Path(x).stem: x for x in glob(os.path.join(raw_data_folder, 'rsna*', '*', '*.dcm'))}

balanced_sampled_df['path'] = balanced_sampled_df['id'].map(all_image_paths.get)

def convert_dicom_to_png(dicom_path, output_path, target_size=(224, 224)):
    # read dicom
    dicom_image = pydicom.dcmread(dicom_path)
    image_array = dicom_image.pixel_array
    image_array = np.interp(image_array, (image_array.min(), image_array.max()), (0, 255))
    image_array = image_array.astype(np.uint8)
    image = Image.fromarray(image_array)
    # resize
    image = image.resize(target_size, Image.ANTIALIAS)
    image.save(output_path, format='png')

for index, row in balanced_sampled_df.iterrows():
    dicom_path = row['path']
    Path.mkdir(Path(dicom_path).parent.parent.joinpath("resized"), exist_ok=True)
    output_path = os.path.join(Path(dicom_path).parent.parent, "resized", f"{row['id']}.png")  
    convert_dicom_to_png(dicom_path, output_path)
    balanced_sampled_df.at[index, 'path'] = output_path

balanced_sampled_df.to_csv(f'{raw_data_folder}/balanced_sampled.csv', index=False)
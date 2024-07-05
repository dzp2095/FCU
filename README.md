# FCU
1. Download the ISIC dataset from [ISIC2018 Skin Lesion](https://www.kaggle.com/datasets/shonenkov/isic2018) and Kaggle Intracranial Hemorrhage dataset from [ICH](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection)

2. Run preprocess_ich.py, and then run the scprits (scripts/isic2018_prepare_fl.py and scripts/chestxray14_prepare_fl.py) to generate FL datasets, the corresponding configure file is in configs/ich/scripts_conf.yaml and configs/isic/scripts_conf.yaml

3. Run the training code.
   python3 train.py --config "../configs/ich/run_conf.yaml"  --excluded_clients 0 --is_unlearn 1
   python3 train.py --config "../configs/isic/run_conf.yaml" --excluded_clients 0 --is_unlearn 1
# citation
Please cite our paper if you find this code useful for your research.

```bibtex
@article{deng2024enable,
  title={Enable the Right to be Forgotten with Federated Client Unlearning in Medical Imaging},
  author={Deng, Zhipeng and Luo, Luyang and Chen, Hao},
  journal={arXiv preprint arXiv:2407.02356},
  year={2024}
}
```

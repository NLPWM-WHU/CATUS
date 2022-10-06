# CATUS
Code and datasets of the pre-training model for sequential POI recommendation, named CATUS.

## 1. Usage
We provide the pre-training of CATUS on Gowalla dataset in the `run_pretrain_GowallaLoc1_GowallaLoc2.sh`:
```
sh run_pretrain_GowallaLoc1_GowallaLoc2.sh
```

We provide the fine-tuning of CATUS+DS on GowallaLoc1 (Austin) and GowallaLoc2 (San Francisco) sub-datasets with SASRec as the downstream model in the `run_finetune_GowallaLoc1.sh` and `run_finetune_GowallaLoc2.sh`, respectively:
```
sh run_finetune_GowallaLoc1.sh
sh run_finetune_GowallaLoc2.sh
```

If one wants to get the recommendation results without any pre-training, just set `use_pretrain` to `0`.

## 2. Checkpoints
We provide the pre-training and fine-tuning logs in the log folder.

```
log/pretrain/COM_GowallaLoc1_GowallaLoc2
log/finetune/COM_GowallaLoc1_GowallaLoc2
```

We also provide our pre-trained models in the output folder, one can directly try them on various downstream models:
```
output/COM_GowallaLoc1_GowallaLoc2
```

## 3. Data Preprocessing
Download the large [Gowalla](https://drive.google.com/file/d/1HUG4bz_PAA29n9bpFrRM4rkmMRbW0Rv-/view?usp=sharing) dataset, and then unzip it to `data/Gowalla`. Run `Gowalla_statistics.py` to extract two city sub-datasets from the large Gowalla dataset:
```
python Gowalla_statistics.py
```
Run `POI_data_cross_city_process.py` to preprocess two city sub-datasets:
```
python POI_data_cross_city_process.py
```
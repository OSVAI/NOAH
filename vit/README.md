# ViT backbones with NOAH

For experiments on ViTs, we select backbones from [DeiT](https://arxiv.org/abs/2012.12877) and [PVT](https://arxiv.org/abs/2102.12122) families. This code is build based on the official implementation of [DeiT](https://github.com/facebookresearch/deit) and [PVT](https://github.com/whai362/PVT).

## Results and models

Results for ViT backbones with NOAH trained on ImageNet.

| Backbones         | Name                           | Params | $N$ | $r$ | Top-1 Acc(%) | Google Drive                                                                                |
|:----------------- |:------------------------------ |:------:|:---:|:---:|:------------:|:-------------------------------------------------------------------------------------------:|
| DeiT-Base         | deit_base_patch16_224          | 86.86M | 4   | 1/2 | 81.85        | [model](https://drive.google.com/file/d/1w2FX00uDSQG_W0fRh8CV5c9APC7D9tCy/view?usp=sharing) |
| + NOAH            | noah_deit_base_patch16_224     | 86.86M | 4   | 1/2 | 82.22        | [model](https://drive.google.com/file/d/1J-E1lhaNQlZchywQmzJm9pkKrJLvdGvf/view?usp=sharing) |
| DeiT-Small        | deit_small_patch16_224         | 22.06M | 4   | 1/2 | 79.78        | [model](https://drive.google.com/file/d/1NUW76MMR2N3k3r4JS_fvRACWdsd77SH_/view?usp=sharing) |
| + NOAH            | noah_deit_small_patch16_224    | 22.06M | 4   | 1/2 | 80.56        | [model](https://drive.google.com/file/d/1SatPUgds1vtkfv1QdCks_27hwRMIlgni/view?usp=sharing) |
| DeiT-Tiny (×1.0)  | deit_tiny_patch16_224          | 5.72M  | 4   | 1/2 | 72.16        | [model](https://drive.google.com/file/d/1Kl_rEyGNkaTIjJwGtTfjNcCr5HKk_dFv/view?usp=sharing) |
| + NOAH            | noah_deit_tiny_patch16_224     | 5.72M  | 4   | 1/2 | 74.29        | [model](https://drive.google.com/file/d/1RqhNCFkWQNCuh7pq020qsJm0MD9yUuH3/view?usp=sharing) |
| DeiT-Tiny (×0.75) | deit_tiny_075_patch16_224      | 3.29M  | 4   | 1/2 | 62.55        | [model](https://drive.google.com/file/d/1tj4AAuNJfImYGRL-OjnC-uQ3lucolgUK/view?usp=sharing) |
| + NOAH            | noah_deit_tiny_075_patch16_224 | 3.30M  | 4   | 1/2 | 66.64        | [model](https://drive.google.com/file/d/1DyoceZC9vmjxU_mMCFOkGmwqjIHSKKEf/view?usp=sharing) |
| DeiT-Tiny (×0.5)  | deit_tiny_050_patch16_224      | 1.53M  | 4   | 1/2 | 51.36        | [model](https://drive.google.com/file/d/18M8rx369PkaA_RtXtjRPObuODJ1btxYw/view?usp=sharing) |
| + NOAH            | noah_deit_tiny_050_patch16_224 | 1.54M  | 4   | 1/2 | 56.66        | [model](https://drive.google.com/file/d/1_4VaeiIbMEmKWmyPuEfBhYIB8UwIRaXp/view?usp=sharing) |

| Backbones        | Name              | Params | $N$ | $r$ | Top-1 Acc(%) | Google Drive                                                                                |
|:---------------- |:----------------- |:------:|:---:|:---:|:------------:|:-------------------------------------------------------------------------------------------:|
| PVT-Tiny (×1.0)  | pvt_tiny          | 13.23M | 4   | 1/2 | 75.10        | [model](https://drive.google.com/file/d/1q4wivIKXRn3nC8VexefNkRoXulB_B15q/view?usp=sharing) |
| + NOAH           | noah_pvt_tiny     | 13.24M | 4   | 1/2 | 76.51        | [model](https://drive.google.com/file/d/1vGhoJilp5bloX6mQlgymE9buftqM-A3_/view?usp=sharing) |
| PVT-Tiny (×0.75) | pvt_tiny_075      | 7.62M  | 4   | 1/2 | 71.81        | [model](https://drive.google.com/file/d/1AhYnuRm88wvCeO0JV_tFKSMYzfqQKdSC/view?usp=sharing) |
| + NOAH           | noah_pvt_tiny_075 | 7.62M  | 4   | 1/2 | 74.22        | [model](https://drive.google.com/file/d/1G4qdAANZgULbk3XP2KLRjqw3RgBqxPtS/view?usp=sharing) |
| PVT-Tiny (×0.5)  | pvt_tiny_050      | 3.54M  | 4   | 1/2 | 65.33        | [model](https://drive.google.com/file/d/133GeiITuRQkOECKfAMfh8oKqrT_fKyga/view?usp=sharing) |
| + NOAH           | noah_pvt_tiny_050 | 3.55M  | 4   | 1/2 | 68.50        | [model](https://drive.google.com/file/d/1qtsa7AbzHd-Ipet3sl1H67e1NdK4Ymzr/view?usp=sharing) |

## Training and evaluation on DeiT

Please follow [DeiT](https://github.com/facebookresearch/deit) on how to prepare the environment. Then attach our code to the origin project.

#### Training

To train DeiT models:

```shell
python -m torch.distributed.launch --nproc_per_node={ngpus} --use_env main.py \
--model {model name} --batch-size {batch size} --data-path {path to dataset} --output_dir {path to checkpoint}
```

For example, you can use following command to train DeiT-Tiny with NOAH $(r=1/2, N=4)$:

```shell
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
--model deit_tiny_patch16_224 --batch-size 128 --data-path ./datasets/ILSVRC2012 --output_dir ./checkpoints/noah_deit_tiny
```

#### Evaluation

To evaluate a pre-trained DeiT model:

```shell
python main.py --eval --resume {path to model} --model {model name} --data-path {path to dataset}
```

## Training and evaluation on PVT

Please follow [PVT](https://github.com/whai362/PVT) on how to prepare the environment. Then attach our code to the origin project.

#### Training

To train PVT models:

```shell
bash dist_train.sh {path to config file} {ngpus} --data-path {path to dataset}
```

For example, you can use following command to train PVT-Tiny with NOAH $(r=1/2, N=4)$:

```shell
bash dist_train.sh configs/pvt/pvt_tiny.py 8 --data-path ./datasets/ILSVRC2012
```

#### Evaluation

To evaluate a pre-trained PVT model:

```shell
bash dist_train.sh {path to config file} {ngpus} --data-path {path to dataset} --resume {path to model} --eval
```

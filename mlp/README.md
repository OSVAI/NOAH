# MLP backbones with NOAH

We use the popular [timm](https://github.com/rwightman/pytorch-image-models) library for experiments on the ImageNet dataset with MLP backbones. For MLPs, we select backbones from Mixer and gMLP families.

## Results and models

Results for MLP backbones with NOAH trained on ImageNet.

| Backbones             | Name                   | Params | $N$ | $r$ | Top-1 Acc(%) | Google Drive                                                                                |
|:--------------------- |:---------------------- |:------:|:---:|:---:|:------------:|:-------------------------------------------------------------------------------------------:|
| Mixer-Base            | mixer_b16_224          | 59.88M | -   | -   | 77.14        | [model](https://drive.google.com/file/d/1rnaWDRhlpCecKZTnjYUAraofXAIpXgKt/view?usp=sharing) |
| + NOAH                | noah_mixer_b16_224     | 59.88M | 4   | 1/2 | 77.49        | [model](https://drive.google.com/file/d/1zZZ3tHzItF_TW7oGpdJuj4s0-QBmpAMj/view?usp=sharing) |
| Mixer-Small    (×1.0) | mixer_s16_224          | 18.53M | -   | -   | 74.18        | [model](https://drive.google.com/file/d/1wrvSh1WjslOBx3Hlp9iXXFzk2PIH6uwy/view?usp=sharing) |
| + NOAH                | noah_mixer_s16_224     | 18.54M | 4   | 1/2 | 75.09        | [model](https://drive.google.com/file/d/1fuZawsAVSRqqAUQzt7wi425MGKxOx_pK/view?usp=sharing) |
| Mixer-Small (×0.75)   | mixer_s16_075_224      | 10.75M | -   | -   | 71.13        | [model](https://drive.google.com/file/d/1Ry4EiOALuyqP27Z_fkP3aBYWlZo_kk3R/view?usp=sharing) |
| + NOAH                | noah_mixer_s16_075_224 | 10.76M | 4   | 1/2 | 72.32        | [model](https://drive.google.com/file/d/13EH3iZ7WqDtIZbj9mpONzXMF-wfaepOv/view?usp=sharing) |
| Mixer-Small (×0.5)    | mixer_s16_050_224      | 5.07M  | -   | -   | 65.22        | [model](https://drive.google.com/file/d/1jA-ycxakDSaNSWriCqVz1NcVABSpnnlk/view?usp=sharing) |
| + NOAH                | noah_mixer_s16_050_224 | 5.08M  | 4   | 1/2 | 66.81        | [model](https://drive.google.com/file/d/1SdS4EgAOPg2SojiCnnbskv6d5pWx4kYZ/view?usp=sharing) |

| Backbones           | Name                   | Params | $N$ | $r$ | Top-1 Acc(%) | Google Drive                                                                                |
|:------------------- |:---------------------- |:------:|:---:|:---:|:------------:|:-------------------------------------------------------------------------------------------:|
| gMLP-Small          | gmlp_s16_224           | 19.42M | -   | -   | 79.65        | [model](https://drive.google.com/file/d/14IBEnTbU43SutWVos201v7IjAQBOJ2ma/view?usp=sharing) |
| + NOAH              | noah_gmlp_s16_224      | 19.42M | 4   | 1/2 | 79.95        | [model](https://drive.google.com/file/d/1C3xeP201OmzW3JdplaPmodsjUuQ2_Q-a/view?usp=sharing) |
| gMLP-Tiny    (×1.0) | gmlp_ti16_224          | 5.87M  | -   | -   | 72.05        | [model](https://drive.google.com/file/d/1x5H2IpdQOGVtQ2iv9uXvkQYNxqYARLvZ/view?usp=sharing) |
| + NOAH              | noah_gmlp_ti16_224     | 5.87M  | 4   | 1/2 | 73.39        | [model](https://drive.google.com/file/d/1KETgcHuacDo9BPzkFXAnnoTyNAlP9zSS/view?usp=sharing) |
| gMLP-Tiny (×0.75)   | gmlp_ti16_075_224      | 3.91M  | -   | -   | 65.95        | [model](https://drive.google.com/file/d/1B9lEYLE81QTMxCFFn8SRXC56GIStr8ht/view?usp=sharing) |
| + NOAH              | noah_gmlp_ti16_075_224 | 3,91M  | 4   | 1/2 | 67.71        | [model](https://drive.google.com/file/d/1Yy-b9-QpwiMb07W2-WL94pV5V0ZAcNYm/view?usp=sharing) |
| gMLP-Tiny (×0.5)    | gmlp_ti16_050_224      | 2.41M  | -   | -   | 54.99        | [model](https://drive.google.com/file/d/15G6S-1iY1326-343tiEb7Z05SLuscX5B/view?usp=sharing) |
| + NOAH              | noah_gmlp_ti16_050_224 | 2.41M  | 4   | 1/2 | 56.89        | [model](https://drive.google.com/file/d/1mKS8ElM_s8Qp6TrjsI0wZBhw7LkGGPFP/view?usp=sharing) |

## Training

Please follow [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) on how to prepare the environment. Then attach our codes to the origin project.

To train a MLP-based network:

```shell
bash distributed_train.sh {ngpus} {path to dataset} --config configs/default_noah.yaml --model {model name}
```

For example, you can use following command to train gMLP-Tiny $(r=1/2, N=4)$:

```shell
bash distributed_train.sh 8 ./datasets/ILSVRC2012 --config configs/default_noah.yaml --model noah_gmlp_ti16_224
```

You can add **--amp** to enable Automatic Mixed Precision to reduce memory usage and speed up training.

## Evaluation

To evaluate a pre-trained model:

```shell
python validate.py ./datasets/ILSVRC2012 --model {model name} --checkpoint {path to pre-trained model} --crop-pct 0.9
```

# VA-Count
[ECCV 2024] Zero-shot Object Counting with Good Exemplars
[[paper](https://arxiv.org/abs/2407.04948)]   
![figure](figure.png)
# Zero-shot Object Counting with Good Exemplars
## News
VA-Count is accepted by ECCV2024. 
Our code will be available soon！
## Overview 
Overview of the proposed method. The proposed method focuses on two main elements: the Exemplar Enhancement Module (EEM) for improving exemplar quality through a patch selection integrated with Grounding DINO, and the Noise Suppression Module (NSM) that distinguishes between positive and negative class samples using density maps. It employs a Contrastive Loss function to refine the precision in identifying target class objects from others in an image.
## Environment
```
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install timm==0.3.2
pip install numpy
pip install matplotlib tqdm 
pip install tensorboard
pip install scipy
pip install imgaug
pip install opencv-python
pip3 install hub
```
### For more information on Grounding DINO, please refer to the following link: 
https://github.com/IDEA-Research/GroundingDINO .
We are very grateful for the Grounding DINO approach, which has been instrumental in our work！

## Datasets

* [FSC147](https://github.com/cvlab-stonybrook/LearningToCountEverything)

* [CARPK](https://lafi.github.io/LPN/)

Preparing the datasets as follows:

```
./data/
|--FSC147
|  |--images_384_VarV2
|  |  |--2.jpg
|  |  |--3.jpg
|  |--gt_density_map_adaptive_384_VarV2
|  |  |--2.npy
|  |  |--3.npy
|  |--annotation_FSC147_384.json
|  |--Train_Test_Val_FSC_147.json
|  |--ImageClasses_FSC147.txt
|  |--train.txt
|  |--test.txt
|  |--val.txt

```
## Inference
+  For inference, you can download the model from [Baidu-Disk](https://pan.baidu.com/s/11sbdDYLDfTOIPx5pZvBpmw?pwd=paeh), passward:paeh
```
python FSC_test.py --output_dir ./data/out/results_base --resume ./data/checkpoint_FSC.pth

```
## Generate exemplars
```
python groundingbi.py
```

## Train

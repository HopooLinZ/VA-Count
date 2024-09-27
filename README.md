# VA-Count
[ECCV 2024] Zero-shot Object Counting with Good Exemplars
[[paper](https://arxiv.org/abs/2407.04948)]   
![figure](figure.png)
# Zero-shot Object Counting with Good Exemplars
## News🚀
* **2024.09.26**: Our inference code has been updated, and the code for selecting exemplars and the training code will be coming soon.
* **2024.07.02**: VA-Count is accepted by ECCV2024.
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
|--CARPK/
|  |--Annotations/
|  |--Images/
|  |--ImageSets/
```
## Inference
+  For inference, you can download the model from [Baidu-Disk](https://pan.baidu.com/s/11sbdDYLDfTOIPx5pZvBpmw?pwd=paeh), passward:paeh
```
python FSC_test.py --output_dir ./data/out/results_base --resume ./data/checkpoint_FSC.pth
```
## Single and Multiple Object Classifier Training
```
python datasetmake.py
python biclassify.py
```
## Generate exemplars
```
python grounding_pos.py
python grounding_neg.py
```

## Train

```
CUDA_VISIBLE_DEVICES=0 python FSC_pretrain.py \
    --epochs 500 \
    --warmup_epochs 10 \
    --blr 1.5e-4 --weight_decay 0.05
```
```
CUDA_VISIBLE_DEVICES=0 python FSC_train.py --epochs 1000 --batch_size 8 --lr 1e-5 --output_dir ./data/out/
```

## Citation

```
@inproceedings{zhu2024zero,
  title={Zero-shot Object Counting with Good Exemplars},
  author={Zhu, Huilin and Yuan, Jingling and Yang, Zhengwei and Guo, Yu and Wang, Zheng and Zhong, Xian and He, Shengfeng},
  booktitle={Proceedings of the European Conference on Computer Vision},
  year={2024}
}
```

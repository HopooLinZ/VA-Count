import torch
import os
import clip
import inflect
import argparse
from torchvision.ops import box_convert
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict
from PIL import Image
import numpy as np
import json
import torch.nn as nn
import torch.nn.functional as F

# 定义全局变量
device = "cuda" if torch.cuda.is_available() else "cpu"
BOX_THRESHOLD = 0.05
TEXT_THRESHOLD = 0.05

# 初始化inflect引擎
p = inflect.engine()

# 定义 ClipClassifier 类
class ClipClassifier(nn.Module):
    def __init__(self, clip_model, embed_dim=512):
        super(ClipClassifier, self).__init__()
        self.clip_model = clip_model.to(device)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(clip_model.visual.output_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim, 2)  # 二分类

    def forward(self, images):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images).float().to(device)
        x = self.fc(image_features)
        x = F.relu(x)
        logits = self.classifier(x)
        return logits

# 加载 CLIP 模型
clip_model, preprocess = clip.load("ViT-B/32", device)
clip_model.eval()

# 初始化并加载二分类模型
binary_classifier = ClipClassifier(clip_model).to(device)
model_weights_path = './data/out/classify/best_model.pth'
binary_classifier.load_state_dict(torch.load(model_weights_path, map_location=device))
binary_classifier.eval()

# 判断 patch 是否有效
def is_valid_patch(patch, binary_classifier, preprocess, device):
    if patch.size[0] <= 0 or patch.size[1] <= 0:
        return False
    patch_tensor = preprocess(patch).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = binary_classifier(patch_tensor)
        probabilities = torch.softmax(logits, dim=1)
        prob_label_1 = probabilities[0, 1]
    return prob_label_1.item() > 0.8

# 处理图片的主函数
def process_images(text_file_path, dataset_path, model, preprocess, clip_model, output_folder, device='cpu'):
    boxes_dict = {}
    with open(text_file_path, 'r') as f:
        for line in f:
            image_name, class_name = line.strip().split('\t')
            print(f"Processing image: {image_name}")
            text_prompt = class_name + ' .'
            image_path = os.path.join(dataset_path, image_name)
            img = Image.open(image_path).convert("RGB")
            image_source, image = load_image(image_path)
            h, w, _ = image_source.shape
            boxes, logits, _ = predict(model, image, text_prompt, BOX_THRESHOLD, TEXT_THRESHOLD)
            patches = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")

            top_patches = []
            for i, (box, logit) in enumerate(zip(patches, logits)):
                box = box.cpu().numpy() * np.array([w, h, w, h], dtype=np.float32)
                x1, y1, x2, y2 = box.astype(int)
                x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
                patch = img.crop((x1, y1, x2, y2))

                if patch.size == (0, 0) or not is_valid_patch(patch, binary_classifier, preprocess, device) or x2 - x1 > w / 2 or y2 - y1 > h / 2 or y2 - y1 < 5 or x2 - x1 < 5:
                    print(f"Skipping patch due to binary classifier at box {box}")
                    continue
                top_patches.append((i, logit))

            top_patches.sort(key=lambda x: x[1], reverse=True)
            top_3_indices = [patch[0] for patch in top_patches[:3]]

            # 确保每张图像都有三个边界框
            while len(top_3_indices) < 3:
                if len(top_3_indices) > 0:
                    top_3_indices.append(top_3_indices[-1])
                else:
                    default_box = torch.tensor([0, 0, 20 / w, 20 / h]).unsqueeze(0)
                    patches = torch.cat((patches, default_box.to(boxes.device)), dim=0)
                    top_3_indices.append(len(patches) - 1)

            boxes_dict[image_name] = [patches[idx].cpu().numpy().tolist() * np.array([w, h, w, h], dtype=np.float32) for idx in top_3_indices]

    return boxes_dict

# 主函数
def main(args):
    # 设置固定的默认路径
    model_config = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    model_weights = "GroundingDINO/weights/groundingdino_swint_ogc.pth"
    output_folder = os.path.join(args.root_path, "annotated_images")

    # 根据 root_path 设置路径
    text_file_path = os.path.join(args.root_path, "ImageClasses_FSC147.txt")
    dataset_path = os.path.join(args.root_path, "images_384_VarV2")
    input_json_path = os.path.join(args.root_path, "annotation_FSC147_384_old.json")
    output_json_path = os.path.join(args.root_path, "annotation_FSC147_pos.json")
    
    os.makedirs(output_folder, exist_ok=True)

    # 加载 GroundingDINO 模型
    model = load_model(model_config, model_weights, device=device)

    # 处理图片并生成边界框
    boxes_dict = process_images(text_file_path, dataset_path, model, preprocess, clip_model, output_folder, device=device)

    # 更新 JSON 文件
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    for image_name, boxes in boxes_dict.items():
        if image_name in data:
            new_boxes = [[[x1, y1], [x1, y2], [x2, y2], [x2, y1]] for x1, y1, x2, y2 in boxes]
            data[image_name]["box_examples_coordinates"] = new_boxes

    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Processing Script")
    parser.add_argument("--root_path", type=str, required=True, help="Root path to the dataset and output files")
    args = parser.parse_args()
    main(args)

import torch
import os
import inflect
import argparse
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict
from PIL import Image
import numpy as np
from torchvision.ops import box_convert
import json
import torch.nn as nn
import torch.nn.functional as F
import clip

# 定义全局变量
device = "cuda" if torch.cuda.is_available() else "cpu"

# 阈值设置
BOX_THRESHOLD = 0.02
TEXT_THRESHOLD = 0.02
BOX_THRESHOLD_class = 0.01
TEXT_THRESHOLD_class = 0.01

# 初始化inflect引擎
p = inflect.engine()

# 将单词转换为单数形式的函数
def to_singular(word):
    singular_word = p.singular_noun(word)
    return singular_word if singular_word else word

# 定义ClipClassifier类
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

# 初始化和加载二分类模型
clip_model, preprocess = clip.load("ViT-B/32", device)
binary_classifier = ClipClassifier(clip_model).to(device)

# 加载保存的权重
model_weights_path = './data/out/classify/best_model.pth'
binary_classifier.load_state_dict(torch.load(model_weights_path, map_location=device))

# 确认模型已经被设置为评估模式
binary_classifier.eval()

# 计算两个边界框的IoU
def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    intersection_x1 = max(x1, x2)
    intersection_y1 = max(y1, y2)
    intersection_x2 = min(x1 + w1, x2 + w2)
    intersection_y2 = min(y1 + h1, y2 + h2)

    intersection_area = max(intersection_x2 - intersection_x1, 0) * max(intersection_y2 - intersection_y1, 0)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0

    return iou

# 检查patch是否有效
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
def process_images(text_file_path, dataset_path, model, preprocess, binary_classifier, output_folder, device='cpu'):
    boxes_dict = {}

    with open(text_file_path, 'r') as f:
        for line in f:
            image_name, class_name = line.strip().split('\t')
            print(f"Processing image: {image_name}")
            text_prompt = class_name + ' .'
            object_prompt = "object ."
            image_path = os.path.join(dataset_path, image_name)
            img = Image.open(image_path).convert("RGB")
            image_source, image = load_image(image_path)
            h, w, _ = image_source.shape
            boxes_object, logits_object, _ = predict(model, image, object_prompt, BOX_THRESHOLD, TEXT_THRESHOLD)
            boxes_class, logits_class, _ = predict(model, image, text_prompt, BOX_THRESHOLD_class, TEXT_THRESHOLD_class)

            patches_object = box_convert(boxes_object, in_fmt="cxcywh", out_fmt="xyxy")
            patches_class = box_convert(boxes_class, in_fmt="cxcywh", out_fmt="xyxy")

            top_patches = []
            iou_matrix = np.zeros((len(boxes_object), len(boxes_class)))
            
            for j, box_class in enumerate(patches_class):
                box_object_class = box_class.cpu().numpy() * np.array([w, h, w, h], dtype=np.float32)
                x1_, y1_, x2_, y2_ = box_object_class.astype(int)
                x1_, y1_, x2_, y2_ = max(x1_, 0), max(y1_, 0), min(x2_, w), min(y2_, h)
                patch_ = img.crop((x1_, y1_, x2_, y2_)) 
                if x2_ - x1_ > w / 2 or y2_ - y1_ > h / 2 or not is_valid_patch(patch_, binary_classifier, preprocess, device):    
                    print(f"Skipping patch at box {box_class}")
                    continue
                for i, box_object in enumerate(patches_object):
                    iou_matrix[i][j] = calculate_iou(box_object.cpu().numpy(), box_class.cpu().numpy())
            
            for i, box_object in enumerate(patches_object):
                max_iou = np.max(iou_matrix[i])
                if max_iou < 0.5:
                    box_object = box_object.cpu().numpy() * np.array([w, h, w, h], dtype=np.float32)
                    x1, y1, x2, y2 = box_object.astype(int)
                    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
                    patch = img.crop((x1, y1, x2, y2))
                    if patch.size == (0, 0) or not is_valid_patch(patch, binary_classifier, preprocess, device) or x2 - x1 > w / 2 or y2 - y1 > h / 2 or y2 - y1 < 5 or x2 - x1 < 5:
                        print(f"Skipping patch at box {box_object}")
                        continue
                    patch_logits = logits_object[i]
                    top_patches.append((i, patch_logits.item()))

            top_patches.sort(key=lambda x: x[1], reverse=True)
            top_3_indices = [patch[0] for patch in top_patches[:3]]

            while len(top_3_indices) < 3:
                if len(top_3_indices) > 0:
                    top_3_indices.append(top_3_indices[-1])
                else:
                    default_box = torch.tensor([0,0,20/w,20/h]).unsqueeze(0)
                    patches_object = torch.cat((patches_object, default_box.to(boxes_object.device)), dim=0)
                    top_3_indices.append(len(patches_object) - 1)

            boxes_dict[image_name] = [patches_object[idx].cpu().numpy().tolist() * np.array([w, h, w, h], dtype=np.float32) for idx in top_3_indices]

    return boxes_dict

def main(args):
    # 设置固定的默认路径
    model_config = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    model_weights = "GroundingDINO/weights/groundingdino_swint_ogc.pth"
    
    # 根据root_path设置路径
    text_file_path = os.path.join(args.root_path, "ImageClasses_FSC147.txt")
    dataset_path = os.path.join(args.root_path, "images_384_VarV2")
    input_json_path = os.path.join(args.root_path, "annotation_FSC147_384.json")
    output_json_path = os.path.join(args.root_path, "annotation_FSC147_neg.json")
    output_folder = os.path.join(args.root_path, "annotated_images_n")
    
    os.makedirs(output_folder, exist_ok=True)

    # 加载GroundingDINO模型
    model = load_model(model_config, model_weights, device=device)

    # 处理图片并生成边界框
    boxes_dict = process_images(text_file_path, dataset_path, model, preprocess, binary_classifier, output_folder, device=device)

    # 更新JSON文件
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

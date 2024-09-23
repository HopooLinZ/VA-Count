from PIL import Image
import os
import random

def is_image_file(filename):
    """判断文件是否是图像文件"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']  # 支持的图像文件扩展名列表
    return any(filename.lower().endswith(ext) for ext in image_extensions)

def random_crop(img, size=(256, 256)):
    """从给定的图片中随机裁剪出指定大小的区域"""
    width, height = img.size
    crop_width, crop_height = size

    if width < crop_width or height < crop_height:
        return None  # 如果图片尺寸小于裁剪尺寸，则返回None

    x_left = random.randint(0, width - crop_width)
    y_upper = random.randint(0, height - crop_height)

    return img.crop((x_left, y_upper, x_left + crop_width, y_upper + crop_height))

# 文件夹路径设置（根据实际情况修改）
single_object_folder = './data/FSC147/box'
multiple_objects_folder = './data/FSC147/images_384_VarV2'
output_folder = './data/FSC147/one'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

output_txt_path = os.path.join(output_folder, 'labels.txt')
with open(output_txt_path, 'w') as f:
    for folder, label in [(single_object_folder, 'one'), (multiple_objects_folder, 'more')]:
        for filename in os.listdir(folder):
            if is_image_file(filename):  # 只处理图像文件
                img_path = os.path.join(folder, filename)
                img = Image.open(img_path)

                # 保存原图并记录到txt文件
                original_img_output_path = os.path.join(output_folder, filename)
                img.save(original_img_output_path)
                f.write(f"{filename},{label}\n")

                # 从原图中随机裁剪并保存裁剪图像
                for size in [(256, 384), (256, 256), (384, 384),(128,256),(256,128)]:
                    img_cropped = random_crop(img, size=size)
                    if img_cropped:
                        cropped_img_output_path = os.path.join(output_folder, f"{filename[:-4]}_random_{size[0]}x{size[1]}.jpg")
                        img_cropped.save(cropped_img_output_path)
                        f.write(f"{filename[:-4]}_random_{size[0]}x{size[1]}.jpg,{label}\n")

print("数据集准备完成。")

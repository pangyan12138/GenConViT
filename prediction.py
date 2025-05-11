import os
import argparse
import json
import torch
from time import perf_counter
from datetime import datetime
import pandas as pd
from PIL import Image
import cv2
from torchvision import transforms
from torch.utils.data import dataset
from model.pred_func import *
from model.config import load_config

config = load_config()
print('CONFIG')
print(config)

def is_image(file_path):
    return file_path.lower().endswith(".png")

class FolderDataset(dataset.Dataset):
    def __init__(self, folder, transforms):
        self.folder = folder
        self.img_list = open(os.path.join(folder, "img_list.txt"), "r").readlines()
        self.face_info = open(os.path.join(folder, "face_info.txt"), "r").readlines()
        assert len(self.img_list) == len(self.face_info)
        self.transforms = transforms

    def __len__(self):
        return len(self.img_list)

    def get_img_name(self):
        return self.img_list

    def read_crop_face(self, idx, scale = 1.3):
        img = cv2.imread(os.path.join(self.folder, "imgs", self.img_list[idx].strip()))
        height, width = img.shape[:2]
        box = self.face_info[idx].split(" ")
        box = [float(x) for x in box]
        x1, y1, x2, y2 = box[:4]
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        size_bb = int(max(x2 - x1, y2 - y1) * scale)
        x1 = max(int(center_x - size_bb // 2), 0)
        y1 = max(int(center_y - size_bb // 2), 0)
        size_bb = min(width - x1, size_bb)
        size_bb = min(height - y1, size_bb)
        cropped_face = img[y1:y1 + size_bb, x1:x1 + size_bb]
        return cropped_face

    def __getitem__(self, idx):
        img = self.read_crop_face(idx)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = self.transforms(img)
        return img


# def pred_img(img_tensor, model):
#     model.eval()
#     with torch.no_grad():
#         output = model(img_tensor)
#         print(output)
#         prob = torch.sigmoid(output).item()
#         label = 1 if prob > 0.5 else 0
#         return label, prob

def pred_img(img_tensor, model):
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 禁用梯度计算
        # 执行模型预测
        output = model(img_tensor)
        print(f"Model output:\n{output}")
        # 对每个元素应用 Sigmoid
        sigmoid_output = torch.sigmoid(output)
        print(f"Sigmoid output:\n{sigmoid_output}")
        # 按列求平均值
        avg_output = torch.mean(sigmoid_output, dim=0)  # 按列求平均值
        print(f"Average output:\n{avg_output}")

        # avg_output = torch.mean(output, dim=0)  # 按列求平均值
        # print(avg_output)
        # # # Step 3: 对最大值应用 Softmax函数
        # prob_val = torch.softmax(avg_output, dim=0)  # 转换为概率值
        # prob_val = prob_val[1].item()
                
        if avg_output[1] > avg_output[0]:
            # 取第一个值并应用sigmoid函数
            prob_val = avg_output[1].item()
        else:
            # 取第二个值应用sigmoid函数，然后用1减去这个值
            prob_val = (1 - avg_output[0]).item()

        # 根据阈值确定分类标签
        label = 1 if prob_val > 0.5 else 0  # 阈值为 0.5

        return label, prob_val

def imgs(ed_weight, vae_weight, root_dir="sample_prediction_data", dataset=None, net=None, fp16=False):
    img_names = []
    predictions = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_genconvit(config, net, ed_weight, vae_weight, fp16)

    count = 0

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = FolderDataset(root_dir, transform)

    start = perf_counter()
    for idx in range(len(dataset)):
        try:
            img_tensor = dataset[idx].unsqueeze(0)  # add batch dim
            if fp16:
                img_tensor = img_tensor.half()

            # 将图像张量移动到 GPU 上
            img_tensor = img_tensor.to(device)

            y, y_val = pred_img(img_tensor, model)
            img_name = dataset.img_list[idx].strip()

            img_names.append(img_name)
            predictions.append(y_val)

            print(f"Prediction: {img_name} -> {y_val:.4f} ({real_or_fake(y)})")
            count += 1

        except Exception as e:
            print(f"Error processing image {idx}: {str(e)}")

    total_time = perf_counter() - start

    output_path = os.path.join("result", f"prediction_image_{net}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.xlsx")
    with pd.ExcelWriter(output_path) as writer:
        pd.DataFrame({"img_names": img_names, "predictions": predictions}).to_excel(writer, sheet_name="predictions", index=False)
        pd.DataFrame({"Data Volume": [count], "Time": [total_time]}).to_excel(writer, sheet_name="time", index=False)

    print(f"\nResults saved to {output_path}\n")
    return {"count": count, "time": total_time}

def gen_parser():
    parser = argparse.ArgumentParser("GenConViT prediction")
    parser.add_argument("--p", type=str, help="video or image path")
    parser.add_argument("--f", type=int, help="number of frames to process for prediction")
    parser.add_argument("--d", type=str, help="dataset type, dfdc, faceforensics, timit, celeb, image")
    parser.add_argument("--s", help="model size type: tiny, large.")
    parser.add_argument("--e", nargs='?', const='genconvit_ed_inference', default='genconvit_ed_inference', help="weight for ed.")
    parser.add_argument("--v", '--value', nargs='?', const='genconvit_vae_inference', default='genconvit_vae_inference', help="weight for vae.")
    parser.add_argument("--fp16", type=str, help="half precision support")

    args = parser.parse_args()
    path = args.p
    num_frames = args.f if args.f else 15
    dataset = args.d if args.d else "other"
    fp16 = True if args.fp16 else False

    net = 'genconvit'
    ed_weight = 'genconvit_ed_inference'
    vae_weight = 'genconvit_vae_inference'

    if args.e and args.v:
        ed_weight = args.e
        vae_weight = args.v
    elif args.e:
        net = 'ed'
        ed_weight = args.e
    elif args.v:
        net = 'vae'
        vae_weight = args.v

    print(f'\nUsing {net}\n')

    if args.s:
        if args.s in ['tiny', 'large']:
            config["model"]["backbone"] = f"convnext_{args.s}"
            config["model"]["embedder"] = f"swin_{args.s}_patch4_window7_224"
            config["model"]["type"] = args.s

    return path, dataset, num_frames, net, fp16, ed_weight, vae_weight

def main():
    start_time = perf_counter()
    path, dataset, num_frames, net, fp16, ed_weight, vae_weight = gen_parser()

    if dataset == "image":
        result = imgs(ed_weight, vae_weight, path, dataset, net, fp16)
    else:
        result = (
            globals()[dataset](ed_weight, vae_weight, path, dataset, num_frames, net, fp16)
            if dataset in ["dfdc", "faceforensics", "timit", "celeb"]
            else vids(ed_weight, vae_weight, path, dataset, num_frames, net, fp16)
        )

        curr_time = datetime.now().strftime("%B_%d_%Y_%H_%M_%S")
        file_path = os.path.join("result", f"prediction_{dataset}_{net}_{curr_time}.json")

        with open(file_path, "w") as f:
            json.dump(result, f)

    end_time = perf_counter()
    print("\n\n--- %s seconds ---" % (end_time - start_time))

if __name__ == "__main__":
    main()
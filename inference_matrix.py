import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt

from backbones import get_model


@torch.no_grad()
def inference_folder(weight, name, folder_path):
    
    image_files = [os.path.join(folder_path, f)
                   for f in os.listdir(folder_path)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if len(image_files) == 0:
        print("❌ No images found in folder:", folder_path)
        return

    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight, map_location='cpu'))
    net.eval()


    feats = []
    for path in image_files:
        img = cv2.imread(path)
        img = cv2.resize(img, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        img.div_(255).sub_(0.5).div_(0.5)
        feat = net(img)
        feat = F.normalize(feat, p=2, dim=1)  # 正規化方便算 cosine
        feats.append(feat.cpu().numpy())

    feats = np.vstack(feats)
    n = len(image_files)
    print(f"✅ Extracted features for {n} images")

    similarity = np.dot(feats, feats.T)


    plt.figure(figsize=(8, 6))
    sns.heatmap(similarity, annot=False, cmap='viridis')
    plt.title("Cosine Similarity Matrix of Faces")
    plt.xlabel("Image Index")
    plt.ylabel("Image Index")
    plt.tight_layout()
    plt.show()


    np.save("similarity_matrix.npy", similarity)
    print("💾 Saved similarity matrix to similarity_matrix.npy")


    mean_sim = (np.sum(similarity) - np.trace(similarity)) / (n * (n - 1))
    print(f"📊 Average pairwise similarity: {mean_sim:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ArcFace Folder Inference')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, required=True, help='path to model weights')
    parser.add_argument('--imgs_path', type=str, required=True, help='folder containing images')
    args = parser.parse_args()

    inference_folder(args.weight, args.network, args.imgs_path)

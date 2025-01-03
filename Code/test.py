import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import os

# 定義自編碼器模型結構（與訓練時相同）
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # 編碼器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # 解碼器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 載入訓練好的模型
def load_trained_model(model, model_path):
    model.load_state_dict(torch.load(model_path))  # 載入模型的權重
    model.eval()  # 設定為評估模式
    print(f"模型已載入：{model_path}")
    return model

# 載入自定義的BMP圖片
def load_image(image_path):
    image = Image.open(image_path)
    # 使用相同的轉換處理
    transform = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(image).unsqueeze(0)  # 加一個batch維度

# 顯示圖片的輔助函數
def show_image(tensor_image, title=""):
    # 反向正規化並轉換為numpy格式
    image = tensor_image.squeeze(0).cpu().detach().numpy()
    image = (image + 1) / 2  # 反正規化到[0, 1]範圍
    plt.imshow(image.transpose(1, 2, 0))  # 轉換為HWC格式
    plt.title(title)
    plt.show()

# 儲存圖片為BMP格式
def save_image_as_bmp(tensor_image, output_path):
    # 反正規化並轉換為PIL格式
    image = tensor_image.squeeze(0).cpu().detach().numpy()
    image = (image + 1) / 2  # 反正規化到[0, 1]範圍
    image = (image * 255).astype('uint8')  # 放大到[0, 255]範圍並轉為uint8型態
    image_pil = Image.fromarray(image.transpose(1, 2, 0))  # 轉換為PIL格式（HWC）
    image_pil.save(output_path, format="BMP")  # 儲存為BMP檔案
    print(f"圖片已儲存至 {output_path}")

# 計算壓縮率
def calculate_compression_rate(original_image, compressed_image):
    # 取得原始圖像大小 (高 x 寬 x 通道)
    original_size = original_image.size(2) * original_image.size(3) * original_image.size(1)  # C x H x W
    # 取得編碼後圖像的大小
    encoded_size = compressed_image.size(1) * compressed_image.size(2) * compressed_image.size(3)  # C x H x W
    compression_rate = encoded_size / original_size
    return compression_rate

# 計算均方誤差 (MSE)
def calculate_mse(original_image, reconstructed_image):
    mse = torch.mean((original_image - reconstructed_image) ** 2)
    return mse.item()

# 計算峰值信噪比 (PSNR)
def calculate_psnr(mse, max_pixel_value=255.0):
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
    return psnr

# 測試圖片並計算壓縮率與失真率
def test_and_calculate_metrics(model, image_path):
    model.eval()  # 設定為評估模式
    image = load_image(image_path)
    with torch.no_grad():
        reconstructed_image = model(image)
    
    # 計算壓縮率
    compression_rate = calculate_compression_rate(image, reconstructed_image)
    
    # 計算 MSE
    mse = calculate_mse(image, reconstructed_image)
    
    # 計算 PSNR
    psnr = calculate_psnr(mse)
    
    print(f"壓縮率: {compression_rate:.4f}")
    print(f"均方誤差 (MSE): {mse:.4f}")
    print(f"峰值信噪比 (PSNR): {psnr:.4f} dB")
    
    # 顯示原圖與重建圖
    # show_image(image, "Original Image")
    # show_image(reconstructed_image, "Reconstructed Image")
    show_images(image, reconstructed_image, "Original Image", "Reconstructed Image")

# 顯示兩張圖片的輔助函數
def show_images(image1, image2, title1="Image 1", title2="Image 2"):
    # 反向正規化並轉換為numpy格式
    image1 = image1.squeeze(0).cpu().detach().numpy()
    image1 = (image1 + 1) / 2  # 反正規化到[0, 1]範圍
    
    image2 = image2.squeeze(0).cpu().detach().numpy()
    image2 = (image2 + 1) / 2  # 反正規化到[0, 1]範圍
    
    # 使用subplots顯示兩張圖
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1列2行的排列方式
    
    # 顯示第一張圖片
    axes[0].imshow(image1.transpose(1, 2, 0))  # 轉換為HWC格式
    axes[0].set_title(title1)
    axes[0].axis('off')  # 關閉坐標軸

    # 顯示第二張圖片
    axes[1].imshow(image2.transpose(1, 2, 0))  # 轉換為HWC格式
    axes[1].set_title(title2)
    axes[1].axis('off')  # 關閉坐標軸
    
    plt.show()

# 載入訓練好的模型
model = Autoencoder()
model_path = 'Model/autoencoder_model.pth'  # 替換為你的模型檔案路徑
model = load_trained_model(model, model_path)

test_image_path = 'Test_IMG/LenaRGB.bmp'  # 替換為你自己的 BMP 檔案路徑
output_image_path = 'Final_IMG/123.bmp'  # 儲存重建圖片的檔案路徑

# 測試圖片並計算指標
test_and_calculate_metrics(model, test_image_path)

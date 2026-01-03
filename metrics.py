from skimage.metrics import structural_similarity as ssim
import torch
import numpy as np
import cv2


class Metrics:
    def __init__(self, image1, image2, device='cuda', lpips_model=None):
        self.image1 = image1
        self.image2 = image2
        self.device = device
        self.lpips_model = lpips_model

        self.image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        self.image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        self.mse = self.get_mse()
        self.inv_ssim = self.get_inv_ssim()

        # 🔑 IMPORTANT FIX
        if self.lpips_model is not None:
            self.lpips = self.get_lpips()
        else:
            self.lpips = None

        self.difference = self.get_difference()

    def get_mse(self):
        return np.mean(
            (self.image1_gray.astype("float") -
             self.image2_gray.astype("float")) ** 2
        )

    def get_inv_ssim(self):
        ssim_val, _ = ssim(
            self.image1_gray,
            self.image2_gray,
            full=True,
            data_range=255
        )
        return 1.0 - ssim_val

    @staticmethod
    def lpips_preprocess(img, device, size=256):
        img = cv2.resize(img, (size, size))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = (
            torch.from_numpy(img_rgb)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float() / 255.0
        )
        tensor = (tensor * 2) - 1
        return tensor.to(device)

    def get_lpips(self):
        t1 = self.lpips_preprocess(self.image1, self.device)
        t2 = self.lpips_preprocess(self.image2, self.device)
        with torch.no_grad():
            return self.lpips_model(t1, t2).item()

    def get_difference(self, alpha=0.5, beta=0.3, gamma=0.2):
        lp = self.lpips if self.lpips is not None else 0.0
        return (
            alpha * self.mse +
            beta * self.inv_ssim +
            gamma * lp
        )
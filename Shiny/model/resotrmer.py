import torch
from PIL import Image
import torchvision.transforms as T
from torchvision import transforms
from torchvision.transforms.functional import to_tensor, to_pil_image
import sys
import os

sys.path.append(os.path.abspath('Restormer'))
from Restormer.basicsr.models.archs.restormer_arch import Restormer

model_names = {"blind": "Restormer_models/gaussian_color_denoising_blind.pth",
               "sigma15": "Restormer_models/gaussian_color_denoising_sigma15.pth",
               "sigma25": "Restormer_models/gaussian_color_denoising_sigma25.pth",
               "sigma50": "Restormer_models/gaussian_color_denoising_sigma50.pth",
               "real": "Restormer_models/real_denoising.pth"}

class Restormer_Denoise():
    def __init__(self, model_name="blind"):
        self.model = Restormer()
        self.model.load_state_dict(torch.load(f'{model_names[model_name]}')['params'], strict=False)
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def load_image(self, image_path)->Image.Image:
        image = Image.open(image_path).convert("RGB")
        return image

    def denoise_image(self, img: Image.Image)->Image.Image:
        transform = T.Compose([
            transforms.Resize((248, 248)),
            T.ToTensor()
        ])
        img_tensor = transform(img).unsqueeze(0).to(self.device)  # Add batch dimension

        with torch.no_grad():
            restored = self.model(img_tensor)
            restored = torch.clamp(restored, 0, 1).squeeze(0).cpu()
            return to_pil_image(restored)



def main():
    restormer = Restormer_Denoise()
    img = restormer.denoise_image(restormer.load_image("noisy.png"))
    img.save('restormer.png')


if __name__ == "__main__":
    main()
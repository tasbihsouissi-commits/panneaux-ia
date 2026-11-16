"""MobileNetV2 préentraîné (ImageNet) pour classer un panneau recadré (43 classes)."""
import torch, torch.nn as nn
from PIL import Image
from torchvision import models, transforms

SIZE = 64
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
eval_tf = transforms.Compose([transforms.Resize((SIZE, SIZE)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def build(num_classes=43, pretrained=True):
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
    m.classifier[1] = nn.Linear(m.last_channel, num_classes)  # nouvelle tête
    return m


class SignClassifier:
    def __init__(self, weights, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build(pretrained=False)
        self.model.load_state_dict(torch.load(weights, map_location=self.device))
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, bgr_crop):
        rgb = bgr_crop[:, :, ::-1].copy()
        x = eval_tf(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        p = self.model(x).softmax(1)[0]
        c = int(p.argmax())
        return c, float(p[c])

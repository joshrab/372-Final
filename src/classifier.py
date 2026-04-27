import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def get_class_names(data_dir):
    return sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

def load_classifier(weights_path, num_classes, device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return model.to(device).eval()

def classify_image(image_path, model, class_names, device):
    img = TRANSFORM(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        idx = model(img).argmax(1).item()
    return class_names[idx].replace("_", " ")
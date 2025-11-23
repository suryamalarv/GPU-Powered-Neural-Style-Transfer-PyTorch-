import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, utils
from PIL import Image
import os

# ✅ Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)

# ✅ Ensure output folder exists
os.makedirs("output", exist_ok=True)

# ✅ Image loader
def load_image(path, max_size=512):
    image = Image.open(path).convert("RGB")
    size = min(max(image.size), max_size)
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)
    return image.to(device)

content = load_image("input/content.jpg")
style = load_image("input/style.jpg")

# ✅ Normalize for VGG
normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
normalization = transforms.Normalize(mean=normalization_mean, std=normalization_std)

def normalize_batch(batch):
    return normalization(batch.squeeze(0)).unsqueeze(0)

# ✅ Model and layers
vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
for param in vgg.parameters():
    param.requires_grad_(False)

layers = {
    '0': 'conv1_1',
    '5': 'conv2_1',
    '10': 'conv3_1',
    '19': 'conv4_1',
    '28': 'conv5_1'
}

def get_features(image, model):
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    return torch.mm(tensor, tensor.t())

# ✅ Extract features
content_features = get_features(normalize_batch(content), vgg)
style_features = get_features(normalize_batch(style), vgg)
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# ✅ Hyperparameters
content_weight = 1e0
style_weight = 1e6
num_steps = 400

# ✅ Target initialization
target = content.clone().requires_grad_(True)

optimizer = optim.Adam([target], lr=0.01)

print("Training started...")

for step in range(1, num_steps + 1):
    target_features = get_features(normalize_batch(target), vgg)

    # Content loss
    content_loss = torch.mean((target_features['conv4_1'] - content_features['conv4_1'])**2)

    # Style loss
    style_loss = 0
    for layer in style_grams:
        target_gram = gram_matrix(target_features[layer])
        style_gram = style_grams[layer]
        layer_loss = torch.mean((target_gram - style_gram)**2)
        style_loss += layer_loss / (style_gram.numel())

    total_loss = content_weight * content_loss + style_weight * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step [{step}/{num_steps}] | Total loss: {total_loss.item():.4f}")
        utils.save_image(target, f"output/step_{step}.jpg")

# ✅ Final save
utils.save_image(target, "output/final_stylized.jpg")
print("Stylized image saved to output/final_stylized.jpg")

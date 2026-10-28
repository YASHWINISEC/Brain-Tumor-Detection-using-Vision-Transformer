import torch
import cv2
import numpy as np
import os
from cnn_model import SimpleCNN

CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
MODEL_PATH = "models/best_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN(num_classes=4)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

feature_maps = None
gradients = None


def forward_hook(module, input, output):
    global feature_maps
    feature_maps = output


def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]


model.conv3.register_forward_hook(forward_hook)
model.conv3.register_full_backward_hook(backward_hook)


# ---------- SIMPLE BRAIN MASK ----------
def get_brain_mask(gray):
    _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    mask = mask / 255.0
    return mask


def predict_image(image_path):
    global feature_maps, gradients

    original = cv2.imread(image_path)
    h, w, _ = original.shape

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    brain_mask = get_brain_mask(gray)

    # ---------- PREPROCESS ----------
    img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    img = np.transpose(img, (2, 0, 1))
    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    img_tensor.requires_grad_(True)

    # ---------- FORWARD ----------
    outputs = model(img_tensor)
    probs = torch.softmax(outputs, dim=1)[0]

    class_idx = torch.argmax(probs).item()
    class_name = CLASS_NAMES[class_idx]
    confidence = probs[class_idx].item() * 100

    # ---------- BACKWARD ----------
    model.zero_grad()
    outputs[0, class_idx].backward()

    # ---------- GRAD-CAM ----------
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * feature_maps, dim=1).squeeze()

    cam = cam.detach().cpu().numpy()
    cam = np.maximum(cam, 0)
    cam /= (cam.max() + 1e-8)
    cam = cv2.resize(cam, (w, h))

    # ---------- APPLY BRAIN MASK (KEY FIX) ----------
    cam = cam * brain_mask

    cam = cv2.GaussianBlur(cam, (21, 21), 0)

    # ---------- HEATMAP ----------
    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam), cv2.COLORMAP_JET
    )

    alpha = cam[..., None]
    alpha = np.clip(alpha, 0, 1)

    final = (original * (1 - alpha) + heatmap * alpha).astype(np.uint8)

    # ---------- SAVE ----------
    cam_path = image_path.replace("uploads", "uploads/gradcam")
    os.makedirs(os.path.dirname(cam_path), exist_ok=True)
    cv2.imwrite(cam_path, final)

    gradcam_web_path = os.path.relpath(cam_path, "static").replace("\\", "/")

    return class_name, confidence, gradcam_web_path

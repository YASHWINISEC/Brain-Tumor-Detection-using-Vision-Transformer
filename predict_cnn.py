# predict_cnn.py
import os
import json
import argparse
from PIL import Image
import torch
import torchvision.transforms as transforms

DEFAULT_CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

def _clean_state_dict(sd: dict):
    # If keys are wrapped in 'module.' (DataParallel) remove prefix
    if any(k.startswith('module.') for k in sd.keys()):
        return {k.replace('module.', '', 1): v for k, v in sd.items()}
    return sd

def load_model_from_path(model_path: str, device: torch.device, num_classes: int = 4):
    """
    Tries to import user's cnn_model.CNNModel. If not present, uses torchvision.resnet18.
    Loads the state dict robustly.
    """
    # Try to use a user-provided architecture first
    model = None
    try:
        # attempt to import their CNN definition
        from cnn_model import CNNModel  # noqa: E402
        model = CNNModel()
        print("Using cnn_model.CNNModel() architecture.")
    except Exception:
        # fallback to resnet18 if cnn_model isn't available/matches
        try:
            import torchvision.models as models  # noqa: E402
            model = models.resnet18(pretrained=False)
            in_features = model.fc.in_features
            model.fc = torch.nn.Linear(in_features, num_classes)
            print("Falling back to torchvision.models.resnet18 with adjusted fc.")
        except Exception as e:
            raise RuntimeError("No model architecture available (cnn_model or torchvision).") from e

    # load state dict
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    state = torch.load(model_path, map_location=device)

    # state might be dict with nested keys like 'model_state_dict'
    if isinstance(state, dict):
        # common saved checkpoint formats
        if 'model_state_dict' in state:
            raw_sd = state['model_state_dict']
        elif 'state_dict' in state:
            raw_sd = state['state_dict']
        else:
            raw_sd = state
    else:
        raise RuntimeError("Unexpected model file format.")

    raw_sd = _clean_state_dict(raw_sd)
    model.load_state_dict(raw_sd)
    model.to(device)
    model.eval()
    return model

def predict_image(image_path: str, model_path: str = None, device: str = None, classes: list = None):
    """
    Returns: dict { label: str, confidence: float (0-1), probs: {class: prob, ...} }
    """
    if classes is None:
        classes = DEFAULT_CLASSES
    num_classes = len(classes)

    device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # locate a model if not provided: check models/best_resnet18_model.pth, models/best_model.pth, or any .pth
    if model_path is None:
        default_dirs = ['models', './models']
        found = None
        for d in default_dirs:
            if os.path.isdir(d):
                # prefer best_resnet18_model.pth then best_model.pth
                for prefer in ['best_resnet18_model.pth', 'best_model.pth']:
                    p = os.path.join(d, prefer)
                    if os.path.exists(p):
                        found = p
                        break
                if found:
                    break
                # otherwise pick the first .pth file
                for fname in os.listdir(d):
                    if fname.endswith('.pth'):
                        found = os.path.join(d, fname)
                        break
            if found:
                break
        if found is None:
            raise FileNotFoundError("No .pth model found in models/ and no model_path provided.")
        model_path = found

    model = load_model_from_path(model_path, device, num_classes=num_classes)

    # transforms (standard ImageNet-like)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        if outputs.shape[1] == 1:
            # handle unexpected single-output models by applying sigmoid
            probs = torch.sigmoid(outputs).cpu().numpy().ravel().tolist()
            # cannot map easily to classes; return raw
            result = {
                'label': None,
                'confidence': float(probs[0]),
                'probs': probs
            }
            return result
        else:
            # multi-class
            import torch.nn.functional as F
            p = F.softmax(outputs, dim=1).cpu().numpy().ravel()
            top_idx = int(p.argmax())
            label = classes[top_idx] if top_idx < len(classes) else str(top_idx)
            probs_dict = {classes[i] if i < len(classes) else str(i): float(p[i]) for i in range(len(p))}
            return {'label': label, 'confidence': float(p[top_idx]), 'probs': probs_dict, 'model_path': model_path}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict an image using the CNN model.")
    parser.add_argument('--image', '-i', required=True, help="Path to input image")
    parser.add_argument('--model', '-m', default=None, help="Path to model .pth file (optional)")
    parser.add_argument('--device', '-d', default=None, help="Device: cpu or cuda")
    args = parser.parse_args()

    res = predict_image(args.image, model_path=args.model, device=args.device)
    print(json.dumps(res, indent=2))

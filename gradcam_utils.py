import torch
import cv2
import numpy as np

def generate_gradcam(model, img_tensor, original_img):
    model.eval()

    gradients = []
    activations = []

    # ✅ Correct layer for SimpleCNN
    target_layer = model.conv3   # ← IMPORTANT (matches SimpleCNN)

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    output = model(img_tensor)
    class_idx = output.argmax().item()

    model.zero_grad()
    output[0, class_idx].backward()

    grads = gradients[0].mean(dim=[2, 3])[0]
    acts = activations[0][0]

    cam = torch.zeros(acts.shape[1:], device=acts.device)
    for i, w in enumerate(grads):
        cam += w * acts[i]

    cam = torch.relu(cam)
    cam = cam / cam.max()
    cam = cam.cpu().numpy()

    cam = cv2.resize(cam, original_img.size)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    img_np = np.array(original_img)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    return overlay

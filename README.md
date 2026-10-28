# Brain Tumor Detection Using Deep Learning

A web-based brain MRI image classification project that uses a Convolutional Neural Network (CNN) built with PyTorch to classify MRI images into four categories: **glioma**, **meningioma**, **no tumor**, and **pituitary**. The Flask interface lets users upload an image and view the predicted class and confidence score. The project also includes Grad-CAM visualizations to help show image regions that influenced a prediction.

> **Important medical disclaimer:** This project is for educational and research purposes only. It is not a medical device and must not be used to diagnose, treat, or rule out a brain tumor. Always consult a qualified medical professional.

## Features

- Upload a brain MRI image through a Flask web interface.
- Classify images into four supported categories.
- Display the model's predicted class and confidence score.
- Generate a Grad-CAM heatmap to visualize influential image regions.
- Train and evaluate CNN models using a labeled image dataset.
- Save training metrics and evaluation outputs for analysis.

## Technology Stack

- **Python**
- **PyTorch** and **Torchvision** — deep learning and image transformations
- **Flask** — web application
- **OpenCV** and **NumPy** — image preprocessing
- **Matplotlib / Seaborn / Pandas** — analysis and visualization

## Project Structure

The archive contains a main project directory and a `Backups/` directory with runnable-looking application and model scripts. The exact layout can vary depending on which version you use.

```text
BrainTumor_x/
├── Backups/
│   ├── app.py                 # Flask application
│   ├── cnn_model.py            # CNN model definition
│   ├── train.py                # Model training
│   ├── predict.py              # Image prediction and Grad-CAM
│   ├── evaluate.py             # Model evaluation
│   ├── requirements.txt        # Python dependencies
│   ├── models/                 # Saved model weights
│   ├── dataset/                # Training/testing/validation images
│   └── static/                 # Uploaded images and generated outputs
├── dataset/                    # Dataset files (if included in selected version)
├── metrics/                    # Training metrics (if included)
└── README.md
```

## Dataset

The model code is configured for four image classes:

1. `glioma`
2. `meningioma`
3. `no_tumor`
4. `pituitary`

Organize the dataset according to the folder structure expected by the training script. For `torchvision.datasets.ImageFolder`, each split should contain one subfolder per class, for example:

```text
dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── no_tumor/
│   └── pituitary/
├── Testing/
│   ├── glioma/
│   ├── meningioma/
│   ├── no_tumor/
│   └── pituitary/
└── validation/
    ├── glioma/
    ├── meningioma/
    ├── no_tumor/
    └── pituitary/
```

Use only data you have permission to use, and ensure the class-folder names match the model's class mapping.

## Setup

### 1. Clone or download the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_PROJECT_DIRECTORY>
```

If running the scripts inside `Backups/`, move into that directory:

```bash
cd Backups
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The repository's requirements file may include packages for other experiments. If installation fails, install the dependencies needed by the selected script, and check that your installed PyTorch and Torchvision versions are compatible.

### 4. Check the model weights

The prediction script expects a trained checkpoint at:

```text
models/best_model.pth
```

Make sure the checkpoint exists relative to the working directory and matches the architecture defined in `cnn_model.py`. If the model weights are not included in the repository, train the model first or obtain the checkpoint from your project owner.

## Run the Web Application

From the directory containing `app.py`:

```bash
python app.py
```

Open the local address printed by Flask in your terminal (commonly `http://127.0.0.1:5000/`) in a browser. Upload an MRI image to view the prediction and Grad-CAM result.

**Development note:** The sample Flask code runs with `debug=True`. Use debug mode only for local development, never for a public production deployment.

## Train the Model

Review the paths and hyperparameters in `train.py` first, then run:

```bash
python train.py
```

Training can take significant time and memory. A compatible GPU can speed up training, but a CPU may work for smaller experiments. Confirm where the script saves the best checkpoint and update the prediction script's `MODEL_PATH` if needed.

## How It Works

1. The user uploads an MRI image.
2. The application reads and preprocesses the image.
3. The trained CNN produces scores for the four classes.
4. Softmax converts the scores into class probabilities.
5. The class with the highest probability is shown with its confidence score.
6. Grad-CAM uses gradients and feature maps from a convolutional layer to create a heatmap highlighting regions that influenced the model output.

Grad-CAM highlights model attention; it does not prove that a highlighted region contains a tumor or explain a prediction clinically.

## Troubleshooting

- **Checkpoint not found:** Confirm `models/best_model.pth` exists relative to the directory from which you run the app.
- **State-dictionary mismatch:** Use weights trained with the same model architecture and layer names as the current code.
- **Missing package:** Activate the virtual environment and install the required dependencies.
- **Dataset folders not found:** Check the paths configured in the training script.
- **No Grad-CAM output:** Confirm the selected model layer registers forward and backward hooks and that gradients are computed during prediction.

## Future Improvements

- Evaluate on a separate, representative test set and report per-class precision, recall, F1-score, and a confusion matrix.
- Add input validation, upload size limits, and safer temporary-file handling.
- Improve explainability visualizations and test their stability.
- Add automated tests and reproducible environment specifications.
- Deploy behind a production WSGI server with appropriate privacy and security controls.

## License

Add a license file before publishing this repository. If the dataset or research materials have separate usage terms, follow those terms as well.

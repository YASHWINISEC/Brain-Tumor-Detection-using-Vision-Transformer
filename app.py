import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from predict import predict_image

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_path = None
    gradcam_image = None

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html")

        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        prediction, confidence, gradcam_image = predict_image(save_path)
        image_path = f"uploads/{filename}"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image=image_path,
        gradcam=gradcam_image
    )


if __name__ == "__main__":
    app.run(debug=True)

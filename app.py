from flask import Flask, render_template, request, send_from_directory
import os
from datetime import datetime
from ultralytics import YOLO

app = Flask(__name__)

RESULTS_FOLDER = os.path.join("static", "results")
os.makedirs(RESULTS_FOLDER, exist_ok=True)

model = YOLO("best.pt")  # Make sure best.pt is in your repo root


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file part", 400

    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    original_name = file.filename
    ext = os.path.splitext(original_name)[1]

    input_path = os.path.join(RESULTS_FOLDER, f"input_{timestamp}{ext}")
    output_path = os.path.join(RESULTS_FOLDER, f"output_{timestamp}{ext}")

    file.save(input_path)

    results = model.predict(
        source=input_path,
        save=True,
        project=RESULTS_FOLDER,
        name=f"run_{timestamp}",
        exist_ok=True,
        verbose=False
    )

    run_dir = os.path.join(RESULTS_FOLDER, f"run_{timestamp}")
    for fname in os.listdir(run_dir):
        processed_file = fname
        break

    os.rename(os.path.join(run_dir, processed_file), output_path)
    os.rmdir(run_dir)

    return send_from_directory(
        RESULTS_FOLDER,
        os.path.basename(output_path),
        as_attachment=False
    )


if __name__ == "__main__":
    app.run()

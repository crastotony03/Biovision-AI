from flask import Flask, render_template, request, send_from_directory
import os
from datetime import datetime
from ultralytics import YOLO

app = Flask(__name__)

# Folder to save outputs
RESULTS_FOLDER = os.path.join("static", "results")
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO("best.pt")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
    if "file" not in request.files:
        return "No file part", 400

    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    # Create a unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    original_name = file.filename
    ext = os.path.splitext(original_name)[1]
    input_path = os.path.join(RESULTS_FOLDER, f"input_{timestamp}{ext}")
    output_path = os.path.join(RESULTS_FOLDER, f"output_{timestamp}{ext}")

    # Save uploaded file
    file.save(input_path)

    # Run YOLO prediction
    # For images and videos, ultralytics handles both
    results = model.predict(
        source=input_path,
        save=True,
        project=RESULTS_FOLDER,
        name=f"run_{timestamp}",
        exist_ok=True,
        verbose=False
    )

    # ultralytics creates a folder like: static/results/run_{timestamp}/
    # and saves the processed file there. We need to find that file.
    run_dir = os.path.join(RESULTS_FOLDER, f"run_{timestamp}")
    # there should be exactly one result file in that directory
    for fname in os.listdir(run_dir):
        processed_file = fname
        break

    # Move/rename it to output_path for simpler serving
    os.rename(os.path.join(run_dir, processed_file), output_path)
    os.rmdir(run_dir)  # remove empty folder

    # Return the processed file to the user
    return send_from_directory(
        RESULTS_FOLDER,
        os.path.basename(output_path),
        as_attachment=False
    )


if __name__ == "__main__":
    app.run(debug=True)
import os
import uuid
 
from flask import Flask, render_template, request, redirect, url_for, flash
 
from process_video import process_video
 
app = Flask(__name__)
app.secret_key = "dev-only-change-this"  # only needed for flash messages
 
UPLOAD_DIR = os.path.join(app.static_folder, "uploads")
OUTPUT_DIR = os.path.join(app.static_folder, "outputs")
ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv"}
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100 MB -- keep demo clips short
 
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
 
 
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
 
 
@app.route("/")
def index():
    return render_template("index.html")
 
 
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("video")
 
    if file is None or file.filename == "":
        flash("Choose a video file first.")
        return redirect(url_for("index"))
 
    if not allowed_file(file.filename):
        flash("That file type isn't supported -- try mp4, mov, avi, or mkv.")
        return redirect(url_for("index"))
 
    job_id = uuid.uuid4().hex
    input_path = os.path.join(UPLOAD_DIR, f"{job_id}_input.mp4")
    raw_output_path = os.path.join(OUTPUT_DIR, f"{job_id}_raw.mp4")
    final_output_path = os.path.join(OUTPUT_DIR, f"{job_id}.mp4")
 
    file.save(input_path)
 
    try:
        stats = process_video(input_path, raw_output_path, final_output_path)
    except Exception as exc:
        flash(f"Processing failed: {exc}")
        return redirect(url_for("index"))
    finally:
        if os.path.exists(raw_output_path):
            os.remove(raw_output_path)
        if os.path.exists(input_path):
            os.remove(input_path)
 
    return render_template(
        "result.html",
        video_filename=f"{job_id}.mp4",
        stats=stats,
    )
 
 
if __name__ == "__main__":
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    app.run(debug=True)
 
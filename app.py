from flask import Flask, render_template, request, redirect
import base64
import io


ALLOWED_EXTENSIONS = {"wav", "mp3"}

app = Flask(__name__)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def handle_audio(file_stream, lang="en"):
    from faster_whisper import WhisperModel

    model_size = "tiny"
    model = WhisperModel(
        model_size, device="cpu", compute_type="int8", download_root="/tmp"
    )
    print("finish load")

    # data = file.read()
    print("start transcribe")
    segments, info = model.transcribe(file_stream, beam_size=5)
    res = []
    for segment in segments:
        res.append("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

        rest = "Detected language '%s' with probability %f" % (
            info.language,
            info.language_probability,
        )
        res.append(rest)
    return res


@app.route("/upload", methods=["POST", "GET"])
def upload_audio():
    if request.method == "GET":
        return render_template("index.html")

    if "file" not in request.files:
        return redirect(request.url)

    file = request.files["file"]

    if file.filename == "":
        return redirect(request.url)

    if file and not allowed_file(file.filename):
        # The image file seems valid! Detect faces and return the result.
        params = {"message": "File type not allowed"}
        return render_template("index.html", **params)

    # file_content = file.read()
    sentences = handle_audio(file)
    params = {"sentences": sentences}
    return render_template("index.html", **params)


@app.route("/")
def hello():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8888, debug=True)

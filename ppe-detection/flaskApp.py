from flask import Flask, render_template, Response, session, send_file
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField, IntegerRangeField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
from flask_bootstrap import Bootstrap
from flask_socketio import SocketIO, emit
import os
import secrets
import cv2
from hubconfCustom import video_detection

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", secrets.token_hex(24))
app.config["UPLOAD_FOLDER"] = "static/files"
Bootstrap(app)
socketio = SocketIO(app)


class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    conf_slide = IntegerRangeField(
        "Confidence:  ", default=25, validators=[InputRequired()]
    )
    submit = SubmitField("Run")


def send_socketio_data(detect_count, safe_count):
    socketio.emit(
        "update_data", {"detect_count": detect_count, "safe_count": safe_count}
    )


def generate_frames(path_x="", conf_=0.25):
    yolo_output = video_detection(path_x, conf_)
    for detection_, d_count, s_count in yolo_output:
        ref, buffer = cv2.imencode(".jpg", detection_)
        frame = buffer.tobytes()

        send_socketio_data(d_count, s_count)

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/", methods=["GET", "POST"])
def front():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        conf_ = form.conf_slide.data
        file_path = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            app.config["UPLOAD_FOLDER"],
            secure_filename(file.filename),
        )
        file.save(file_path)
        session["video_path"] = file_path
        session["conf_"] = conf_
    return render_template("video.html", form=form)


@app.route("/video")
def video():
    video_path = session.get("video_path", None)

    if video_path is None:
        return send_file(
            os.path.join(app.config["UPLOAD_FOLDER"], "Black.png"), mimetype="image/png"
        )

    conf_ = session.get("conf_", 25)
    return Response(
        generate_frames(path_x=video_path, conf_=round(float(conf_) / 100, 2)),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@socketio.on("connect")
def handle_connect():
    print("Client connected")


@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected")


if __name__ == "__main__":
    socketio.run(app, debug=True)

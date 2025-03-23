from flask import Flask, render_template, Response, session
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField, IntegerRangeField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
import os
import secrets
from flask_bootstrap import Bootstrap
import cv2
import asone
from asone import ASOne
from flask_socketio import SocketIO

from hubconfCustom import video_detection

app = Flask(__name__)
Bootstrap(app)
socketio = SocketIO(app)

app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", secrets.token_hex(24))
app.config["UPLOAD_FOLDER"] = "static/files"

dt_obj = ASOne(
    tracker=asone.DEEPSORT, detector=asone.YOLOV8N_PYTORCH, weights=None, use_cuda=True
)


class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    conf_slide = IntegerRangeField(
        "Confidence:  ", default=25, validators=[InputRequired()]
    )
    submit = SubmitField("Run")


def send_socketio_data(dpf):
    socketio.emit("update_data", {"detected_objects": dpf})


def generate_frames(path_x="", conf_=0.25, dt_obj=dt_obj):
    yolo_output = video_detection(path_x, conf_, dt_obj)
    for detection_, dpf in yolo_output:
        ref, buffer = cv2.imencode(".jpg", detection_)

        send_socketio_data(dpf)

        frame = buffer.tobytes()
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
    return Response(
        generate_frames(
            path_x=session.get("video_path", None),
            conf_=round(float(session.get("conf_", None)) / 100, 2),
        ),
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

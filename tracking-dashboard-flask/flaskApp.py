from flask import Flask, render_template, Response, jsonify, request, session, send_file
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField, IntegerRangeField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
from flask_bootstrap import Bootstrap
from flask_socketio import SocketIO
import os
import secrets
import cv2
from hubconfCustom import video_detection

app = Flask(__name__)
Bootstrap(app)
socketio = SocketIO(app)

app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", secrets.token_hex(24))
app.config["UPLOAD_FOLDER"] = "static/files"


class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    conf_slide = IntegerRangeField(
        "Confidence:  ", default=25, validators=[InputRequired()]
    )
    submit = SubmitField("Run")


def send_socketio_data(FPS_, size, dpf):
    socketio.emit(
        "update_data", {"fps": FPS_, "image_size": size[0], "detected_objects": dpf}
    )


def generate_frames(path_x="", conf_=0.25):
    yolo_output = video_detection(path_x, conf_)
    for detection_, FPS_, size, dpf in yolo_output:
        ref, buffer = cv2.imencode(".jpg", detection_)
        frame = buffer.tobytes()

        socketio.start_background_task(send_socketio_data, FPS_, size, dpf)

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/", methods=["GET", "POST"])
def front():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        conf_ = form.conf_slide.data
        try:
            file_path = os.path.join(
                os.path.abspath(os.path.dirname(__file__)),
                app.config["UPLOAD_FOLDER"],
                secure_filename(file.filename),
            )
            file.save(file_path)
            session["video_path"] = file_path
            session["conf_"] = conf_
        except Exception as e:
            return jsonify({"error": str(e)})
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

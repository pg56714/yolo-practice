from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
import io
from converter import convert_file

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        f = request.files["file"]
        field = request.form["field"]
        filename = secure_filename(f.filename)
        
        if not filename.lower().endswith(".pt"):
            return jsonify({"error": "Only .pt files are supported."}), 400
        
        file_bytes = io.BytesIO(f.read())
        new_filename, file_data = convert_file(filename, file_bytes, field)
        
        return send_file(file_data, download_name=new_filename, as_attachment=True)
    
    return render_template("index.html")

@app.route("/reset", methods=["GET"])
def reset():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

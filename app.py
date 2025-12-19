from flask import Flask, render_template, request, redirect, url_for
import os

from predict import predict_insect      # <-- NEW (real prediction)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

CLASS_MAP = {
    0: "Chorthippus biguttulus",
    1: "Gryllus bimaculatus",
    2: "Ruspolia nitidula",
    3: "Other Insects",
    4: "Environmental Noise"
}

INSECT_INFO = {
    "Chorthippus biguttulus": {
        "about": "A grasshopper species known for strong stridulation sounds. Generally not harmful but can feed on leaves.",
        "harmful": False,
        "control": "No major action needed. If population increases, use organic repellents like neem spray.",
        "image": "images/Chorthippus_biguttulus.png"
    },
    "Gryllus bimaculatus": {
        "about": "A loud field cricket commonly found near crops. Usually harmless.",
        "harmful": False,
        "control": "No treatment required unless population becomes a disturbance.",
        "image": "images/Gryllus_bimaculatus.png"
    },
    "Ruspolia nitidula": {
        "about": "Conehead grasshopper. Can feed on grains.",
        "harmful": True,
        "control": "Use light traps or pheromone traps.",
        "image": "images/Ruspolia_nitidula.png"
    },
    "Other Insects": {
        "about": "Insect sound does not match primary species.",
        "harmful": "Unknown",
        "control": "Try recording closer.",
        "image": "images/insect_silhouette.png"
    },
    "Env": {
        "about": "Likely background noise, not insects.",
        "harmful": False,
        "control": "Record again in a quiet environment.",
        "image": "images/env.png"
    }
}

# ------------------------------------------------------

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/test')
def test():
    return render_template('test.html')

@app.route('/research')
def research():
    return render_template('research.html')

# --------------- REAL PREDICTION ----------------------

@app.route('/predict_upload', methods=['POST'])
def predict_upload():
    file = request.files.get('audio')

    if not file:
        return redirect(url_for('test'))

    # Save uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # RUN REAL MODEL
    predicted_label = predict_insect(filepath)
    info = INSECT_INFO[predicted_label]

    insect_data = {
        "name": predicted_label,
        "image": url_for('static', filename=info["image"]),
        "about": info["about"],
        "control": info["control"],
        "harmful": info["harmful"]
    }

    return render_template("result.html", insect=insect_data)

# ------------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True)

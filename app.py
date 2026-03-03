import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing import image

# ===============================
# App Configuration
# ===============================

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(STATIC_FOLDER, "images"), exist_ok=True)

# ===============================
# Load Model
# ===============================

model = tf.keras.models.load_model("PlantCareAI_Model.keras")

# ===============================
# Class Names (All 38 Classes)
# ===============================

class_names = [
"Apple___Apple_scab",
"Apple___Black_rot",
"Apple___Cedar_apple_rust",
"Apple___healthy",
"Blueberry___healthy",
"Cherry_(including_sour)___Powdery_mildew",
"Cherry_(including_sour)___healthy",
"Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
"Corn_(maize)___Common_rust_",
"Corn_(maize)___Northern_Leaf_Blight",
"Corn_(maize)___healthy",
"Grape___Black_rot",
"Grape___Esca_(Black_Measles)",
"Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
"Grape___healthy",
"Orange___Haunglongbing_(Citrus_greening)",
"Peach___Bacterial_spot",
"Peach___healthy",
"Pepper,_bell___Bacterial_spot",
"Pepper,_bell___healthy",
"Potato___Early_blight",
"Potato___Late_blight",
"Potato___healthy",
"Raspberry___healthy",
"Soybean___healthy",
"Squash___Powdery_mildew",
"Strawberry___Leaf_scorch",
"Strawberry___healthy",
"Tomato___Bacterial_spot",
"Tomato___Early_blight",
"Tomato___Late_blight",
"Tomato___Leaf_Mold",
"Tomato___Septoria_leaf_spot",
"Tomato___Spider_mites Two-spotted_spider_mite",
"Tomato___Target_Spot",
"Tomato___Tomato_Yellow_Leaf_Curl_Virus",
"Tomato___Tomato_mosaic_virus",
"Tomato___healthy"
]
disease_info = {
    "Northern Leaf Blight": {
        "description": "A fungal disease causing long gray-green lesions on leaves.",
        "treatment": "Use resistant hybrids and apply fungicides if necessary."
    },
    "Early blight": {
        "description": "Fungal disease causing dark spots with concentric rings.",
        "treatment": "Remove infected leaves and apply fungicide."
    },
    "Late blight": {
        "description": "Serious fungal disease causing water-soaked spots.",
        "treatment": "Use certified seeds and spray recommended fungicides."
    }
}
# ===============================
# Prediction Function
# ===============================
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0])) * 100

    if predicted_index < len(class_names):
        return class_names[predicted_index], round(confidence, 2)
    else:
        return "Unknown", 0
# ===============================
# Routes
# ===============================

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded")

        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html", error="No selected file")

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        predicted_class, confidence = predict_image(filepath)

        if predicted_class != "Unknown":
            plant, disease = predicted_class.split("___")
            plant = plant.replace("_", " ")
            disease = disease.replace("_", " ")

            status = "Healthy" if "healthy" in disease.lower() else "Diseased"
            color = "green" if status == "Healthy" else "red"

            info = disease_info.get(disease, {
                "description": "No description available.",
                "treatment": "No treatment information available."
            })

        else:
            plant = "Unknown"
            disease = "Unknown"
            confidence = 0
            status = "Unknown"
            color = "black"
            info = {"description": "", "treatment": ""}

        if os.path.exists(filepath):
            os.remove(filepath)

        return render_template("result.html",
                               plant=plant,
                               disease=disease,
                               confidence=confidence,
                               status=status,
                               color=color,
                               description=info["description"],
                               treatment=info["treatment"],
                               image_path="uploads/" + file.filename)

    except Exception as e:
        return {"error": str(e)}
# ===============================
# Run App
# ===============================

if __name__ == "__main__":
    app.run(debug=True)
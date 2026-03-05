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

# ===================== APPLE =====================
"Apple scab": {
    "description": "Fungal disease causing dark olive-green spots on leaves and fruit.",
    "treatment": "Apply fungicides early in season and remove infected leaves."
},
"Black rot": {
    "description": "Fungal infection causing black lesions on fruit and leaves.",
    "treatment": "Prune infected branches and apply appropriate fungicide."
},
"Cedar apple rust": {
    "description": "Fungal disease causing yellow-orange spots on leaves.",
    "treatment": "Remove nearby cedar trees and apply fungicide."
},

# ===================== CHERRY =====================
"Powdery mildew": {
    "description": "White powder-like fungal growth on leaves.",
    "treatment": "Improve air circulation and apply sulfur-based fungicide."
},

# ===================== CORN =====================
"Cercospora leaf spot Gray leaf spot": {
    "description": "Gray rectangular lesions on leaves caused by fungus.",
    "treatment": "Use resistant varieties and apply foliar fungicides."
},
"Common rust": {
    "description": "Reddish-brown pustules on leaves.",
    "treatment": "Plant resistant hybrids and apply fungicide if severe."
},
"Northern Leaf Blight": {
    "description": "Long gray-green lesions on corn leaves caused by fungus.",
    "treatment": "Use resistant hybrids and apply fungicides when needed."
},

# ===================== GRAPE =====================
"Grape Black rot": {
    "description": "Dark brown lesions on leaves and shriveled black fruit.",
    "treatment": "Remove infected fruit and apply fungicide."
},
"Esca (Black Measles)": {
    "description": "Causes leaf discoloration and fruit spotting.",
    "treatment": "Prune infected wood and avoid plant stress."
},
"Leaf blight (Isariopsis Leaf Spot)": {
    "description": "Small brown spots on grape leaves.",
    "treatment": "Apply fungicide and ensure good drainage."
},

# ===================== ORANGE =====================
"Haunglongbing (Citrus greening)": {
    "description": "Serious bacterial disease causing yellow shoots and bitter fruit.",
    "treatment": "Remove infected trees and control psyllid insects."
},

# ===================== PEACH =====================
"Bacterial spot": {
    "description": "Dark water-soaked spots on leaves and fruit.",
    "treatment": "Use copper-based sprays and resistant varieties."
},

# ===================== POTATO =====================
"Early blight": {
    "description": "Dark concentric ring spots on leaves.",
    "treatment": "Remove infected leaves and apply fungicide."
},
"Late blight": {
    "description": "Water-soaked lesions that rapidly spread.",
    "treatment": "Use certified seeds and apply recommended fungicides."
},

# ===================== SQUASH =====================
"Powdery mildew Squash": {
    "description": "White powdery fungal growth on squash leaves.",
    "treatment": "Apply sulfur fungicide and ensure proper spacing."
},

# ===================== STRAWBERRY =====================
"Leaf scorch": {
    "description": "Purple or red spots that enlarge and dry leaves.",
    "treatment": "Remove infected leaves and apply fungicide."
},

# ===================== TOMATO =====================
"Tomato Bacterial spot": {
    "description": "Small dark spots on tomato leaves and fruit.",
    "treatment": "Use copper sprays and disease-free seeds."
},
"Tomato Early blight": {
    "description": "Brown spots with concentric rings on lower leaves.",
    "treatment": "Apply fungicide and rotate crops."
},
"Tomato Late blight": {
    "description": "Large dark lesions on leaves and fruit.",
    "treatment": "Use resistant varieties and fungicides."
},
"Tomato Leaf Mold": {
    "description": "Yellow spots on upper leaf surface with mold underneath.",
    "treatment": "Improve ventilation and apply fungicide."
},
"Tomato Septoria leaf spot": {
    "description": "Small circular spots with gray centers.",
    "treatment": "Remove infected leaves and apply fungicide."
},
"Tomato Spider mites Two-spotted spider mite": {
    "description": "Tiny pests causing yellow speckles on leaves.",
    "treatment": "Use insecticidal soap or neem oil."
},
"Tomato Target Spot": {
    "description": "Brown circular spots with concentric rings.",
    "treatment": "Apply fungicide and avoid overhead watering."
},
"Tomato Yellow Leaf Curl Virus": {
    "description": "Leaves curl upward and turn yellow.",
    "treatment": "Control whiteflies and remove infected plants."
},
"Tomato mosaic virus": {
    "description": "Mottled light and dark green patterns on leaves.",
    "treatment": "Remove infected plants and disinfect tools."
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

        # Save inside static/uploads
        upload_path = os.path.join("static", "uploads")
        os.makedirs(upload_path, exist_ok=True)

        filepath = os.path.join(upload_path, file.filename)
        file.save(filepath)

        predicted_class, confidence = predict_image(filepath)

        if predicted_class != "Unknown":
            plant, disease = predicted_class.split("___")
            plant = plant.replace("_", " ")
            disease = disease.replace("_", " ")

            status = "Healthy" if "healthy" in disease.lower() else "Diseased"

            info = disease_info.get(disease, {
                "description": "No description available.",
                "treatment": "No treatment information available."
            })
        else:
            plant = "Unknown"
            disease = "Unknown"
            confidence = 0
            status = "Unknown"
            info = {"description": "", "treatment": ""}

        return render_template("result.html",
                               plant=plant,
                               disease=disease,
                               confidence=confidence,
                               status=status,
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
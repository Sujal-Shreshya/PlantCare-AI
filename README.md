# 🌿 PlantCareAI – Plant Disease Detection using Deep Learning

PlantCareAI is an **AI-powered web application** that detects plant diseases from leaf images using **Deep Learning and Computer Vision**. The system helps farmers, gardeners, and agricultural researchers quickly identify plant diseases and receive treatment recommendations.

---

# 📌 Project Overview

Plant diseases are a major cause of crop loss worldwide. Early detection is crucial for preventing large-scale damage.

PlantCareAI solves this problem by allowing users to **upload a plant leaf image**, which is analyzed by a **trained Convolutional Neural Network (CNN)** model to detect diseases and suggest treatments.

---

# 🚀 Features

✅ Upload plant leaf images for disease detection
✅ Detect **38 plant diseases** from the PlantVillage dataset
✅ Display **plant name and disease type**
✅ Show **prediction confidence**
✅ Provide **disease description**
✅ Suggest **treatment methods**
✅ Simple and user-friendly web interface

---

# 🧠 Technologies Used

### Backend

* Python
* Flask

### AI / Machine Learning

* TensorFlow
* Keras
* Convolutional Neural Networks (CNN)

### Frontend

* HTML
* CSS
* JavaScript

### Dataset

* PlantVillage Dataset (38 plant disease classes)

Dataset Source:
[https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

---

# 📂 Project Structure

```
PlantCareAI
│
├── app.py
├── model.h5
├── requirements.txt
│
├── templates
│   ├── index.html
│   └── result.html
│
├── static
│   └── uploads
│
└── README.md
```

---

# ⚙️ How It Works

1️⃣ User uploads or captures a plant leaf image
2️⃣ Image is processed by the **trained CNN model**
3️⃣ The model predicts the **plant disease class**
4️⃣ System displays:

* Plant name
* Disease name
* Confidence score
* Disease description
* Treatment suggestions

---

# ▶️ How to Run the Project

### 1️⃣ Clone the repository

```
git clone https://github.com/your-username/PlantCareAI.git
```

### 2️⃣ Navigate to project folder

```
cd PlantCareAI
```

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 4️⃣ Run the application

```
python app.py
```

### 5️⃣ Open in browser

```
http://127.0.0.1:5000
```

---

# 📊 Model Information

* Model Type: **Convolutional Neural Network (CNN)**
* Dataset: **PlantVillage Dataset**
* Number of Classes: **38 plant diseases**
* Framework: **TensorFlow / Keras**

---

# 🎯 Benefits

🌱 Early detection of plant diseases
🌾 Helps farmers reduce crop loss
⚡ Fast and automated diagnosis
📱 Easy-to-use web interface

---

# 🔮 Future Improvements

* Mobile application for farmers
* Real-time camera detection
* Larger plant disease dataset
* Region-specific treatment recommendations
* Integration with agricultural advisory systems

---

# 👨‍💻 Author

Tanya MAheshwari
AI / Machine Learning Project

---

# 📜 License

This project is for expirimental Learning


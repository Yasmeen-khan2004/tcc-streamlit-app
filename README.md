# tcc-streamlit-app
# 🌩️ Tropical Cloud Cluster (TCC) Identification and Prediction

This project is built for the **ISRO Hackathon** 🛰️  
It uses satellite `.h5` data to:
- Detect tropical cloud clusters (TCCs)
- Extract statistical features (e.g., brightness temperature, cluster size)
- Predict whether a cluster may become a tropical cyclone 🌪️

🚀 The project includes:
✅ A **Streamlit app** for easy upload & prediction  
✅ Visualizations of detected clusters  
✅ Sample `.h5` test data

---

## 📦 **How it works**
1. Upload an HDF5 satellite file (e.g., from INSAT)
2. The app:
   - Extracts brightness temperature data
   - Detects clusters
   - Calculates features
   - Runs a trained machine learning model
3. Shows prediction & visualization

---

## 🛠 **Tech stack**
- Python, Streamlit
- NumPy, pandas, h5py, matplotlib, scikit-learn

---

## 🧩 **Folder structure**
streamlit_app/
├── app.py ← Streamlit app
├── tcc_model.pkl ← Trained ML model
├── sample_data/ ← Sample .h5 files for testing
├── requirements.txt
└── README.md

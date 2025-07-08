import streamlit as st
import h5py
import numpy as np
import pandas as pd
from scipy import ndimage
import joblib

st.title('🌩 Tropical Cloud Cluster Detection and Prediction')

uploaded_file = st.file_uploader("📂 Upload HDF5 Satellite File", type="h5")

if uploaded_file is not None:
    # Load trained model
    model = joblib.load('tcc_model.pkl')
    
    # Load data
    with h5py.File(uploaded_file, 'r') as f:
        st.write('Available datasets:', list(f.keys()))
        # Adjust dataset name below to your actual key
        bt_data = np.array(f['TIR1_BT'])[0]

    # Detect clusters
    threshold = 230
    clusters, num_clusters = ndimage.label(bt_data < threshold)
    st.write(f'Detected clusters: {num_clusters}')

    # Extract features
    features = []
    for cluster_idx in range(1, num_clusters+1):
        coords = np.argwhere(clusters == cluster_idx)
        if coords.size == 0:
            continue
        tb_values = bt_data[clusters == cluster_idx]
        min_tb_idx = np.argmin(tb_values)
        center = coords[min_tb_idx]
        lat, lon = center[0], center[1]
        pixel_count = len(tb_values)
        mean_tb = np.mean(tb_values)
        min_tb = np.min(tb_values)
        median_tb = np.median(tb_values)
        std_tb = np.std(tb_values)
        center_point = np.array(center)
        distances = np.linalg.norm(coords - center_point, axis=1)
        max_radius = np.max(distances)
        min_radius = np.min(distances)
        mean_radius = np.mean(distances)
        max_height = (250 - min_tb) * 0.1
        mean_height = (250 - mean_tb) * 0.1
        features.append({
            'lat': lat, 'lon': lon, 'pixel_count': pixel_count, 'mean_tb': mean_tb, 'min_tb': min_tb,
            'median_tb': median_tb, 'std_tb': std_tb, 'max_radius': max_radius, 'min_radius': min_radius,
            'mean_radius': mean_radius, 'max_cloud_top_height': max_height, 'mean_cloud_top_height': mean_height
        })

    if features:
        features_df = pd.DataFrame(features)

        # Predict
        X = features_df[['pixel_count', 'mean_tb', 'min_tb', 'median_tb', 'std_tb',
                         'max_radius', 'min_radius', 'mean_radius',
                         'max_cloud_top_height', 'mean_cloud_top_height']]
        predictions = model.predict(X)
        features_df['predicted_target'] = predictions

        st.subheader('✅ Predicted Clusters:')
        st.dataframe(features_df)

        # Optionally download results
        csv = features_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results as CSV", csv, "tcc_predictions.csv", "text/csv")
    else:
        st.warning("No clusters detected with the given threshold.")

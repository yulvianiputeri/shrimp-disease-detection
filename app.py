import streamlit as st
import cv2
import numpy as np
import os
import joblib

# Import page functions
from pages import render_prediksi, render_galeri, render_pipeline, render_evaluasi

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Deteksi Penyakit Udang", 
    page_icon="🦐", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS STYLING
# ============================================================
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
* { font-family: 'Poppins', sans-serif !important; }
.stApp { background-color: #FAFBFC; }
.card { background: white; border-radius: 16px; padding: 24px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); margin-bottom: 20px; border: 1px solid #E8ECF0; }
.card-header { font-size: 18px; font-weight: 600; color: #1B4D3E; margin-bottom: 16px; }
.header-card { background: linear-gradient(135deg, #1B4D3E 0%, #2D6A5A 100%); border-radius: 20px; padding: 40px; text-align: center; color: white; margin-bottom: 30px; box-shadow: 0 8px 30px rgba(27,77,62,0.3); }
.header-title { font-size: 36px; font-weight: 700; margin-bottom: 10px; }
.header-subtitle { font-size: 18px; font-weight: 300; opacity: 0.9; }
.header-divider { height: 3px; width: 100px; background: linear-gradient(90deg, #FFD700, #FF8C00); margin: 20px auto 0; border-radius: 2px; }
.pred-card { background: linear-gradient(145deg, #FFFFFF 0%, #F8F9FA 100%); border-radius: 16px; padding: 24px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); border: 1px solid #E8ECF0; text-align: center; transition: all 0.3s ease; }
.pred-card:hover { transform: translateY(-4px); box-shadow: 0 8px 25px rgba(0,0,0,0.12); }
.pred-card-healthy { border-top: 4px solid #28A745; background: linear-gradient(145deg, #FFFFFF 0%, #E8F5E9 100%); }
.pred-card-diseased { border-top: 4px solid #DC3545; background: linear-gradient(145deg, #FFFFFF 0%, #FFE8E8 100%); }
.pred-header { font-size: 16px; font-weight: 600; color: #495057; margin-bottom: 16px; }
.pred-percentage { font-size: 42px; font-weight: 700; margin: 16px 0; }
.pred-percentage-healthy { color: #28A745; }
.pred-percentage-diseased { color: #DC3545; }
.pred-badge { display: inline-block; padding: 8px 20px; border-radius: 50px; font-weight: 600; font-size: 14px; margin-top: 12px; }
.badge-healthy { background: linear-gradient(135deg, #28A745, #34CE57); color: white; }
.badge-diseased { background: linear-gradient(135deg, #DC3545, #FF6B6B); color: white; }
.progress-container { background: #E9ECEF; border-radius: 10px; height: 24px; overflow: hidden; margin: 16px 0; }
.progress-bar { height: 100%; border-radius: 10px; display: flex; align-items: center; justify-content: flex-end; padding-right: 12px; font-size: 12px; font-weight: 600; color: white; }
.progress-bar-healthy { background: linear-gradient(90deg, #28A745, #34CE57); }
.progress-bar-diseased { background: linear-gradient(90deg, #DC3545, #FF6B6B); }
.heatmap-container { background: white; border-radius: 16px; padding: 24px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); margin-bottom: 20px; }
.heatmap-description { background: #E3F2FD; border-radius: 12px; padding: 16px 20px; margin-bottom: 20px; color: #1565C0; font-size: 14px; border-left: 4px solid #1976D2; }
.disease-card { background: white; border-radius: 16px; padding: 28px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); margin-bottom: 20px; }
.disease-header { display: flex; align-items: center; gap: 12px; margin-bottom: 16px; }
.disease-title { font-size: 22px; font-weight: 600; color: #1B4D3E; }
.disease-text { color: #495057; line-height: 1.8; font-size: 15px; margin-bottom: 20px; }
.rekomendasi-section { background: #FFF8E1; border-radius: 12px; padding: 20px; border-left: 4px solid #FF8C00; margin-top: 16px; }
.rekomendasi-title { font-weight: 600; color: #E65100; margin-bottom: 10px; }
.rekomendasi-text { color: #5D4037; font-size: 14px; line-height: 1.6; }
.feature-card { background: white; border-radius: 16px; padding: 24px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); margin-bottom: 20px; }
.feature-header { font-size: 18px; font-weight: 600; color: #1B4D3E; margin-bottom: 20px; }
.sidebar-header { background: linear-gradient(135deg, #1B4D3E 0%, #2D6A5A 100%); border-radius: 12px; padding: 20px; text-align: center; color: white; margin-bottom: 20px; }
.sidebar-logo { font-size: 40px; margin-bottom: 8px; }
.sidebar-title { font-size: 16px; font-weight: 600; }
.sidebar-footer { margin-top: 30px; padding-top: 20px; border-top: 1px solid #E8ECF0; text-align: center; color: #6C757D; font-size: 12px; }
.metric-card { background: linear-gradient(145deg, #FFFFFF 0%, #F8F9FA 100%); border-radius: 12px; padding: 20px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.06); border: 1px solid #E8ECF0; }
.metric-value { font-size: 32px; font-weight: 700; color: #1B4D3E; }
.metric-label { font-size: 14px; color: #6C757D; margin-top: 4px; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;} .stDeployButton {display: none;}
.section-header { font-size: 20px; font-weight: 600; color: #1B4D3E; margin: 30px 0 20px; display: flex; align-items: center; gap: 10px; }
.section-header::before { content: ''; width: 4px; height: 24px; background: linear-gradient(180deg, #1B4D3E, #34CE57); border-radius: 2px; }
.glcm-table { width: 100%; border-collapse: separate; border-spacing: 0 8px; }
.glcm-table th { background: #1B4D3E; color: white; padding: 14px 16px; text-align: left; }
.glcm-table th:first-child { border-radius: 10px 0 0 10px; }
.glcm-table th:last-child { border-radius: 0 10px 10px 0; }
.glcm-table td { background: #F8F9FA; padding: 14px 16px; }
.glcm-table td:first-child { border-left: 1px solid #E8ECF0; border-radius: 10px 0 0 10px; font-weight: 500; color: #1B4D3E; }
.glcm-table td:last-child { border-right: 1px solid #E8ECF0; border-radius: 0 10px 10px 0; }
.glcm-value { font-weight: 600; font-size: 16px; }
.glcm-value-high { color: #DC3545; }
.glcm-value-medium { color: #FF8C00; }
.glcm-value-low { color: #28A745; }
.pipeline-step { background: white; border-radius: 16px; padding: 24px; margin-bottom: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); border-left: 4px solid #1B4D3E; }
.pipeline-step-title { font-size: 18px; font-weight: 600; color: #1B4D3E; margin-bottom: 12px; display: flex; align-items: center; gap: 12px; }
.pipeline-step-number { background: linear-gradient(135deg, #1B4D3E, #34CE57); color: white; width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 14px; }
.pipeline-step-desc { color: #495057; line-height: 1.8; font-size: 15px; }
.evaluation-card { background: white; border-radius: 16px; padding: 24px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); margin-bottom: 20px; }
.eval-title { font-size: 20px; font-weight: 600; color: #1B4D3E; margin-bottom: 20px; text-align: center; }
.metric-row { display: flex; justify-content: space-around; padding: 16px 0; border-bottom: 1px solid #E8ECF0; }
.metric-item { text-align: center; }
.metric-item-value { font-size: 28px; font-weight: 700; color: #1B4D3E; }
.metric-item-label { font-size: 14px; color: #6C757D; margin-top: 4px; }
</style>""", unsafe_allow_html=True)

# ============================================================
# LOAD MODELS
# ============================================================
# ===============================
# LOAD MAIN MODELS (WAJIB)
# ===============================
try:
    knn = joblib.load("knn_model.pkl")
    svm = joblib.load("svm_model.pkl")
    voting = joblib.load("voting_model.pkl")
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca.pkl")

    st.sidebar.markdown(
        '<div class="sidebar-header"><div class="sidebar-logo">🦐</div><div class="sidebar-title">Sistem Deteksi Udang</div></div>',
        unsafe_allow_html=True
    )

except FileNotFoundError as e:
    st.error(f"Model utama tidak ditemukan: {e}")
    st.stop()


# ===============================
# LOAD EVALUATION FILES (OPSIONAL)
# ===============================
try:
    X_test_eval = joblib.load("X_test.pkl")
    y_test_eval = joblib.load("y_test.pkl")
    eval_available = True
except:
    eval_available = False


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def count_images_per_class():
    counts = {}
    for folder in ["1. Healthy", "2. BG", "3. WSSV", "4. WSSV_BG"]:
        path = os.path.join("data_udang", folder)
        try:
            counts[folder] = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        except:
            counts[folder] = 0
    return counts

counts = count_images_per_class()

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("### Menu Navigasi")
menu = st.sidebar.radio("Pilih Halaman", ["Prediksi", "Galeri", "Pipeline", "Evaluasi Model"])

st.sidebar.markdown("### Ringkasan Data")
for folder, count in counts.items():
    folder_name = folder.split('. ', 1)[-1]
    st.sidebar.markdown(f'<div class="metric-card" style="margin-bottom: 10px; padding: 12px;"><div style="font-size: 12px; color: #6C757D;">{folder_name}</div><div class="metric-value" style="font-size: 24px;">{count}</div></div>', unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-footer">2025 - Proyek Klasifikasi Penyakit Udang</div>', unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="header-card"><div class="header-title">Deteksi Penyakit Udang Berbasis Citra</div><div class="header-subtitle">Sistem Klasifikasi Menggunakan Machine Learning (KNN & SVM)</div><div class="header-divider"></div></div>', unsafe_allow_html=True)

# ============================================================
# ROUTING
# ============================================================
if menu == "Prediksi":
    render_prediksi(knn, svm, voting, scaler, pca)

elif menu == "Galeri":
    render_galeri()

elif menu == "Pipeline":
    render_pipeline()

elif menu == "Evaluasi Model":
    if eval_available:
        render_evaluasi(knn, svm, voting, X_test_eval, y_test_eval)
    else:
        st.warning("File evaluasi tidak tersedia pada versi deployment.")

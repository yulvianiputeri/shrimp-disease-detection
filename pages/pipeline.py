import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
import mahotas
from PIL import Image, ImageEnhance
from utils import get_sample_image_from_dataset, generate_augmentation_samples, generate_preprocessing_samples

def render_pipeline():
    """Render Pipeline page with visualizations"""
        
    # ============================================================
    # DATASET STATISTICS
    # ============================================================
    st.markdown("---")
    st.markdown("### Statistik Dataset")
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    with col_stat1:
        st.markdown("""<div class="metric-card">
        <div class="metric-value">1,149</div>
        <div class="metric-label">Total Images</div>
        </div>""", unsafe_allow_html=True)
    
    with col_stat2:
        st.markdown("""<div class="metric-card">
        <div class="metric-value">9,192</div>
        <div class="metric-label">After Augmentation</div>
        </div>""", unsafe_allow_html=True)
    
    with col_stat3:
        st.markdown("""<div class="metric-card">
        <div class="metric-value">7,353</div>
        <div class="metric-label">Training Set (80%)</div>
        </div>""", unsafe_allow_html=True)
    
    with col_stat4:
        st.markdown("""<div class="metric-card">
        <div class="metric-value">1,839</div>
        <div class="metric-label">Test Set (20%)</div>
        </div>""", unsafe_allow_html=True)
    
    # ============================================================
    # TAHAP 1: DATA COLLECTION & AUGMENTATION
    # ============================================================
    st.markdown("---")
    st.markdown("### 1: Data Collection & Augmentation")
    
    # Check if dataset exists
    dataset_exists = os.path.exists("data_udang") and any(
        os.path.exists(os.path.join("data_udang", folder)) 
        for folder in ["1. Healthy", "2. BG", "3. WSSV", "4. WSSV_BG"]
    )
    
    sample_img = None
    
    if dataset_exists:
        try:
            sample_img = get_sample_image_from_dataset()
            if sample_img:
                st.success("✅ Dataset ditemukan! Menggunakan sample dari dataset.")
        except Exception as e:
            st.warning(f"⚠️ Error loading dataset: {str(e)}")
            dataset_exists = False
    
    if not dataset_exists or sample_img is None:
        st.warning("⚠️ Dataset tidak tersedia (deployment mode). Silakan upload gambar udang untuk demo.")
        
        uploaded_demo = st.file_uploader(
            "Upload gambar udang untuk demo augmentasi:",
            type=["jpg", "png", "jpeg"],
            key="pipeline_demo_upload"
        )
        
        if uploaded_demo:
            sample_img = Image.open(uploaded_demo).convert('RGB')
            st.success("✅ Gambar berhasil di-upload!")
        else:
            st.info("👆 Upload gambar udang di atas untuk melihat demo augmentasi data.")
    
    if sample_img:
        aug_samples = generate_augmentation_samples(sample_img)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.image(aug_samples['original'], caption="Original", use_container_width=True)
        with col2:
            st.image(aug_samples['flip'], caption="Horizontal Flip", use_container_width=True)
        with col3:
            st.image(aug_samples['rotate'], caption="Rotate +10°", use_container_width=True)
        with col4:
            st.image(aug_samples['brightness'], caption="Brightness +20%", use_container_width=True)
        
        st.info("💡 Setiap gambar original menghasilkan 8 variasi (flip, rotate ±10°, brightness ±20%, contrast ±20%)")
        
        # ============================================================
        # TAHAP 2: PREPROCESSING
        # ============================================================
        st.markdown("---")
        st.markdown("### Tahap 2: Preprocessing")
        
        preproc_samples = generate_preprocessing_samples(sample_img)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image(preproc_samples['original'], caption="Original (High Res)", use_container_width=True)
        with col2:
            st.image(preproc_samples['resized'], caption="Resized (128x128)", use_container_width=True)
        with col3:
            st.image(preproc_samples['grayscale'], caption="Grayscale", use_container_width=True)
        
        # ============================================================
        # TAHAP 3: FEATURE EXTRACTION
        # ============================================================
        st.markdown("---")
        st.markdown("### Tahap 3: Feature Extraction")
        
        img_np = np.array(preproc_samples['grayscale'])
        
        from skimage.feature import local_binary_pattern
        lbp = local_binary_pattern(img_np, 24, 3, 'uniform')
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_lbp, ax_lbp = plt.subplots(figsize=(6, 5))
            im_lbp = ax_lbp.imshow(lbp, cmap='viridis')
            ax_lbp.set_title('LBP (Local Binary Pattern)', fontweight='bold')
            ax_lbp.axis('off')
            plt.colorbar(im_lbp, ax=ax_lbp)
            st.pyplot(fig_lbp)
            plt.close()
            st.caption("📐 LBP menghasilkan 26 features (histogram bins)")
        
        with col2:
            glcm_features = mahotas.features.haralick(img_np).mean(axis=0)
            
            fig_glcm, ax_glcm = plt.subplots(figsize=(6, 5))
            ax_glcm.barh(range(len(glcm_features)), glcm_features, color='#1B4D3E')
            ax_glcm.set_yticks(range(len(glcm_features)))
            ax_glcm.set_yticklabels([f'F{i+1}' for i in range(len(glcm_features))], fontsize=8)
            ax_glcm.set_xlabel('Value')
            ax_glcm.set_title('GLCM Features (Haralick)', fontweight='bold')
            ax_glcm.invert_yaxis()
            st.pyplot(fig_glcm)
            plt.close()
            st.caption("📊 GLCM menghasilkan 13 features (texture properties)")
        
        if os.path.exists("pca_clusters_augmented.png"):
            st.image("pca_clusters_augmented.png", caption="PCA Visualization: 38 components, 100% variance retained", use_container_width=True)
            st
        
        # ============================================================
        # TAHAP 4: PCA VISUALIZATION
        # ============================================================
        st.markdown("---")
        st.markdown("### Tahap 4: Normalization & PCA")
        
        if os.path.exists("pca_clusters_augmented.png"):
            st.image("pca_clusters_augmented.png", caption="PCA Visualization: 38 components, 100% variance retained", use_container_width=True)
        else:
            st.info("PCA visualization akan muncul setelah training model")
        
        st.markdown("""
        **Proses:**
        1. **StandardScaler**: Normalisasi 39 features → mean=0, std=1
        2. **PCA**: Reduksi dimensi 39 → 38 components
        3. **Variance retained**: 100% (tidak ada informasi hilang)
        """)
        
    else:
        st.warning("⚠️ Sample image tidak ditemukan di dataset. Pastikan folder 'data_udang' tersedia.")
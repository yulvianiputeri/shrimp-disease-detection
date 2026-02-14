import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mahotas
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.feature import local_binary_pattern
from utils import extract_features_single

def render_prediksi(knn, svm, voting, scaler, pca):
    """Render Prediksi page"""
    
    st.markdown('<div class="section-header">Unggah & Prediksi Gambar Udang</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Unggah gambar udang...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Load images
        img_original = Image.open(uploaded_file).convert('RGB')
        img_resized = img_original.resize((128, 128))
        img_np = np.array(img_resized)

        st.markdown('<div class="card"><div class="card-header">Gambar Input</div>', unsafe_allow_html=True)
        st.image(img_original, caption="Gambar yang Diunggah", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Extract features & predict
        features = extract_features_single(img_np).reshape(1, -1)
        features_scaled = scaler.transform(features)
        features_pca = pca.transform(features_scaled)

        knn_pred = knn.predict(features_pca)[0]
        svm_pred = svm.predict(features_pca)[0]
        voting_pred = voting.predict(features_pca)[0]
        
        knn_proba = knn.predict_proba(features_pca)[0]
        svm_proba = svm.predict_proba(features_pca)[0]
        voting_proba = voting.predict_proba(features_pca)[0]

        st.markdown('<div class="section-header">Hasil Prediksi Model</div>', unsafe_allow_html=True)
        
        # ============================================================
        # VOTING ENSEMBLE - FINAL PREDICTION (Highlighted)
        # ============================================================
        voting_class = 'Sehat' if voting_pred == 0 else 'Terintegrasi'
        voting_confidence = voting_proba[voting_pred] * 100
        voting_color = '#4CAF50' if voting_pred == 0 else '#F44336'
        
        st.markdown(f'''
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 20px; padding: 30px; text-align: center; 
                    margin-bottom: 30px; box-shadow: 0 8px 30px rgba(102,126,234,0.3);">
            <div style="color: white; font-size: 16px; font-weight: 500; margin-bottom: 10px; opacity: 0.9;">
                FINAL PREDICTION (Voting Ensemble)
            </div>
            <div style="color: white; font-size: 48px; font-weight: 700; margin: 20px 0;">
                {voting_confidence:.1f}%
            </div>
            <div style="color: white; font-size: 28px; font-weight: 600; margin-bottom: 20px;">
                {voting_class}
            </div>
            <div style="background: rgba(255,255,255,0.2); border-radius: 15px; height: 40px; overflow: hidden; margin: 20px 0;">
                <div style="display: flex; height: 100%;">
                    <div style="width: {voting_proba[0]*100}%; background: #4CAF50; display: flex; align-items: center; justify-content: center; font-size: 14px; font-weight: bold; color: white;">
                        Sehat
                    </div>
                    <div style="width: {voting_proba[1]*100}%; background: #F44336; display: flex; align-items: center; justify-content: center; font-size: 14px; font-weight: bold; color: white;">
                        Sakit
                    </div>
                </div>
            </div>
            <div style="color: white; font-size: 14px; opacity: 0.85;">
                Sehat: {voting_proba[0]*100:.1f}% | Sakit: {voting_proba[1]*100:.1f}%
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        # ============================================================
        # INDIVIDUAL MODELS - Supporting Details
        # ============================================================
        st.markdown('<div style="color: #666; font-size: 14px; margin-bottom: 15px; text-align: center;">Detail dari Model Individual:</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)

        with col1:
            # KNN prediction
            knn_class = 'Sehat' if knn_pred == 0 else 'Terintegrasi'
            knn_confidence = knn_proba[knn_pred] * 100
            knn_color = '#4CAF50' if knn_pred == 0 else '#F44336'
            knn_sc = 'pred-card-healthy' if knn_pred == 0 else 'pred-card-diseased'
            
            st.markdown(f'''
            <div class="pred-card {knn_sc}" style="border-left: 5px solid {knn_color};">
                <div class="pred-header" style="color: {knn_color};">Model KNN</div>
                <div class="pred-percentage" style="font-size: 32px; font-weight: bold; color: {knn_color};">
                    {knn_confidence:.1f}%<br><span style="font-size: 18px;">{knn_class}</span>
                </div>
                <div class="progress-container" style="height: 30px; background: #E8ECF0; border-radius: 15px; overflow: hidden; margin: 16px 0;">
                    <div style="display: flex; height: 100%;">
                        <div style="width: {knn_proba[0]*100}%; background: #4CAF50; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold; color: white;">Sehat</div>
                        <div style="width: {knn_proba[1]*100}%; background: #F44336; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold; color: white;">Sakit</div>
                    </div>
                </div>
                <div style="font-size: 13px; color: #666;">
                    Sehat: {knn_proba[0]*100:.1f}% | Sakit: {knn_proba[1]*100:.1f}%
                </div>
            </div>
            ''', unsafe_allow_html=True)

        with col2:
            # SVM prediction
            svm_class = 'Sehat' if svm_pred == 0 else 'Terintegrasi'
            svm_confidence = svm_proba[svm_pred] * 100
            svm_color = '#4CAF50' if svm_pred == 0 else '#F44336'
            svm_sc = 'pred-card-healthy' if svm_pred == 0 else 'pred-card-diseased'
            
            st.markdown(f'''
            <div class="pred-card {svm_sc}" style="border-left: 5px solid {svm_color};">
                <div class="pred-header" style="color: {svm_color};">Model SVM</div>
                <div class="pred-percentage" style="font-size: 32px; font-weight: bold; color: {svm_color};">
                    {svm_confidence:.1f}%<br><span style="font-size: 18px;">{svm_class}</span>
                </div>
                <div class="progress-container" style="height: 30px; background: #E8ECF0; border-radius: 15px; overflow: hidden; margin: 16px 0;">
                    <div style="display: flex; height: 100%;">
                        <div style="width: {svm_proba[0]*100}%; background: #4CAF50; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold; color: white;">Sehat</div>
                        <div style="width: {svm_proba[1]*100}%; background: #F44336; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold; color: white;">Sakit</div>
                    </div>
                </div>
                <div style="font-size: 13px; color: #666;">
                    Sehat: {svm_proba[0]*100:.1f}% | Sakit: {svm_proba[1]*100:.1f}%
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        # === Grad-CAM Visualization ===
        st.markdown('<div class="section-header">Area yang Mempengaruhi Prediksi</div>', unsafe_allow_html=True)
        st.markdown('<div class="heatmap-description">Visualisasi area yang paling berpengaruh menggunakan Grad-CAM.</div>', unsafe_allow_html=True)
        st.markdown('<div class="heatmap-container">', unsafe_allow_html=True)

        try:
            # Segmentation
            gray_mask = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            contours_m, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            shrimp_mask = np.zeros_like(binary)
            if len(contours_m) > 0:
                largest = max(contours_m, key=cv2.contourArea)
                cv2.drawContours(shrimp_mask, [largest], -1, 255, -1)
            else:
                shrimp_mask = np.ones_like(binary) * 255
            
            # Grad-Cam with probability scores
            h, w = img_np.shape[:2]
            patch_size = 32
            stride = 16
            
            prob_map = np.zeros((h, w), dtype=np.float32)
            count_map = np.zeros((h, w), dtype=np.int32)
            
            for y in range(0, h - patch_size + 1, stride):
                for x in range(0, w - patch_size + 1, stride):
                    window_mask = shrimp_mask[y:y+patch_size, x:x+patch_size]
                    shrimp_ratio = (window_mask > 0).sum() / (patch_size * patch_size)
                    
                    if shrimp_ratio < 0.3:
                        continue
                    
                    window = img_np[y:y+patch_size, x:x+patch_size]
                    gray_win = cv2.cvtColor(window, cv2.COLOR_RGB2GRAY)
                    
                    try:
                        lbp = local_binary_pattern(gray_win, 24, 3, 'uniform')
                        lbp_hist, _ = np.histogram(lbp, bins=26, range=(0, 26))
                        lbp_hist = lbp_hist.astype("float") / (lbp_hist.sum() + 1e-6)
                        
                        if gray_win.shape[0] >= 5 and gray_win.shape[1] >= 5:
                            glcm = mahotas.features.haralick(gray_win).mean(axis=0)
                        else:
                            glcm = np.zeros(13)
                        
                        feats = np.hstack([lbp_hist, glcm]).reshape(1, -1)
                        feats_scaled = scaler.transform(feats)
                        feats_pca = pca.transform(feats_scaled)
                        
                        proba = svm.predict_proba(feats_pca)[0]
                        diseased_prob = proba[1]
                        
                        prob_map[y:y+patch_size, x:x+patch_size] += diseased_prob
                        count_map[y:y+patch_size, x:x+patch_size] += 1
                    except:
                        continue
            
            attention_map = np.divide(prob_map, count_map, where=count_map>0, out=np.zeros_like(prob_map))
            attention_map[shrimp_mask == 0] = 0
            attention_smooth = gaussian_filter(attention_map, sigma=2)
            attention_smooth[shrimp_mask == 0] = 0
            attention_smooth = np.clip(attention_smooth, 0, 1)
            
            fg_values = attention_smooth[shrimp_mask > 0]
            if len(fg_values) > 0:
                max_score = fg_values.max()
                mean_score = fg_values.mean()
                median_score = np.median(fg_values)
            else:
                max_score = mean_score = median_score = 0
            
            # 3-Panel Visualization
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
            
            ax1.imshow(img_np)
            ax1.set_title('Gambar Asli', fontsize=14, fontweight='bold')
            ax1.axis('off')
            
            im = ax2.imshow(attention_smooth, cmap='RdYlGn_r', vmin=0, vmax=1)
            ax2.set_title(f'Peta Perhatian\nMax: {max_score:.3f}, Mean: {mean_score:.3f}', fontsize=14, fontweight='bold')
            ax2.axis('off')
            plt.colorbar(im, ax=ax2, label='Prob. Sakit')
            
            ax3.imshow(img_np)
            ax3.imshow(attention_smooth, cmap='RdYlGn_r', alpha=0.5, vmin=0, vmax=1)
            ax3.set_title('Overlay', fontsize=14, fontweight='bold')
            ax3.axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        except Exception as e:
            st.error(f"Error: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Disease Information
        st.markdown("""<div class="section-header">Informasi Penyakit</div>""", unsafe_allow_html=True)
        
# === STEP 1: MULTI-METHOD Foreground Segmentation ===
        st.markdown("""<div class="section-header">Area Penyakit Terdeteksi</div>""", unsafe_allow_html=True)

        if 'attention_smooth' in locals() and attention_smooth is not None:
            try:
                # === User Manual Override Option ===
                use_manual = st.checkbox("🔧 Mode Manual (centang jika segmentasi otomatis gagal)", value=False)
                
                if use_manual:
                    st.warning("⚠️ **Mode Manual Aktif** - Seluruh gambar akan dianalisis (tanpa segmentasi background)")
                    foreground_mask = np.ones((img_np.shape[0], img_np.shape[1]), dtype=np.uint8) * 255
                    fg_mask_bool = foreground_mask > 0
                    mask_ratio = 1.0
                
                else:
                    # === AUTO SEGMENTATION ===
                    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                    edges = cv2.Canny(blurred, 30, 100)
                    
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
                    
                    contours_edge, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    h, w = img_np.shape[:2]
                    mask_grabcut = np.zeros((h, w), np.uint8)
                    
                    margin_x = int(w * 0.15)
                    margin_y = int(h * 0.15)
                    rect = (margin_x, margin_y, w - 2*margin_x, h - 2*margin_y)
                    
                    bgd_model = np.zeros((1, 65), np.float64)
                    fgd_model = np.zeros((1, 65), np.float64)
                    
                    try:
                        cv2.grabCut(img_np, mask_grabcut, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
                        mask_grabcut_binary = np.where((mask_grabcut == 2) | (mask_grabcut == 0), 0, 1).astype('uint8')
                        foreground_mask_grabcut = mask_grabcut_binary * 255
                    except:
                        foreground_mask_grabcut = None
                    
                    _, mask_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    masks = []
                    
                    if len(contours_edge) > 0:
                        mask_edge = np.zeros_like(gray)
                        largest_edge_contour = max(contours_edge, key=cv2.contourArea)
                        cv2.drawContours(mask_edge, [largest_edge_contour], -1, 255, -1)
                        masks.append(mask_edge)
                    
                    if foreground_mask_grabcut is not None:
                        masks.append(foreground_mask_grabcut)
                    
                    masks.append(mask_thresh)
                    
                    if len(masks) > 0:
                        mask_sum = np.sum(masks, axis=0)
                        foreground_mask = (mask_sum > (len(masks) * 127)).astype(np.uint8) * 255
                    else:
                        foreground_mask = np.ones_like(gray) * 255
                    
                    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel_clean, iterations=2)
                    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel_clean, iterations=1)
                    
                    contours_final, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if len(contours_final) > 0:
                        largest = max(contours_final, key=cv2.contourArea)
                        foreground_mask = np.zeros_like(foreground_mask)
                        cv2.drawContours(foreground_mask, [largest], -1, 255, -1)
                    
                    fg_mask_bool = foreground_mask > 0
                    mask_ratio = fg_mask_bool.sum() / fg_mask_bool.size
                    
                    st.info(f"""
                    🔍 **Segmentasi Otomatis:**
                    - Metode: Edge + GrabCut + Threshold (voting)
                    - Area mask: {mask_ratio*100:.1f}%
                    - Status: {'✅ OK' if 0.2 < mask_ratio < 0.8 else '⚠️ Mungkin gagal - coba Mode Manual'}
                    """)
                    
                    if mask_ratio > 0.95:
                        st.error(f"❌ Segmentasi gagal (mask {mask_ratio*100:.1f}%). Otomatis beralih ke Mode Manual.")
                        foreground_mask = np.ones_like(foreground_mask) * 255
                        fg_mask_bool = foreground_mask > 0
                        mask_ratio = 1.0
                
                # === STEP 2: Apply Mask ===
                attention_masked = attention_smooth.copy()
                if mask_ratio < 0.99:
                    attention_masked[~fg_mask_bool] = 0
                
                # === STEP 3: Guide ===
                st.markdown("""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 20px; border-radius: 12px; color: white; margin-bottom: 20px;">
                    <h4 style="margin:0; color: white;">Cara Membaca:</h4>
                    <ul style="margin-top: 10px;">
                        <li><strong style="color: #ff6b6b;">MERAH</strong> = Pola sakit</li>
                        <li><strong style="color: #ffd93d;">KUNING</strong> = Pola mencurigakan</li>
                        <li><strong style="color: #6bcf7f;">HIJAU</strong> = Pola sehat</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # === STEP 4: Stats ===
                if mask_ratio < 0.99:
                    attention_fg = attention_masked[fg_mask_bool]
                else:
                    attention_fg = attention_smooth.flatten()
                
                if len(attention_fg) > 50:
                    max_score = attention_fg.max()
                    mean_score = attention_fg.mean()
                    threshold_high = np.percentile(attention_fg, 85)
                    threshold_medium = np.percentile(attention_fg, 70)
                else:
                    max_score = mean_score = threshold_high = threshold_medium = 0
                
                # === STEP 5: 4-PANEL VISUALISASI ===
                fig, axes = plt.subplots(2, 2, figsize=(16, 16))
                
                # Panel 1
                axes[0, 0].imshow(img_np)
                axes[0, 0].set_title('① Gambar Asli', fontsize=16, fontweight='bold')
                axes[0, 0].axis('off')
                
                # Panel 2 - Segmentation
                if mask_ratio < 0.99:
                    mask_viz = np.zeros_like(img_np)
                    mask_viz[fg_mask_bool] = [255, 255, 255]
                    mask_viz[~fg_mask_bool] = [50, 50, 200]
                    title2 = f'② Segmentasi (Putih=Udang, Biru=Background)\nArea: {mask_ratio*100:.1f}%'
                else:
                    mask_viz = img_np.copy()
                    title2 = '② Segmentasi: Mode Manual (Semua Area)'
                
                axes[0, 1].imshow(mask_viz)
                axes[0, 1].set_title(title2, fontsize=16, fontweight='bold')
                axes[0, 1].axis('off')
                
                # Panel 3 - Heatmap
                im = axes[1, 0].imshow(attention_masked, cmap='RdYlGn_r', vmin=0, vmax=1)
                if threshold_high > 0:
                    axes[1, 0].contour(attention_masked, levels=[threshold_medium], colors='yellow', linewidths=3, linestyles='--')
                    axes[1, 0].contour(attention_masked, levels=[threshold_high], colors='red', linewidths=3, linestyles='-')
                axes[1, 0].set_title(f'③ Attention Map\nMax: {max_score:.3f}, Mean: {mean_score:.3f}', fontsize=16, fontweight='bold')
                axes[1, 0].axis('off')
                plt.colorbar(im, ax=axes[1, 0], label='Skor')
                
                # Panel 4 - Overlay
                img_overlay = img_np.copy()
                overlay = np.zeros_like(img_overlay, dtype=np.float32)
                
                if mask_ratio < 0.99:
                    mask_high = (attention_masked > threshold_high) & fg_mask_bool
                    mask_med = (attention_masked > threshold_medium) & (attention_masked <= threshold_high) & fg_mask_bool
                else:
                    mask_high = attention_smooth > threshold_high
                    mask_med = (attention_smooth > threshold_medium) & (attention_smooth <= threshold_high)
                
                overlay[mask_high] = [255, 0, 0]
                overlay[mask_med] = [255, 255, 0]
                img_overlay = cv2.addWeighted(img_overlay, 0.6, overlay.astype(np.uint8), 0.4, 0)
                
                # Draw contours
                binary_high = mask_high.astype(np.uint8) * 255
                contours_det, _ = cv2.findContours(binary_high, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for cnt in contours_det:
                    if cv2.contourArea(cnt) > 100:
                        cv2.drawContours(img_overlay, [cnt], -1, (0, 255, 255), 3)
                        x, y, w, h = cv2.boundingRect(cnt)
                        cv2.rectangle(img_overlay, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                axes[1, 1].imshow(img_overlay)
                axes[1, 1].set_title(f'④ Deteksi ({len([c for c in contours_det if cv2.contourArea(c) > 100])} area)', fontsize=16, fontweight='bold')
                axes[1, 1].axis('off')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())
        else:
            st.warning("Attention map tidak tersedia.")

        col_dis1, col_dis2 = st.columns(2)

        # Feature Values Section
        st.markdown("""<div class="section-header">Nilai Fitur Ekstraksi</div>""", unsafe_allow_html=True)
        
        try:
            glcm_names = [
                "Angular Second Moment (ASM)", "Contrast", "Correlation", 
                "Sum of Squares: Variance", "Inverse Difference Moment (IDM)", 
                "Sum Average", "Sum Variance", "Sum Entropy",
                "Entropy", "Difference Variance", "Difference Entropy", 
                "Info. Corr. 1", "Info. Corr. 2"
            ]
            glcm_values = features[0, 26:39] if features.shape[1] >= 39 else np.zeros(13)
            
            st.markdown("""<div class="feature-card"><div class="feature-header">Nilai Fitur GLCM (Haralick)</div>""", unsafe_allow_html=True)
            
            glcm_html = '<table class="glcm-table"><thead><tr><th>Fitur</th><th>Nilai</th></tr></thead><tbody>'
            for i, (name, val) in enumerate(zip(glcm_names, glcm_values)):
                val_class = "glcm-value-high" if abs(val) > 0.5 else ("glcm-value-medium" if abs(val) > 0.2 else "glcm-value-low")
                glcm_html += f'<tr><td>{name}</td><td class="glcm-value {val_class}">{val:.4f}</td></tr>'
            glcm_html += '</tbody></table>'
            st.markdown(glcm_html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error displaying GLCM values: {str(e)}")
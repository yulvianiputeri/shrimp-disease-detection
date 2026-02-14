import streamlit as st
from utils import load_gallery_images

def render_galeri():
    """Render Galeri page"""
    
    st.markdown("""<div class="section-header">Galeri Sample Gambar Udang</div>""", unsafe_allow_html=True)
    
    gallery_folders = [
        ("1. Healthy", "Sehat", "gallery-label-healthy"),
        ("2. BG", "White Feces (BG)", "gallery-label-diseased"),
        ("3. WSSV", "White Spot (WSSV)", "gallery-label-diseased"),
        ("4. WSSV_BG", "WSSV + BG", "gallery-label-diseased")
    ]
    
    for folder, label, label_class in gallery_folders:
        st.markdown(f'<div style="margin: 20px 0;"><h3 style="color: #1B4D3E; margin-bottom: 15px;">{label}</h3>', unsafe_allow_html=True)
        
        gallery_images = load_gallery_images(folder, label, sample_size=6)
        
        if gallery_images:
            cols = st.columns(min(len(gallery_images), 3))
            for idx, (img, img_label) in enumerate(gallery_images):
                with cols[idx % 3]:
                    st.image(img, caption=img_label, use_container_width=True)
        else:
            st.info(f"Tidak ada gambar ditemukan di folder {folder}")
        st.markdown('</div>', unsafe_allow_html=True)
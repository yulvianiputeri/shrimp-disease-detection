def render_pipeline():
    st.title("🦐 Full Pipeline - Sistem Deteksi Penyakit Udang")

    # ============================================================
    # STEP 0: LOAD IMAGE (AUTO / UPLOAD)
    # ============================================================
    st.markdown("---")
    st.header("1️Load Image")

    sample_img = None

    # Mode otomatis (lokal)
    if os.path.exists("data_udang"):
        from utils import get_sample_image_from_dataset
        sample_img = get_sample_image_from_dataset()
        st.success("Mode Lokal: Menggunakan dataset otomatis")

    # Kalau dataset tidak ada → upload
    if sample_img is None:
        st.info("Mode Deploy: Upload gambar untuk demo pipeline")

        uploaded_file = st.file_uploader(
            "Upload gambar udang",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            sample_img = Image.open(uploaded_file)

    if sample_img is None:
        st.stop()

    st.image(sample_img, caption="Input Image", use_container_width=True)

    # ============================================================
    # STEP 2: AUGMENTATION
    # ============================================================
    st.markdown("---")
    st.header("Data Augmentation")

    col1, col2, col3, col4 = st.columns(4)

    img_np = np.array(sample_img)

    flip = np.fliplr(img_np)
    rotate = np.rot90(img_np)
    brightness = np.clip(img_np + 30, 0, 255)
    contrast = np.clip(1.2 * img_np, 0, 255)

    with col1:
        st.image(flip, caption="Flip")
    with col2:
        st.image(rotate, caption="Rotate")
    with col3:
        st.image(brightness, caption="Brightness +")
    with col4:
        st.image(contrast, caption="Contrast +")

    # ============================================================
    # STEP 3: PREPROCESSING
    # ============================================================
    st.markdown("---")
    st.header("Preprocessing")

    resized = sample_img.resize((128, 128))
    gray = resized.convert("L")

    col1, col2 = st.columns(2)

    with col1:
        st.image(resized, caption="Resized 128x128")

    with col2:
        st.image(gray, caption="Grayscale")

    gray_np = np.array(gray)

    # ============================================================
    # STEP 4: FEATURE EXTRACTION
    # ============================================================
    st.markdown("---")
    st.header("Feature Extraction")

    # LBP
    lbp = local_binary_pattern(gray_np, 24, 3, 'uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(),
                                bins=np.arange(0, 27),
                                range=(0, 26))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    # GLCM (Haralick)
    glcm_features = mahotas.features.haralick(gray_np).mean(axis=0)

    features = np.hstack([lbp_hist, glcm_features])

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots()
        ax1.imshow(lbp, cmap='viridis')
        ax1.set_title("LBP")
        ax1.axis('off')
        st.pyplot(fig1)
        plt.close()

    with col2:
        fig2, ax2 = plt.subplots()
        ax2.barh(range(len(glcm_features)), glcm_features)
        ax2.set_title("GLCM Features")
        st.pyplot(fig2)
        plt.close()

    st.success(f"Total Features: {len(features)} (LBP + GLCM)")

    # ============================================================
    # STEP 5: NORMALIZATION + PCA
    # ============================================================
    st.markdown("---")
    st.header("5️⃣ Normalization & PCA")

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform([features])

    pca = PCA(n_components=min(len(features), 10))
    features_pca = pca.fit_transform(features_scaled)

    st.write("PCA Components:", features_pca.shape[1])
    st.write("Explained Variance Ratio:", pca.explained_variance_ratio_)

    # ============================================================
    # STEP 6: PREDICTION (OPTIONAL IF MODEL EXISTS)
    # ============================================================
    st.markdown("---")
    st.header("Prediction")

    if os.path.exists("model.pkl"):
        model = joblib.load("model.pkl")

        prediction = model.predict(features_scaled)
        st.success(f"Hasil Prediksi: {prediction[0]}")

    else:
        st.warning("Model belum tersedia (model.pkl tidak ditemukan)")

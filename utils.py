# MultipleFiles/utils.py
import os
import numpy as np
import cv2
from PIL import Image
from skimage.feature import local_binary_pattern
import mahotas

def extract_features_single(img):
    """
    Ekstraksi fitur LBP dan Haralick (GLCM) dari satu gambar.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    radius, n_points, METHOD = 3, 24, 'uniform' 
    lbp = local_binary_pattern(gray, n_points, radius, METHOD)
    hist, _ = np.histogram(lbp, bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6) 

    glcm = mahotas.features.haralick(gray).mean(axis=0)
    
    return np.hstack([hist, glcm])

def load_gallery_images(base_folder, label, sample_size=6):
    """
    Memuat contoh gambar untuk galeri.
    """
    images = []
    folder_path = os.path.join("data_udang", base_folder)
    
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return []

    files = os.listdir(folder_path)

    image_files = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg'))][:sample_size]

    for fname in image_files:
        img_path = os.path.join(folder_path, fname)
        try:
            img = Image.open(img_path).convert('RGB').resize((128,128))
            images.append((img, label))
        except Exception as e:
            print(f"Error loading gallery image {img_path}: {e}")
            continue
    return images

def generate_augmentation_samples(img):
    """
    Generate augmented versions of an image for pipeline visualization.
    Returns: dict with 'flip', 'rotate', 'brightness', 'contrast' keys
    """
    from PIL import ImageEnhance
    
    aug_samples = {}
    aug_samples['original'] = img
    aug_samples['flip'] = img.transpose(Image.FLIP_LEFT_RIGHT)
    aug_samples['rotate'] = img.rotate(15)
    
    enhancer_bright = ImageEnhance.Brightness(img)
    aug_samples['brightness'] = enhancer_bright.enhance(1.3)
    
    enhancer_contrast = ImageEnhance.Contrast(img)
    aug_samples['contrast'] = enhancer_contrast.enhance(1.3)
    
    return aug_samples

def generate_preprocessing_samples(img):
    """
    Generate preprocessing stages for pipeline visualization.
    Returns: dict with 'original', 'resized', 'grayscale' keys
    """
    preproc = {}
    preproc['original'] = img
    preproc['resized'] = img.resize((128, 128))
    preproc['grayscale'] = preproc['resized'].convert('L')
    
    return preproc

def get_sample_image_from_dataset(base_path="data_udang"):
    """
    Auto-select a sample image from dataset for pipeline visualization.
    Returns: PIL Image or None
    """
    sample_folders = ["1. Healthy", "3. WSSV", "2. BG", "4. WSSV_BG"]
    
    for folder in sample_folders:
        folder_path = os.path.join(base_path, folder)
        if os.path.exists(folder_path):
            files = [f for f in os.listdir(folder_path) 
                    if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if files:
                img_path = os.path.join(folder_path, files[0])
                try:
                    return Image.open(img_path).convert('RGB')
                except:
                    continue
    return None
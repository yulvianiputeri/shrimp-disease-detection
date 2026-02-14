"""
Pages module for Streamlit app
Contains all menu/page functions
"""

from .prediksi import render_prediksi
from .galeri import render_galeri
from .pipeline import render_pipeline
from .evaluasi import render_evaluasi

__all__ = [
    'render_prediksi',
    'render_galeri', 
    'render_pipeline',
    'render_evaluasi'
]
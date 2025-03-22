import streamlit as st
import requests
import json
import re
from io import BytesIO
import subprocess
import sys
import numpy as np

# Install scikit-learn if not present
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ModuleNotFoundError:
    st.warning("scikit-learn not found. Attempting installation...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        st.success("scikit-learn successfully installed.")
    except subprocess.CalledProcessError as e:
        st.error(f"Failed to install scikit-learn: {e}")
        st.stop()

# Install pdfplumber if not present
try:
    import pdfplumber
except ModuleNotFoundError:
    st.warning("pdfplumber not found. Attempting installation...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pdfplumber"])
        import pdfplumber
        st.success("pdfplumber successfully installed.")
    except subprocess.CalledProcessError as e:
        st.error(f"Failed to install pdfplumber: {e}")
        st.stop()

# ... (rest of your Streamlit app code remains the same)

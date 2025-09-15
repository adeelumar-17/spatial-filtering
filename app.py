import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Spatial Filtering & Smoothing",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Utility Functions
# -------------------------
def convert_to_grayscale(image):
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image

def convert_to_bw(image, threshold=127):
    gray = convert_to_grayscale(image)
    _, bw = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return bw

def apply_first_order_filter(image, filter_type):
    if filter_type == "Sobel X":
        return cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    elif filter_type == "Sobel Y":
        return cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    elif filter_type == "Sobel Combined":
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(sobelx**2 + sobely**2)
    elif filter_type == "Prewitt X":
        kernel = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=np.float32)
        return cv2.filter2D(image, cv2.CV_64F, kernel)
    elif filter_type == "Prewitt Y":
        kernel = np.array([[-1,-1,-1],[0,0,0],[1,1,1]], dtype=np.float32)
        return cv2.filter2D(image, cv2.CV_64F, kernel)
    elif filter_type == "Roberts X":
        kernel = np.array([[1,0],[0,-1]], dtype=np.float32)
        return cv2.filter2D(image, cv2.CV_64F, kernel)
    elif filter_type == "Roberts Y":
        kernel = np.array([[0,1],[-1,0]], dtype=np.float32)
        return cv2.filter2D(image, cv2.CV_64F, kernel)
    return image

def apply_second_order_filter(image, filter_type):
    if filter_type == "Laplacian":
        return cv2.Laplacian(image, cv2.CV_64F)
    elif filter_type == "Laplacian of Gaussian (LoG)":
        blurred = cv2.GaussianBlur(image, (5,5), 1.0)
        return cv2.Laplacian(blurred, cv2.CV_64F)
    elif filter_type == "Custom Laplacian (4-connected)":
        kernel = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=np.float32)
        return cv2.filter2D(image, cv2.CV_64F, kernel)
    elif filter_type == "Custom Laplacian (8-connected)":
        kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=np.float32)
        return cv2.filter2D(image, cv2.CV_64F, kernel)
    return image

def combine_filters(image, f1, f2, method="Add"):
    first_result = apply_first_order_filter(image, f1)
    second_result = apply_second_order_filter(image, f2)
    first_norm = cv2.normalize(first_result, None, 0, 255, cv2.NORM_MINMAX)
    second_norm = cv2.normalize(second_result, None, 0, 255, cv2.NORM_MINMAX)
    if method == "Add":
        combined = cv2.addWeighted(first_norm, 0.5, second_norm, 0.5, 0)
    elif method == "Multiply":
        combined = cv2.multiply(first_norm/255.0, second_norm/255.0) * 255
    elif method == "Maximum":
        combined = np.maximum(first_norm, second_norm)
    else:
        combined = cv2.subtract(first_norm, second_norm)
    return combined, first_result, second_result

def apply_smoothing_filter(image, filter_type, kernel_size=3):
    if filter_type == "Mean":
        return cv2.blur(image, (kernel_size, kernel_size))
    elif filter_type == "Median":
        return cv2.medianBlur(image, kernel_size)
    elif filter_type == "Mode":
        pad = kernel_size // 2
        padded = np.pad(image, pad, mode="edge")
        out = np.zeros_like(image)
        # Optimized mode filtering
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                window = padded[i:i+kernel_size, j:j+kernel_size].ravel()
                counts = np.bincount(window)
                out[i, j] = np.argmax(counts)
        return out
    return image

def normalize_for_display(image):
    if image.dtype != np.uint8:
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return image

def create_comparison_plot(images, titles):
    n = len(images)
    cols = 3 if n > 3 else n
    rows = int(np.ceil(n/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    axes = np.array(axes).reshape(-1)
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(normalize_for_display(img), cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    return fig

# -------------------------
# Main App
# -------------------------
st.markdown('<h1 style="text-align:center; color:#2E86AB;">🔍 Spatial Filtering & Smoothing</h1>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    uploaded = st.file_uploader("Upload an Image", type=['png','jpg','jpeg'])
    reset = st.button("🔄 Reset App")

# Reset functionality
if reset:
    st.session_state.clear()
    st.rerun()

if uploaded:
    image = Image.open(uploaded)
    arr = np.array(image)

    # Preprocessing
    preproc = st.sidebar.selectbox("Preprocessing", ["Grayscale","Black & White","Keep Original"])
    if preproc=="Black & White":
        th = st.sidebar.slider("Threshold", 0,255,127)
    
    if preproc=="Grayscale":
        proc_img = convert_to_grayscale(arr)
    elif preproc=="Black & White":
        proc_img = convert_to_bw(arr, th)
    else:
        proc_img = convert_to_grayscale(arr)

    # Filtering Options
    st.sidebar.markdown("### 🔹 Filtering")
    mode = st.sidebar.radio("Filter Mode", ["Individual","Combined"])
    if mode=="Individual":
        ftype = st.sidebar.selectbox("Filter Type", ["First-Order","Second-Order"])
        if ftype=="First-Order":
            f1 = st.sidebar.selectbox("First-Order Filter", ["Sobel X","Sobel Y","Sobel Combined","Prewitt X","Prewitt Y","Roberts X","Roberts Y"])
        else:
            f2 = st.sidebar.selectbox("Second-Order Filter", ["Laplacian","Laplacian of Gaussian (LoG)","Custom Laplacian (4-connected)","Custom Laplacian (8-connected)"])
    else:
        f1 = st.sidebar.selectbox("First-Order", ["Sobel X","Sobel Y","Sobel Combined","Prewitt X","Prewitt Y"])
        f2 = st.sidebar.selectbox("Second-Order", ["Laplacian","Laplacian of Gaussian (LoG)","Custom Laplacian (4-connected)","Custom Laplacian (8-connected)"])
        method = st.sidebar.selectbox("Combination", ["Add","Multiply","Maximum","Subtract"])

    # Smoothing
    st.sidebar.markdown("### 🔹 Smoothing")
    smoothing = st.sidebar.selectbox("Apply Smoothing", ["None","Mean","Median","Mode"])
    ksize = st.sidebar.slider("Kernel Size", 3,9,3, step=2)

    if st.sidebar.button("🚀 Apply"):
        with st.spinner("Processing image..."):
            if mode=="Individual":
                if ftype=="First-Order":
                    filt = apply_first_order_filter(proc_img, f1)
                    imgs = [proc_img, filt]
                    titles = ["Preprocessed", f1]
                else:
                    filt = apply_second_order_filter(proc_img, f2)
                    imgs = [proc_img, filt]
                    titles = ["Preprocessed", f2]
            else:
                comb, f1r, f2r = combine_filters(proc_img, f1, f2, method)
                imgs = [proc_img, f1r, f2r, comb]
                titles = ["Preprocessed", f1, f2, f"Combined ({method})"]
                filt = comb

            if smoothing != "None":
                smooth = apply_smoothing_filter(normalize_for_display(filt), smoothing, ksize)
                imgs.append(smooth)
                titles.append(f"{smoothing} Smoothing")

            fig = create_comparison_plot(imgs, titles)

            # Center the output
            col_left, col_center, col_right = st.columns([1,3,1])
            with col_center:
                st.pyplot(fig)
else:
    st.info("👆 Upload an image to get started!")

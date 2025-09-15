import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Set page configuration
st.set_page_config(
    page_title="Spatial Filtering App",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .filter-section {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .results-container {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def convert_to_grayscale(image):
    """Convert image to grayscale"""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image

def convert_to_bw(image, threshold=127):
    """Convert image to black and white"""
    gray = convert_to_grayscale(image)
    _, bw = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return bw

def apply_first_order_filter(image, filter_type):
    """Apply first-order derivative filters"""
    if filter_type == "Sobel X":
        return cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    elif filter_type == "Sobel Y":
        return cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    elif filter_type == "Sobel Combined":
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(sobelx**2 + sobely**2)
    elif filter_type == "Prewitt X":
        kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        return cv2.filter2D(image, cv2.CV_64F, kernel)
    elif filter_type == "Prewitt Y":
        kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
        return cv2.filter2D(image, cv2.CV_64F, kernel)
    elif filter_type == "Roberts Cross-Gradient X":
        kernel = np.array([[1, 0], [0, -1]], dtype=np.float32)
        return cv2.filter2D(image, cv2.CV_64F, kernel)
    elif filter_type == "Roberts Cross-Gradient Y":
        kernel = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        return cv2.filter2D(image, cv2.CV_64F, kernel)
    else:
        return image

def apply_second_order_filter(image, filter_type):
    """Apply second-order derivative filters"""
    if filter_type == "Laplacian":
        return cv2.Laplacian(image, cv2.CV_64F)
    elif filter_type == "Laplacian of Gaussian (LoG)":
        # First apply Gaussian blur, then Laplacian
        blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
        return cv2.Laplacian(blurred, cv2.CV_64F)
    elif filter_type == "Custom Laplacian (4-connected)":
        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(image, cv2.CV_64F, kernel)
    elif filter_type == "Custom Laplacian (8-connected)":
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
        return cv2.filter2D(image, cv2.CV_64F, kernel)
    else:
        return image

def combine_filters(image, first_order_type, second_order_type, combination_method="Add"):
    """Combine first and second order filters"""
    first_result = apply_first_order_filter(image, first_order_type)
    second_result = apply_second_order_filter(image, second_order_type)
    
    # Normalize both results to same range
    first_norm = cv2.normalize(first_result, None, 0, 255, cv2.NORM_MINMAX)
    second_norm = cv2.normalize(second_result, None, 0, 255, cv2.NORM_MINMAX)
    
    if combination_method == "Add":
        combined = cv2.addWeighted(first_norm, 0.5, second_norm, 0.5, 0)
    elif combination_method == "Multiply":
        combined = cv2.multiply(first_norm/255.0, second_norm/255.0) * 255
    elif combination_method == "Maximum":
        combined = np.maximum(first_norm, second_norm)
    else:  # Subtract
        combined = cv2.subtract(first_norm, second_norm)
    
    return combined, first_result, second_result

def normalize_for_display(image):
    """Normalize image for proper display"""
    if image.dtype != np.uint8:
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return image



# Main App
st.markdown('<h1 class="main-header">🔍 Spatial Filtering Application</h1>', unsafe_allow_html=True)

st.markdown("""
This application allows you to experiment with various spatial filtering techniques used in computer vision:
- **First-order derivatives**: Edge detection (Sobel, Prewitt, Roberts)
- **Second-order derivatives**: Fine detail detection (Laplacian, LoG)
- **Combinations**: Merge different filtering approaches
""")

# Sidebar for controls
with st.sidebar:
    st.header("🎛️ Filter Controls")
    
    # Image upload
    uploaded_file = st.file_uploader(
        "Upload an Image",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Supported formats: PNG, JPG, JPEG, BMP, TIFF"
    )
    
    if uploaded_file is not None:
        # Load and display original image
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        st.image(image, caption="Original Image", use_column_width=True)
        
        # Image preprocessing options
        st.subheader("1️⃣ Image Preprocessing")
        conversion_type = st.selectbox(
            "Convert Image:",
            ["Keep Original", "Grayscale", "Black & White"]
        )
        
        if conversion_type == "Black & White":
            bw_threshold = st.slider("B&W Threshold", 0, 255, 127)
        
        # Apply preprocessing
        if conversion_type == "Grayscale":
            processed_image = convert_to_grayscale(image_array)
        elif conversion_type == "Black & White":
            processed_image = convert_to_bw(image_array, bw_threshold)
        else:
            processed_image = convert_to_grayscale(image_array)  # Convert to grayscale for filtering
        
        # Filter selection
        st.subheader("2️⃣ Filter Selection")
        
        filter_mode = st.radio(
            "Choose Filter Mode:",
            ["Individual Filters", "Combined Filters"]
        )
        
        if filter_mode == "Individual Filters":
            filter_type = st.selectbox(
                "Select Filter Type:",
                ["First-Order Derivative", "Second-Order Derivative"]
            )
            
            if filter_type == "First-Order Derivative":
                first_order_filter = st.selectbox(
                    "First-Order Filter:",
                    ["Sobel X", "Sobel Y", "Sobel Combined", "Prewitt X", "Prewitt Y", 
                     "Roberts Cross-Gradient X", "Roberts Cross-Gradient Y"]
                )
            else:
                second_order_filter = st.selectbox(
                    "Second-Order Filter:",
                    ["Laplacian", "Laplacian of Gaussian (LoG)", 
                     "Custom Laplacian (4-connected)", "Custom Laplacian (8-connected)"]
                )
        
        else:  # Combined Filters
            first_order_filter = st.selectbox(
                "First-Order Filter:",
                ["Sobel X", "Sobel Y", "Sobel Combined", "Prewitt X", "Prewitt Y"]
            )
            
            second_order_filter = st.selectbox(
                "Second-Order Filter:",
                ["Laplacian", "Laplacian of Gaussian (LoG)", 
                 "Custom Laplacian (4-connected)", "Custom Laplacian (8-connected)"]
            )
            
            combination_method = st.selectbox(
                "Combination Method:",
                ["Add", "Multiply", "Maximum", "Subtract"]
            )
        
        # Process button
        process_button = st.button("🚀 Apply Filters", type="primary", key="sidebar_process")

# Main content area
if uploaded_file is not None and process_button:
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📊 Processing Summary")
        st.write(f"**Original Image Shape:** {image_array.shape}")
        st.write(f"**Preprocessing:** {conversion_type}")
        st.write(f"**Filter Mode:** {filter_mode}")
        
        if filter_mode == "Individual Filters":
            if filter_type == "First-Order Derivative":
                st.write(f"**Filter Applied:** {first_order_filter}")
            else:
                st.write(f"**Filter Applied:** {second_order_filter}")
        else:
            st.write(f"**First-Order:** {first_order_filter}")
            st.write(f"**Second-Order:** {second_order_filter}")
            st.write(f"**Combination:** {combination_method}")
    
    with col2:
        st.subheader("🖼️ Results")
        
        # Apply filters and create visualizations
        if filter_mode == "Individual Filters":
            if filter_type == "First-Order Derivative":
                filtered_image = apply_first_order_filter(processed_image, first_order_filter)
                images = [processed_image, filtered_image]
                titles = ["Preprocessed Image", f"{first_order_filter} Filter"]
            else:
                filtered_image = apply_second_order_filter(processed_image, second_order_filter)
                images = [processed_image, filtered_image]
                titles = ["Preprocessed Image", f"{second_order_filter} Filter"]
        else:
            combined_image, first_result, second_result = combine_filters(
                processed_image, first_order_filter, second_order_filter, combination_method
            )
            images = [processed_image, first_result, second_result, combined_image]
            titles = ["Preprocessed", f"{first_order_filter}", f"{second_order_filter}", 
                     f"Combined ({combination_method})"]
        
        # Create and display comparison plot
        fig = create_comparison_plot(images, titles)
        st.pyplot(fig)
        
        # Download processed images
        st.subheader("💾 Download Results")
        
        if filter_mode == "Individual Filters":
            # Create download button for the filtered image
            filtered_normalized = normalize_for_display(filtered_image)
            img_buffer = BytesIO()
            Image.fromarray(filtered_normalized).save(img_buffer, format='PNG')
            
            st.download_button(
                label="📥 Download Filtered Image",
                data=img_buffer.getvalue(),
                file_name=f"filtered_image_{filter_type.replace(' ', '_').lower()}.png",
                mime="image/png"
            )
        else:
            # Create download button for combined image
            combined_normalized = normalize_for_display(combined_image)
            img_buffer = BytesIO()
            Image.fromarray(combined_normalized).save(img_buffer, format='PNG')
            
            st.download_button(
                label="📥 Download Combined Image",
                data=img_buffer.getvalue(),
                file_name=f"combined_filter_{combination_method.lower()}.png",
                mime="image/png"
            )

# Information section
if uploaded_file is None:
    st.info("👆 Please upload an image using the sidebar to get started!")
    
    # Educational content
    st.subheader("📚 About Spatial Filtering")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **First-Order Derivatives (Edge Detection):**
        - **Sobel**: Emphasizes edges in X or Y direction
        - **Prewitt**: Similar to Sobel but with different kernel weights
        - **Roberts**: Simple cross-gradient operator
        
        **Applications:**
        - Object boundary detection
        - Feature extraction
        - Image segmentation
        """)
    
    with col2:
        st.markdown("""
        **Second-Order Derivatives (Detail Enhancement):**
        - **Laplacian**: Detects rapid intensity changes
        - **LoG**: Combines Gaussian smoothing with Laplacian
        - **Custom Kernels**: Different connectivity patterns
        
        **Applications:**
        - Fine detail enhancement
        - Noise amplification
        - Zero-crossing edge detection
        """)

# Footer
st.markdown("---")
st.markdown(
    "Made with ❤️ using Streamlit | "
    "Perfect for Computer Vision Spatial Filtering Assignments"
)

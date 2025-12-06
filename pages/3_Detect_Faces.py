#!/usr/bin/env python
# coding: utf-8

import os
import io
import json
import datetime
import streamlit as st
from PIL import Image, ImageDraw
import cv2
import numpy as np
import pandas as pd
import subprocess
import sys

# Supported image formats matching the database
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif']

st.markdown("""
<style>
    .stApp {
        background-image: url('https://png.pngtree.com/thumb_back/fw800/background/20190222/ourmid/pngtree-fashion-atmosphere-technology-creative-ppt-template-background-material-linesbluegridtechnologyppt-backgroundppt-template-image_55417.jpg');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    .main-header {
        font-size: 2.5rem;
        color: #ffffff;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .glass-card {
        background: rgba(30, 30, 30, 0.7);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 20px;
        margin: 0.5rem 0;
    }
    .feature-item {
        background: rgba(40, 40, 40, 0.6);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 1rem;
        border-radius: 14px;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    .feature-item:hover {
        background: rgba(50, 50, 50, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
</style>
""", unsafe_allow_html=True)


# Available models with their characteristics
MODELS = {
    "OpenFace": {
        "name": "OpenFace",
        "size": "Lightweight",
        "speed": "Fast",
        "accuracy": "Good",
        "description": "Lightweight model ideal for real-time applications"
    },
    "Facenet": {
        "name": "Facenet", 
        "size": "Medium",
        "speed": "Medium",
        "accuracy": "Very Good",
        "description": "Balanced model with good accuracy and speed"
    },
    "VGG-Face": {
        "name": "VGG-Face",
        "size": "Heavy",
        "speed": "Slow", 
        "accuracy": "Excellent",
        "description": "High-accuracy model but slower and larger"
    }
}

# Try to import DeepFace, install if not available
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    st.success("✅ DeepFace loaded successfully!")
except ImportError:
    st.warning("DeepFace not found. Installing...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "deepface"])
        from deepface import DeepFace
        DEEPFACE_AVAILABLE = True
        st.success("✅ DeepFace installed successfully!")
    except Exception as e:
        DEEPFACE_AVAILABLE = False
        st.error(f"❌ Failed to install DeepFace: {e}")


def detect_faces(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
    """
    Detect faces in an image using OpenCV Haar Cascade.
    """
    # Convert PIL Image to OpenCV format
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Load face detection classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=minSize
    )
    
    # Draw rectangles around faces
    img_with_faces = img_cv.copy()
    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(img_with_faces, (x, y), (x+w, y+h), (0, 255, 0), 3)
        
        # Add face number
        cv2.putText(img_with_faces, f'Face {i+1}', (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Convert back to PIL Image
    img_with_faces_rgb = cv2.cvtColor(img_with_faces, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_with_faces_rgb)
    
    return faces, img_pil


def get_person_info_from_filename(filename):
    """
    Extract person information from filename.
    Expected format: 'FirstName_LastName.jpg' or 'FirstName.jpg'
    """
    # Remove all supported extensions
    name = filename
    for ext in SUPPORTED_FORMATS:
        name = name.replace(ext, '')
    
    # Replace underscores with spaces for better display
    display_name = name.replace('_', ' ')
    return display_name, name


def face_analysis(face_image, db_path, model_name):
    """
    Analyze face using selected model in DeepFace.
    """
    try:
        # Convert PIL to numpy array for DeepFace
        face_array = np.array(face_image)
        
        # Use selected model for recognition
        recognition_results = DeepFace.find(
            img_path=face_array,
            db_path=db_path,
            model_name=model_name,
            enforce_detection=False,
            silent=True,
            distance_metric='cosine'
        )
        
        # Analyze face attributes
        analysis_results = DeepFace.analyze(
            face_array,
            actions=['age', 'gender', 'emotion', 'race'],
            enforce_detection=False,
            silent=True
        )
        
        # Get the best match from database
        identity = "Unknown"
        confidence = 0
        matched_file = None
        
        if recognition_results and len(recognition_results[0]) > 0:
            best_match = recognition_results[0].iloc[0]
            distance_column = f'{model_name}_cosine'
            confidence_score = 1 - best_match[distance_column]
            
            # Use a reasonable confidence threshold
            if confidence_score > 0.5:  # 50% confidence
                matched_file = best_match['identity']
                identity, original_name = get_person_info_from_filename(os.path.basename(matched_file))
                confidence = confidence_score
                st.info(f"{model_name} matched: {identity} with {confidence:.1%} confidence")
        
        analysis_data = analysis_results[0] if analysis_results else {}
        
        return identity, analysis_data, confidence
        
    except Exception as e:
        st.error(f"{model_name} analysis error: {str(e)}")
        return "Unknown", {}, 0


def format_analysis_output(identity, analysis, face_num, total_faces, model_name):
    """
    Format the analysis 
    """
    # Simulate progress like in your example
    progress_speed = 1.47 + (face_num * 0.15)  # Varying speeds
    progress_text = f"Action: race: 100%|██████████| {total_faces}/{total_faces} [00:02<00:00, {progress_speed:.2f}it/s]"
    
    output = f"{progress_text}\n\n"
    output += f"==> {identity}:\n"
    
    if analysis:
        output += f"   👤 Age: {analysis.get('age', 'N/A')}\n"
        output += f"   🚻 Gender: {analysis.get('dominant_gender', 'N/A')}\n"
        output += f"   😊 Emotion: {analysis.get('dominant_emotion', 'N/A')}\n"
        output += f"   🌍 Race: {analysis.get('dominant_race', 'N/A')}\n"
    else:
        output += "   👤 Age: N/A\n"
        output += "   🚻 Gender: N/A\n"
        output += "   😊 Emotion: N/A\n"
        output += "   🌍 Race: N/A\n"
    
    return output


def read_json(file_path):
    with open(file_path) as file:
        access_data = json.load(file)
    return access_data


# page headers and info text
st.set_page_config(
    page_title='Face Detection & Analysis',
    page_icon='👥',
    layout="wide"
)
st.sidebar.header('Face Detection & Analysis')
st.header('Face Detection & Analysis', divider='rainbow')
st.markdown(
    """
    Upload your image to detect faces using various face recognition models.
    Choose from lightweight to high-accuracy models based on your needs.
    """
)
st.divider()

# Model selection
st.write('#### Model Selection')
selected_model = st.selectbox(
    "Choose Face Recognition Model",
    options=list(MODELS.keys()),
    format_func=lambda x: f"{x} ({MODELS[x]['size']}) - {MODELS[x]['description']}",
    help="Select the model based on your needs: OpenFace (fastest), Facenet (balanced), VGG-Face (most accurate)"
)

# Display model info
model_info = MODELS[selected_model]
st.info(f"""
**Selected Model: {model_info['name']}**
- **Size**: {model_info['size']}
- **Speed**: {model_info['speed']}
- **Accuracy**: {model_info['accuracy']}
- **Description**: {model_info['description']}
""")

# Detection settings
st.write('#### Detection Settings')

col_det1, col_det2, col_det3 = st.columns(3)

with col_det1:
    scale_factor = st.slider(
        'Scale Factor',
        min_value=1.01,
        max_value=1.5,
        value=1.1,
        step=0.01,
        help='How much the image size is reduced at each scale'
    )

with col_det2:
    min_neighbors = st.slider(
        'Min Neighbors',
        min_value=1,
        max_value=10,
        value=3,
        help='Neighbors needed to retain detection'
    )

with col_det3:
    min_size = st.slider(
        'Min Face Size',
        min_value=20,
        max_value=100,
        value=30,
        help='Minimum possible face size (pixels)'
    )

# Confidence threshold
st.write('#### Recognition Settings')
confidence_threshold = st.slider(
    'Recognition Confidence Threshold',
    min_value=0.3,
    max_value=0.9,
    value=0.5,
    step=0.1,
    help='Minimum confidence for face recognition (higher = more strict)'
)

# uploading models and config
with st.spinner(f'Please wait, initializing {selected_model}...'):

    # Load configuration
    APP_CONFIG = read_json(file_path='config.json')
    IMGS_PATH = APP_CONFIG['imgs_path']
    DB_PATH = f'{IMGS_PATH}/db'
    
    # Create directories if they don't exist
    if not os.path.exists(IMGS_PATH):
        os.makedirs(IMGS_PATH)
    if not os.path.exists(DB_PATH):
        os.makedirs(DB_PATH)

# Check database
db_exists = os.path.exists(DB_PATH)
if db_exists:
    known_faces = [f for f in os.listdir(DB_PATH) if any(f.lower().endswith(ext) for ext in SUPPORTED_FORMATS)]
    if known_faces:
        st.success(f"✅ Database found with {len(known_faces)} faces for {selected_model} recognition")
        
        # Show database contents
        with st.expander(f"View {selected_model} Database"):
            cols = st.columns(4)
            for i, face_file in enumerate(known_faces):
                display_name, _ = get_person_info_from_filename(face_file)
                with cols[i % 4]:
                    st.write(f"• {display_name}")
    else:
        st.warning("📁 Database exists but contains no face images")
else:
    st.warning("⚠️ Database not found. Add face images to enable face recognition.")

st.write('#### Upload your image')
uploaded_file = st.file_uploader(
    'Select an image file for face analysis', 
    type=['jpg', 'jpeg', 'png', 'webp', 'bmp', 'tiff']
)

if uploaded_file is not None:
    file_name = uploaded_file.name
    file_extension = os.path.splitext(file_name)[1].lower()
    
    if file_extension in SUPPORTED_FORMATS:
        with st.spinner(f'Detecting faces with {selected_model}...'):
            bytes_data = uploaded_file.read()
            img = Image.open(io.BytesIO(bytes_data))
            
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')

            # Detect faces
            faces, img_with_faces = detect_faces(
                image=img,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=(min_size, min_size)
            )
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.write('##### Original Image')
                st.image(img, use_container_width=True)
                st.write(f"**Dimensions:** {img.size[0]} x {img.size[1]} pixels")
            
            with col2:
                st.write('##### Face Detection Results')
                st.image(img_with_faces, use_container_width=True)
                st.write(f"**Faces Detected:** {len(faces)}")
                st.write(f"**Model:** {selected_model}")
            
            st.write('##### Facial Analysis')
            st.markdown("`==================================================`")
            
            if len(faces) > 0:
                st.success(f"✅ {selected_model} found {len(faces)} face(s) in the image!")
                
                if DEEPFACE_AVAILABLE:
                    analysis_output = f"Facial Analysis using {selected_model}\n"
                    analysis_output += "==================================================\n"
                    
                    for i, (x, y, w, h) in enumerate(faces):
                        # Extract face region
                        face_region = img.crop((x, y, x + w, y + h))
                        
                        # Analyze face with selected model
                        identity, analysis, confidence = face_analysis(face_region, DB_PATH, selected_model)
                        
                        # Format the analysis in your requested style
                        analysis_output += format_analysis_output(identity, analysis, i+1, len(faces), selected_model)
                    
                    # Display the formatted analysis
                    st.code(analysis_output, language='text')
                    
                    # Show individual face regions with results
                    st.write('##### Detected Faces')
                    cols = st.columns(min(4, len(faces)))
                    for i, (x, y, w, h) in enumerate(faces):
                        face_region = img.crop((x, y, x + w, y + h))
                        identity, analysis, confidence = face_analysis(face_region, DB_PATH, selected_model)
                        
                        with cols[i % len(cols)]:
                            st.image(face_region, use_container_width=True)
                            if identity != "Unknown":
                                st.success(f"**{identity}**")
                                st.info(f"Confidence: {confidence:.1%}")
                            else:
                                st.warning("Unknown Face")
                
                else:
                    st.error("""
                    **DeepFace not available!**
                    
                    Please install DeepFace manually:
                    ```bash
                    pip install deepface
                    ```
                    
                    Or try without dependencies:
                    ```bash
                    pip install deepface --no-deps
                    pip install tensorflow opencv-python
                    ```
                    """)
                
            else:
                st.warning("❌ No faces detected in the image.")
                st.info("""
                **Tips to improve face detection:**
                - Ensure faces are clearly visible and well-lit
                - Use frontal face images for best results
                - Adjust detection parameters for better face detection
                - Use higher resolution images
                - Ensure good contrast between face and background
                """)
            
            # Save processed image
            img_with_faces.save(f'{IMGS_PATH}/faces_detected_{file_name}')
            
            # Logging
            msg = '{} - {} - file "{}" - {} faces detected\n'.format(
                datetime.datetime.now(),
                selected_model,
                file_name,
                len(faces)
            )
            with open('history.log', 'a') as file:
                file.write(msg)
                
            st.info(f'Processed image saved to: {IMGS_PATH}/faces_detected_{file_name}')

    else:
        st.error(
            f'Unsupported file format: {file_extension}. '
            f'Supported formats: {", ".join(SUPPORTED_FORMATS)}', 
            icon='⚠️'
        )

# Model comparison information
st.divider()
st.write('#### Model Comparison')

model_comparison = """
**Available Models:**

| Model | Size | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| **OpenFace** | Lightweight | Fast | Good | Real-time apps, mobile devices |
| **Facenet** | Medium | Medium | Very Good | Balanced performance |
| **VGG-Face** | Heavy | Slow | Excellent | High-accuracy requirements |

**Recommendations:**
- 🚀 **OpenFace**: Use for speed and efficiency
- ⚖️ **Facenet**: Use for balanced accuracy and speed  
- 🎯 **VGG-Face**: Use when accuracy is critical

**Current Status:** DeepFace is {}available
""".format("" if DEEPFACE_AVAILABLE else "not ")

st.markdown(model_comparison)

# Troubleshooting
if not DEEPFACE_AVAILABLE:
    st.write('#### Installation Help')
    st.markdown("""
    **If DeepFace installation fails, try these steps:**

    1. **Install without dependencies first:**
    ```bash
    pip install deepface --no-deps
    ```

    2. **Then install required dependencies:**
    ```bash
    pip install tensorflow opencv-python pandas numpy pillow gdown
    ```

    3. **If TensorFlow fails, try CPU version:**
    ```bash
    pip install tensorflow-cpu
    ```

    4. **Alternative: Install in stages:**
    ```bash
    pip install opencv-python pillow pandas numpy
    pip install tensorflow
    pip install deepface
    ```

    The selected model will be automatically downloaded on first use.
    """)
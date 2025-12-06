#!/usr/bin/env python
# coding: utf-8

import os
import io
import json
import datetime
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import pandas as pd
import numpy as np


def img_detect(model, img, conf_threshold=0.5):
    """
    Run YOLO inference on an image with newer version.
    """
    # Run inference
    results = model(img, conf=conf_threshold)
    
    # Process results
    result = results[0]
    names = model.names
    
    # Extract detections
    objs = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = box.conf.item()
            obj_name = names[class_id]
            objs.append({obj_name: confidence})
    
    # Plot results
    plotted_img = result.plot()  # Returns BGR numpy array
    img_rgb = Image.fromarray(plotted_img[..., ::-1])  # Convert to RGB PIL Image
    
    return objs, img_rgb, result


def read_json(file_path):
    with open(file_path) as file:
        access_data = json.load(file)
    return access_data


# page headers and info text
st.set_page_config(
    page_title='Object Detection',
    page_icon='🔍'
)
st.sidebar.header('Object Detection')
st.header('AI Object Detection', divider='rainbow')
st.markdown(
    f"""
    Upload your image to detect objects using YOLOv11 model.
    The system will identify and highlight objects with bounding boxes.
    """
)
st.divider()

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

# Confidence threshold slider
conf_threshold = st.slider(
    'Detection Confidence Threshold',
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.1,
    help='Adjust the minimum confidence level for object detection'
)

# uploading models
with st.spinner('Please wait, loading YOLOv11 model...'):

    # Using YOLOv11 model - newer version
    MODEL_DET_NAME = 'yolo11n.pt'  # or 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt'
    try:
        MODEL_DET = YOLO(MODEL_DET_NAME)
        st.success(f"✅ Loaded '{MODEL_DET_NAME}' successfully!")
    except Exception as e:
        st.error(f"Failed to load {MODEL_DET_NAME}. Trying YOLOv8 as fallback...")
        MODEL_DET_NAME = 'yolov8n.pt'
        MODEL_DET = YOLO(MODEL_DET_NAME)
        st.info(f"Using fallback model: {MODEL_DET_NAME}")

    APP_CONFIG = read_json(file_path='config.json')
    IMGS_PATH = APP_CONFIG['imgs_path']
    
    # create images directory if it doesn't exist
    if not os.path.exists(IMGS_PATH):
        os.makedirs(IMGS_PATH)

st.write('#### Upload your image')
uploaded_file = st.file_uploader('Select an image file (JPEG, PNG format)')
if uploaded_file is not None:
    file_name = uploaded_file.name
    if '.jpg' in file_name.lower() or '.png' in file_name.lower() or '.jpeg' in file_name.lower():
        with st.spinner('Detecting objects...'):
            bytes_data = uploaded_file.read()
            img = Image.open(io.BytesIO(bytes_data))

            # object detection model for uploaded image
            objs, img_det, result = img_detect(
                model=MODEL_DET, 
                img=img,
                conf_threshold=conf_threshold
            )
            
            # Filter objects based on confidence threshold
            filtered_objs = [list(x.keys())[0] for x in objs]
            objects_text = ' ,'.join(filtered_objs)
            
            # Display in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.write('##### Original Image')
                st.image(img, use_container_width=True)
            
            with col2:
                st.write('##### Detection Results')
                st.image(img_det, use_container_width=True)
            
            st.write('##### Detection Summary')
            if filtered_objs:
                st.success(f"**Found {len(filtered_objs)} object(s) with confidence ≥ {conf_threshold}**")
                
                # Display objects in a table with confidence scores
                st.write('**Detailed Results:**')
                obj_data = []
                for obj in objs:
                    for obj_name, confidence in obj.items():
                        obj_data.append({
                            'Object': obj_name,
                            'Confidence': f"{confidence:.2%}",
                            'Status': '✅ Detected' if confidence >= conf_threshold else '❌ Below threshold'
                        })
                
                if obj_data:
                    df = pd.DataFrame(obj_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # Show some statistics
                    detected_count = len([x for x in objs if list(x.values())[0] >= conf_threshold])
                    st.metric("Objects Detected", detected_count)
            else:
                st.warning(f"No objects detected with confidence ≥ {conf_threshold}")
                st.info("Try lowering the confidence threshold in the sidebar")
            
            # Save the processed image
            img_det.save(f'{IMGS_PATH}/detected_{file_name}')
            
            # logging
            msg = '{} - file "{}" - {} objects detected: "{}"\n'.format(
                datetime.datetime.now(),
                file_name,
                len(filtered_objs),
                objects_text
            )
            with open('history.log', 'a') as file:
                file.write(msg)
                
            st.info(f'Processed image saved to: {IMGS_PATH}/detected_{file_name}')

    else:
        st.error('Please upload a valid image file (JPEG or PNG format)', icon='⚠️')
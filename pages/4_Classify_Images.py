#!/usr/bin/env python
# coding: utf-8

import os
import io
import json
import datetime
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
from transformers import pipeline


def zeroshot(classifier, classes, img):
    """
    Perform zero-shot image classification.
    """
    scores = classifier(
        img,
        candidate_labels=classes
    )
    return scores


def read_json(file_path):
    with open(file_path) as file:
        access_data = json.load(file)
    return access_data


# page headers and info text
st.set_page_config(
    page_title='Image Classification',
    page_icon='📊'
)
st.sidebar.header('Image Classification')
st.header('AI Image Classification', divider='rainbow')
st.markdown(
    f"""
    Upload your image to classify it using zero-shot learning.
    The system will categorize your image based on predefined classes.
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
    'Classification Confidence Threshold',
    min_value=0.1,
    max_value=0.9,
    value=0.3,
    step=0.1,
    help='Adjust the minimum confidence level for classification'
)

# uploading models
with st.spinner('Please wait, loading classification model...'):

    # classification model and classes
    MODEL_ZERO_NAME = 'openai/clip-vit-base-patch16'
    try:
        CLASSIFIER_ZERO = pipeline('zero-shot-image-classification', model=MODEL_ZERO_NAME)
        st.success("✅ Loaded classification model successfully!")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    # Load configuration
    APP_CONFIG = read_json(file_path='config.json')
    CLASSES = APP_CONFIG['classes']
    DB_DICT = APP_CONFIG['db_dict']
    TH_OTHERS = APP_CONFIG['th_others']
    IMGS_PATH = APP_CONFIG['imgs_path']
    
    # Create directories for each category
    for k, v in DB_DICT.items():
        imgs_path = f'{IMGS_PATH}/{v.strip()}'
        if not os.path.exists(imgs_path):
            os.makedirs(imgs_path)
    imgs_path = f'{IMGS_PATH}/other'
    if not os.path.exists(imgs_path):
        os.makedirs(imgs_path)

st.write('#### Available Classes')
st.write(f"**{len(CLASSES)} categories available:**")
cols = st.columns(3)
for i, class_name in enumerate(CLASSES):
    with cols[i % 3]:
        st.write(f"• {class_name}")

st.divider()

st.write('#### Upload your image')
uploaded_file = st.file_uploader('Select an image file (JPEG, PNG format)')
if uploaded_file is not None:
    file_name = uploaded_file.name
    if '.jpg' in file_name.lower() or '.png' in file_name.lower() or '.jpeg' in file_name.lower():
        with st.spinner('Classifying image...'):
            bytes_data = uploaded_file.read()
            img = Image.open(io.BytesIO(bytes_data))

            # Classify image with zero-shot model
            scores = zeroshot(
                classifier=CLASSIFIER_ZERO, 
                classes=CLASSES, 
                img=img
            )
            
            # Find the best matching category
            max_score = sorted(scores, key=lambda x: x['score'])[-1]
            if max_score['score'] >= TH_OTHERS:
                category = max_score['label']
                save_path = DB_DICT.get(category, 'other')
                status = "✅ Confident Classification"
            else:
                category = 'unknown'
                save_path = 'other'
                status = "⚠️ Low Confidence - Saved as 'other'"
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.write('##### Your Image')
                st.image(img, use_container_width=True)
                st.write(f"**Classification:** {category}")
                st.write(f"**Confidence:** {max_score['score']:.2%}")
                st.write(f"**Status:** {status}")
            
            with col2:
                st.write('##### Classification Results')
                
                # Create dataframe for all scores
                df = pd.DataFrame(scores)
                df['score'] = df['score'].apply(lambda x: f"{x:.2%}")
                df = df.rename(columns={'label': 'Category', 'score': 'Confidence'})
                
                # Highlight the top result
                def highlight_top(row):
                    if row['Category'] == category:
                        return ['background-color: #2e7d32; color: white'] * len(row)
                    else:
                        return [''] * len(row)
                
                st.dataframe(
                    df.style.apply(highlight_top, axis=1),
                    use_container_width=True
                )
                
                # Bar chart visualization
                chart_df = pd.DataFrame(scores)
                chart_df = chart_df.set_index('label')
                st.bar_chart(chart_df)
            
            # Save the image to appropriate category
            img.save(f'{IMGS_PATH}/{save_path}/{file_name}')
            
            # logging
            msg = '{} - file "{}" classified as "{}" with confidence {:.2%}, saved to "{}"\n'.format(
                datetime.datetime.now(),
                file_name,
                category,
                max_score['score'],
                save_path
            )
            with open('history.log', 'a') as file:
                file.write(msg)
                
            st.info(f'Image saved to: {IMGS_PATH}/{save_path}/{file_name}')
            
            # Show storage information
            st.write('##### Storage Information')
            col_info1, col_info2, col_info3 = st.columns(3)
            
            with col_info1:
                total_files = sum([len(files) for r, d, files in os.walk(IMGS_PATH)])
                st.metric("Total Images Stored", total_files)
            
            with col_info2:
                category_files = len(os.listdir(f'{IMGS_PATH}/{save_path}'))
                st.metric(f"Images in {save_path}", category_files)
            
            with col_info3:
                st.metric("Classification Confidence", f"{max_score['score']:.2%}")

    else:
        st.error('Please upload a valid image file (JPEG or PNG format)', icon='⚠️')
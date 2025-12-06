#!/usr/bin/env python
# coding: utf-8

import os
import io
import json
import datetime
import streamlit as st
from PIL import Image
from transformers import (
    BlipProcessor, 
    BlipForConditionalGeneration
)


def img_caption(model, processor, img, text=None):
    """
    Uses BLIP model to caption image.
    """
    res = None
    if text:
        # conditional image captioning
        inputs = processor(img, text, return_tensors='pt')
    else:
        # unconditional image captioning
        inputs = processor(img, return_tensors='pt')
    out = model.generate(**inputs)
    res = processor.decode(out[0], skip_special_tokens=True)
    return res


def read_json(file_path):
    with open(file_path) as file:
        access_data = json.load(file)
    return access_data


# page headers and info text
st.set_page_config(
    page_title='Image Captioning',
    page_icon='📝'
)
st.sidebar.header('Image Captioning')
st.header('AI Image Captioning', divider='rainbow')
st.markdown(
    f"""
    Upload your image and get an AI-generated caption.
    You can also provide text for conditional captioning.
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

# uploading models
with st.spinner('Please wait, application is initializing...'):

    # caption model
    MODEL_CAP_NAME = 'Salesforce/blip-image-captioning-base'
    PROCESSOR_CAP = BlipProcessor.from_pretrained(MODEL_CAP_NAME)
    MODEL_CAP = BlipForConditionalGeneration.from_pretrained(MODEL_CAP_NAME)

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
        # input text for conditional image captioning
        text = st.text_input(
            'Input text for conditional image captioning (optional)',
            '',
            help='Provide context for the caption generation'
        )
        with st.spinner('Generating caption...'):
            bytes_data = uploaded_file.read()
            img = Image.open(io.BytesIO(bytes_data))

            # image caption model for uploaded image
            caption = img_caption(
                model=MODEL_CAP, 
                processor=PROCESSOR_CAP, 
                img=img, 
                text=text
            )
            
            st.write('##### Your Image')
            st.image(img, use_container_width=True)
            
            st.write('##### Generated Caption')
            st.success(caption)
            
            # Save the image
            img.save(f'{IMGS_PATH}/{file_name}')
            
            # logging
            msg = '{} - file "{}" got caption "{}"\n'.format(
                datetime.datetime.now(),
                file_name,
                caption
            )
            with open('history.log', 'a') as file:
                file.write(msg)
                
            st.info(f'Image saved to: {IMGS_PATH}/{file_name}')

    else:
        st.error('Please upload a valid image file (JPEG or PNG format)', icon='⚠️')
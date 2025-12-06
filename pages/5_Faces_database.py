#!/usr/bin/env python
# coding: utf-8

import os
import io
import json
import datetime
import streamlit as st
from PIL import Image


def read_json(file_path):
    with open(file_path) as file:
        access_data = json.load(file)
    return access_data


# page headers and info text
st.set_page_config(
    page_title='Faces database', 
    page_icon=':microscope:'
)
st.sidebar.header('Friends database')
st.header('Database contains images of the friends', divider='rainbow')

st.markdown(
    f"""
    Here you can see all images of the friends 
    and update database with new people.
    """
)
st.divider()

N_COLS = 3
APP_CONFIG = read_json(file_path='config.json')
IMGS_PATH = APP_CONFIG['imgs_path']
DB_PATH = f'{IMGS_PATH}/db'

# Supported image formats
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif']


@st.cache_data
def imgs_data(path):
    img_files = [
        f for f in os.listdir(path) 
        if any(f.lower().endswith(ext) for ext in SUPPORTED_FORMATS)
    ]
    data = []
    for f in img_files:
        # Get file extension
        _, ext = os.path.splitext(f)
        img_data = {
            'img_name': f.replace(f'{path}/', '').replace(ext, ''),
            'img_path': f'{path}/{f}',
            'file_ext': ext
        }
        data.append(img_data)
    return data


# display a gallery of images
st.write('#### Gallery')
n_cols = st.slider('Width:', min_value=1, max_value=5, value=N_COLS)
imgs_list = imgs_data(path=DB_PATH)

if imgs_list:
    cols = st.columns(n_cols)
    for i, img in enumerate(imgs_list):
        with cols[i % n_cols]:
            st.image(
                img['img_path'], 
                caption=f"{img['img_name']} ({img['file_ext']})", 
                use_container_width=True
            )
    st.write(f"**Total images in database:** {len(imgs_list)}")
else:
    st.info("No images found in the database. Upload some images to get started.")

st.divider()

# upload more images
st.write('#### Upload your image')
uploaded_file = st.file_uploader(
    'Select an image file', 
    type=['jpg', 'jpeg', 'png', 'webp', 'bmp', 'tiff']
)

if uploaded_file is not None:
    file_name = uploaded_file.name
    file_extension = os.path.splitext(file_name)[1].lower()
    
    if file_extension in SUPPORTED_FORMATS:
        bytes_data = uploaded_file.read()
        try:
            img = Image.open(io.BytesIO(bytes_data))
            
            # Convert to RGB if necessary (for PNG with transparency)
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            
            # Save the image
            img.save(f'{DB_PATH}/{file_name}')
            
            st.success(f"✅ Image '{file_name}' successfully uploaded to database!")
            
            # Clear cache to refresh the gallery
            st.cache_data.clear()
            
            # logging
            msg = '{} - file "{}" saved in database\n'.format(
                datetime.datetime.now(),
                file_name
            )
            with open('history.log', 'a') as file:
                file.write(msg)
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}", icon='⚠️')
    else:
        st.error(
            f'Unsupported file format: {file_extension}. '
            f'Supported formats: {", ".join(SUPPORTED_FORMATS)}', 
            icon='⚠️'
        )

st.divider()

# Database information
st.write('#### Database Information')
st.markdown(f"""
**Location:** `{DB_PATH}`

**Supported Formats:** {", ".join(SUPPORTED_FORMATS)}

**Usage Tips:**
- Use clear, well-lit face images for best results
- Name files clearly (e.g., `Firstname_Lastname.jpg`)
- Recommended size: 300x300 pixels or larger
- For best face recognition, use frontal face images
""")
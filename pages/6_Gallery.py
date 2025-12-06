#!/usr/bin/env python
# coding: utf-8

import os
import json
import streamlit as st
from PIL import Image


def read_json(file_path):
    with open(file_path) as file:
        return json.load(file)


# Streamlit layout
st.set_page_config(
    page_title='Gallery',
    page_icon=':microscope:'
)

st.sidebar.header('Images gallery')
st.header('Database contains images by categories', divider='rainbow')

st.markdown(
    """
    Here you can see all the images
    classified by the categories with 
    help of AI-assistant.
    """
)
st.divider()


# Load configuration
APP_CONFIG = read_json('config.json')
IMGS_PATH = APP_CONFIG['imgs_path']

# Make safe copies
CLASSES = APP_CONFIG['classes'][:]
DB_DICT = APP_CONFIG['db_dict'].copy()

# Add fallback category
CLASSES.append('other')
DB_DICT['other'] = 'other'

# Keep only classes that exist in mapping
# CLASSES = [c for c in CLASSES if c in DB_DICT]


@st.cache_data
def imgs_data(path, classes, db_dict):
    data = {}

    # Allowed formats
    VALID_EXTENSIONS = (
        '.jpg', '.jpeg', '.png', '.gif', '.bmp',
        '.tiff', '.tif', '.webp', '.heic', '.heif'
    )

    for c in classes:

        if c not in db_dict:
            data[c] = []
            continue

        class_folder = os.path.join(path, db_dict[c])

        if not os.path.isdir(class_folder):
            data[c] = []
            continue

        # List all valid image files
        img_files = [
            f for f in os.listdir(class_folder)
            if f.lower().endswith(VALID_EXTENSIONS)
        ]

        data[c] = [{
            'img_name': f,
            'img_path': os.path.join(class_folder, f)
        } for f in img_files]

    return data


# Gallery display
n_cols = st.slider('Width:', min_value=1, max_value=5, value=3)
imgs_list = imgs_data(IMGS_PATH, CLASSES, DB_DICT)

for c in CLASSES:
    st.write(f'#### Gallery of images from category – **{DB_DICT[c]}** –')

    cols = st.columns(n_cols)

    for i, img in enumerate(imgs_list[c]):
        with cols[i % n_cols]:
            st.image(
                img['img_path'],
                caption=img['img_name'],
                use_container_width=True
            )

    st.divider()

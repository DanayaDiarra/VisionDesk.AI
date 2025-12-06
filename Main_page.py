#!/usr/bin/env python
# coding: utf-8

import os
import json
import streamlit as st


def read_json(file_path):
    with open(file_path) as file:
        access_data = json.load(file)
    return access_data


st.set_page_config(
    page_title='VisionDesk.AI',
    page_icon="🤖",
    layout="wide",
)

# Apple-style blur effects with image
st.markdown("""
<style>
    .stApp {
        background-image: url('https://png.pngtree.com/thumb_back/fw800/background/20190222/ourmid/pngtree-fashion-atmosphere-technology-creative-ppt-template-background-material-linesbluegridtechnologyppt-backgroundppt-template-image_55417.jpg');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    /* Main content area with blur effect */
    .main .block-container {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(25px) !important;
        -webkit-backdrop-filter: blur(25px) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 24px !important;
        margin: 2rem auto !important;
        padding: 2rem !important;
        max-width: 1200px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.36) !important;
    }
    
    .main-header {
        font-size: 3rem;
        color: #ffffff;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        background: linear-gradient(135deg, #ffffff 0%, #e0e0e0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.9);
        text-align: center;
        margin-bottom: 3rem;
        text-shadow: 0 1px 4px rgba(0, 0, 0, 0.2);
        font-weight: 300;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.15) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        background: rgba(255, 255, 255, 0.2) !important;
        border: 1px solid rgba(255, 255, 255, 0.35) !important;
        transform: translateY(-2px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.3);
    }
    
    .feature-item {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 1.2rem;
        border-radius: 16px;
        margin: 0.8rem 0;
        transition: all 0.3s ease;
        color: white;
        font-weight: 500;
    }
    
    .feature-item:hover {
        background: rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.25);
        transform: translateX(8px);
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.15);
    }
    
    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(25px) !important;
        -webkit-backdrop-filter: blur(25px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    .stButton button {
        background: rgba(255, 255, 255, 0.15) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton button:hover {
        background: rgba(255, 255, 255, 0.25) !important;
        border: 1px solid rgba(255, 255, 255, 0.4) !important;
        transform: translateY(-1px);
    }
    
    .footer-glass {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 2rem;
        color: white;
    }
    
    .footer-glass a {
        color: #e0e0ff !important;
        text-decoration: none;
        transition: color 0.3s ease;
    }
    
    .footer-glass a:hover {
        color: #ffffff !important;
        text-decoration: underline;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.3);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.5);
    }
    
    /* Text styling */
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    p, li, div {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Remove default Streamlit styling */
    .st-emotion-cache-1jicfl2 {
        padding: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Header with  styling
st.markdown('<h1 class="main-header">VisionDesk.AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Image Analysis Platform</p>', unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="background: rgba(255,255,255,0.1); backdrop-filter: blur(20px); border-radius: 12px; padding: 1rem; border: 1px solid rgba(255,255,255,0.2);">
<h3 style="color: white; margin-bottom: 1rem;">🎯Navigation</h3>
</div>
""", unsafe_allow_html=True)

st.sidebar.success('Select a task from above')

# Main content with glass cards
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h2 style="color: white; margin-bottom: 1.5rem;">🛠️ Available Tasks</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="feature-item">
    📝 Image Captioning
    </div>
    <div class="feature-item">
    🔍 Object Detection
    </div>
    <div class="feature-item">
    📊 Image Classification
    </div>
    <div class="feature-item">
    📖 OCR (Text Extraction)
    </div>
    <div class="feature-item">
    👥 Face Detection
    </div>
    <div class="feature-item">
    ⬆️ Upload & Extract Text
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h2 style="color: white; margin-bottom: 1.5rem;">♾️ How it Works</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div style="color: white; line-height: 2.5;">
    <div style="display: flex; align-items: center; margin: 1rem 0;">
        <span style="background: rgba(255,255,255,0.2); padding: 0.5rem; border-radius: 8px; margin-right: 1rem;">1</span>
        <strong>Select Task</strong> → Choose from sidebar
    </div>
    <div style="display: flex; align-items: center; margin: 1rem 0;">
        <span style="background: rgba(255,255,255,0.2); padding: 0.5rem; border-radius: 8px; margin-right: 1rem;">2</span>
        <strong>Upload Image</strong> → Your file or existing
    </div>
    <div style="display: flex; align-items: center; margin: 1rem 0;">
        <span style="background: rgba(255,255,255,0.2); padding: 0.5rem; border-radius: 8px; margin-right: 1rem;">3</span>
        <strong>AI Processing</strong> → Models analyze content
    </div>
    <div style="display: flex; align-items: center; margin: 1rem 0;">
        <span style="background: rgba(255,255,255,0.2); padding: 0.5rem; border-radius: 8px; margin-right: 1rem;">4</span>
        <strong>View Results</strong> → Instant insights
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick action buttons
    st.markdown("""
    <div style="margin-top: 2rem;">
        <h4 style="color: white; margin-bottom: 1rem;">Quick Actions</h4>
    </div>
    """, unsafe_allow_html=True)
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button('📸 Upload Image', use_container_width=True):
            st.info("Upload feature coming soon!")
    with col_btn2:
        if st.button('🚀 Start Analysis', use_container_width=True):
            st.info("Select a task to begin analysis!")
    
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# Stats section with glass effect
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<h2 style="color: white; text-align: center; margin-bottom: 2rem;">📈 Platform Statistics</h2>', unsafe_allow_html=True)

col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

with col_stat1:
    st.markdown("""
    <div style="text-align: center; color: white;">
        <h1 style="font-size: 2.5rem; margin: 0; color: #4FC3F7;">1.2K</h1>
        <p style="margin: 0; opacity: 0.8;">Images Processed</p>
    </div>
    """, unsafe_allow_html=True)

with col_stat2:
    st.markdown("""
    <div style="text-align: center; color: white;">
        <h1 style="font-size: 2.5rem; margin: 0; color: #81C784;">98%</h1>
        <p style="margin: 0; opacity: 0.8;">Accuracy Rate</p>
    </div>
    """, unsafe_allow_html=True)

with col_stat3:
    st.markdown("""
    <div style="text-align: center; color: white;">
        <h1 style="font-size: 2.5rem; margin: 0; color: #FFB74D;">24/7</h1>
        <p style="margin: 0; opacity: 0.8;">Uptime</p>
    </div>
    """, unsafe_allow_html=True)

with col_stat4:
    st.markdown("""
    <div style="text-align: center; color: white;">
        <h1 style="font-size: 2.5rem; margin: 0; color: #BA68C8;">5+</h1>
        <p style="margin: 0; opacity: 0.8;">AI Models</p>
    </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Initialize app
APP_CONFIG = read_json(file_path='config.json')
IMGS_PATH = APP_CONFIG['imgs_path']
DB_PATH = f'{IMGS_PATH}/db'

for path in [IMGS_PATH, DB_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)

# Footer with  glass effect
st.markdown("""
<div class="footer-glass">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <h4 style="color: white; margin-bottom: 1rem;">Contact & Support</h4>
            <ul style="list-style: none; padding: 0; margin: 0;">
                <li style="margin: 0.5rem 0;">📧 <a href="mailto:support@visiondesk.ai">support@visiondesk.ai</a></li>
                <li style="margin: 0.5rem 0;">💼 <a href="https://www.linkedin.com/in/diadalknd">LinkedIn</a></li>
                <li style="margin: 0.5rem 0;">📱 <a href="https://t.me/MarkusD01">Telegram</a></li>
            </ul>
        </div>
        <div style="text-align: right;">
            <p style="margin: 0; opacity: 0.8;">Powered by Advanced AI Models</p>
            <p style="margin: 0; opacity: 0.6; font-size: 0.9rem;">© 2024 VisionDesk.AI</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
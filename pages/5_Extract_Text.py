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
import subprocess
import sys

# Apple-style blur effects with background image
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

# Try Tesseract first (faster), with fallback to EasyOCR only if needed
OCR_ENGINE = None

try:
    import pytesseract
    # Test if Tesseract works with basic English
    try:
        # Create a simple test image
        test_img = Image.new('RGB', (100, 100), color='white')
        pytesseract.image_to_string(test_img, lang='eng')
        OCR_ENGINE = 'tesseract'
        st.success("✅ Using Tesseract OCR (Fast)")
    except:
        st.warning("Tesseract installed but not working properly")
        raise ImportError("Tesseract not functional")
        
except ImportError:
    st.warning("Tesseract not available, trying EasyOCR...")
    try:
        import easyocr
        OCR_ENGINE = 'easyocr'
        st.info("Using EasyOCR (Slower but reliable)")
    except ImportError:
        st.error("Installing OCR engines...")
        # Try to install Tesseract first
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pytesseract"])
            import pytesseract
            OCR_ENGINE = 'tesseract'
            st.success("✅ Tesseract installed successfully!")
        except:
            # Fallback to EasyOCR
            subprocess.check_call([sys.executable, "-m", "pip", "install", "easyocr"])
            import easyocr
            OCR_ENGINE = 'easyocr'
            st.info("Using EasyOCR as fallback")

try:
    from pdf2image import convert_from_bytes
    import cv2
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pdf2image", "opencv-python"])
    from pdf2image import convert_from_bytes
    import cv2


def extract_text_from_image(img, language='eng'):
    """
    Extract text from image using available OCR engine with smart fallbacks.
    """
    if OCR_ENGINE == 'tesseract':
        try:
            # Try the requested language first
            text = pytesseract.image_to_string(img, lang=language)
            if text.strip():
                return text.strip()
            else:
                # If no text found, try English as fallback
                text = pytesseract.image_to_string(img, lang='eng')
                return text.strip()
        except Exception as e:
            if 'rus' in str(e) or 'ru' in str(e):
                # Russian language not available, fallback to English
                st.warning(f"Russian language pack not available. Using English instead.")
                text = pytesseract.image_to_string(img, lang='eng')
                return text.strip()
            else:
                st.error(f"Tesseract error: {str(e)}")
                return ""
    else:
        # EasyOCR fallback
        try:
            # Map language codes for EasyOCR
            lang_map = {'eng': 'en', 'rus': 'ru', 'fra': 'fr', 'deu': 'de', 'spa': 'es', 'ita': 'it'}
            easy_lang = lang_map.get(language, 'en')
            reader = easyocr.Reader([easy_lang])
            results = reader.readtext(np.array(img))
            text = ' '.join([result[1] for result in results])
            return text.strip()
        except Exception as e:
            st.error(f"EasyOCR error: {str(e)}")
            return ""


def extract_text_from_pdf(pdf_bytes, language='eng'):
    """
    Extract text from PDF by converting to images and then using OCR.
    """
    all_text = []
    
    try:
        # Convert PDF to images - use raw bytes
        images = convert_from_bytes(pdf_bytes)
        
        progress_bar = st.progress(0)
        total_pages = len(images)
        
        for i, image in enumerate(images):
            # Update progress
            progress_bar.progress((i + 1) / total_pages)
            
            # Extract text from each page
            text = extract_text_from_image(image, language)
            if text.strip():
                all_text.append({
                    'page': i + 1,
                    'text': text.strip()
                })
        
        progress_bar.empty()
        
    except Exception as e:
        st.error(f"PDF processing error: {str(e)}")
    
    return all_text


def get_text_regions(img, language='eng'):
    """
    Detect text regions in the image.
    """
    try:
        if OCR_ENGINE == 'tesseract':
            # Tesseract text detection
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            img_with_boxes = np.array(img).copy()
            
            n_boxes = len(data['level'])
            for i in range(n_boxes):
                if int(data['conf'][i]) > 30:
                    (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                    cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(img_with_boxes, f"{data['text'][i]}", 
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # EasyOCR text detection
            lang_map = {'eng': 'en', 'rus': 'ru', 'fra': 'fr', 'deu': 'de', 'spa': 'es', 'ita': 'it'}
            easy_lang = lang_map.get(language, 'en')
            reader = easyocr.Reader([easy_lang])
            results = reader.readtext(np.array(img))
            img_with_boxes = np.array(img).copy()
            
            for (bbox, text, confidence) in results:
                if confidence > 0.3:
                    (top_left, top_right, bottom_right, bottom_left) = bbox
                    top_left = tuple(map(int, top_left))
                    bottom_right = tuple(map(int, bottom_right))
                    cv2.rectangle(img_with_boxes, top_left, bottom_right, (0, 255, 0), 2)
                    cv2.putText(img_with_boxes, f"{text} ({confidence:.2f})", 
                               (top_left[0], top_left[1] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        img_with_boxes_pil = Image.fromarray(img_with_boxes)
        return data, img_with_boxes_pil
    except Exception as e:
        st.error(f"Text detection error: {str(e)}")
        return {}, img


def summarize_text(text, max_length=150, min_length=30):
    """
    Generate a summary of the extracted text.
    """
    try:
        # Simple extractive summarization
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) > 3:
            # Take first few meaningful sentences as summary
            summary = '. '.join(sentences[:3]) + '.'
            return summary
        else:
            return text
    except Exception as e:
        return f"Summary: {text[:200]}..." if len(text) > 200 else text


def analyze_text(text):
    """
    Perform comprehensive text analysis.
    """
    lines = text.split('\n')
    words = text.split()
    characters = len(text)
    
    # Basic statistics
    stats = {
        'lines': len(lines),
        'words': len(words),
        'characters': characters,
        'sentences': len([s for s in text.split('.') if s.strip()]),
        'paragraphs': len([p for p in text.split('\n\n') if p.strip()]),
        'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
        'avg_words_per_line': len(words) / len(lines) if lines else 0
    }
    
    # Word frequency
    from collections import Counter
    word_freq = Counter(words)
    common_words = word_freq.most_common(15)
    
    # Text complexity
    unique_words = len(set(words))
    stats['vocabulary_diversity'] = unique_words / len(words) if words else 0
    
    return stats, common_words


def read_json(file_path):
    with open(file_path) as file:
        access_data = json.load(file)
    return access_data


# page headers and info text
st.set_page_config(
    page_title='Text Extraction',
    page_icon='📖',
    layout="wide",
)

# Header with glass effect
st.markdown('<h1 class="main-header">Text Extraction & Summarization</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #888; margin-bottom: 2rem;">AI-Powered OCR with Text Analysis</p>', unsafe_allow_html=True)

st.sidebar.header('Text Extraction')

# Main content in glass cards
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📖 About OCR")
    st.write(f"""
    **Using:** {OCR_ENGINE.upper()} OCR Engine
    **Status:** {'Fast & Optimized' if OCR_ENGINE == 'tesseract' else '⚠️ Slower but reliable'}
    
    **Supported Formats:**
    - Images: JPG, PNG, TIFF, BMP
    - Documents: PDF
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("♾️ How it Works")
    st.write("""
    ⬇️ **Upload** - Image or PDF file  
    ⬇️ **Select** - Language and options  
    ⬇️ **Extract** - Text using OCR  
    ✅ **Analyze** - Get insights & summary
    """)
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# OCR settings in glass card
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("⚙️ OCR Settings")

# Language options - only show available languages
if OCR_ENGINE == 'tesseract':
    # Test which languages are available
    available_languages = ['eng']  # English is always available
    
    # Test Russian
    try:
        test_img = Image.new('RGB', (100, 100), color='white')
        pytesseract.image_to_string(test_img, lang='rus')
        available_languages.append('rus')
    except:
        st.info("ℹ️ Russian language pack not installed. Using English fallback.")
    
    # Add other languages that are commonly available
    other_langs = ['fra', 'deu', 'spa', 'ita']
    for lang in other_langs:
        try:
            pytesseract.image_to_string(test_img, lang=lang)
            available_languages.append(lang)
        except:
            pass
    
    language_names = {
        'eng': 'English',
        'rus': 'Russian', 
        'fra': 'French',
        'deu': 'German',
        'spa': 'Spanish',
        'ita': 'Italian'
    }
    
    # Only show available languages
    available_options = [(code, language_names[code]) for code in available_languages if code in language_names]
    
else:
    # EasyOCR languages
    available_options = [
        ('en', 'English'),
        ('ru', 'Russian'),
        ('fr', 'French'),
        ('de', 'German'),
        ('es', 'Spanish'),
        ('it', 'Italian')
    ]

col_set1, col_set2 = st.columns(2)

with col_set1:
    selected_language = st.selectbox(
        'OCR Language',
        [opt[1] for opt in available_options],
        index=0
    )
    # Get language code
    ocr_language = [opt[0] for opt in available_options if opt[1] == selected_language][0]

with col_set2:
    show_bounding_boxes = st.checkbox('Show Text Detection Boxes', value=True)
    generate_summary = st.checkbox('Generate Summary', value=True)

# Show language availability info
if OCR_ENGINE == 'tesseract' and 'rus' not in available_languages:
    st.warning("""
    **Language Packs:** For additional languages, install Tesseract language packs:
    ```bash
    sudo apt-get install tesseract-ocr-[lang]
    # Example: sudo apt-get install tesseract-ocr-rus
    ```
    """)

st.markdown('</div>', unsafe_allow_html=True)

# uploading config
with st.spinner(f'Initializing {OCR_ENGINE.upper()} OCR engine...'):
    APP_CONFIG = read_json(file_path='config.json')
    IMGS_PATH = APP_CONFIG['imgs_path']
    
    if not os.path.exists(IMGS_PATH):
        os.makedirs(IMGS_PATH)

# File upload in glass card
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("📤 Upload Your File")
uploaded_file = st.file_uploader(
    'Select an image or PDF file', 
    type=['jpg', 'jpeg', 'png', 'pdf', 'tiff', 'bmp'],
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    file_name = uploaded_file.name
    file_extension = file_name.lower().split('.')[-1]
    
    # Read the file bytes once and store them
    bytes_data = uploaded_file.getvalue()  # Use getvalue() instead of read()
    
    if file_extension in ['jpg', 'jpeg', 'png', 'tiff', 'bmp']:
        with st.spinner(f'Extracting text from image...'):
            img = Image.open(io.BytesIO(bytes_data))
            
            # Display in glass cards
            col_img1, col_img2 = st.columns(2)
            
            with col_img1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.write('##### Original Image')
                st.image(img, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_img2:
                if show_bounding_boxes:
                    try:
                        data, img_with_boxes = get_text_regions(img, ocr_language)
                        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                        st.write('##### Text Detection')
                        st.image(img_with_boxes, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"Could not generate bounding boxes: {e}")
            
            extracted_text = extract_text_from_image(img, ocr_language)
            
            if extracted_text:
                st.success("Text successfully extracted!")
                
                # Tabs in glass card
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                tab1, tab2, tab3 = st.tabs(["📝 Extracted Text", "📊 Text Analysis", "🤖 Summary"])
                
                with tab1:
                    st.text_area("Full Text", extracted_text, height=300, label_visibility="collapsed")
                
                with tab2:
                    stats, common_words = analyze_text(extracted_text)
                    
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("Words", stats['words'])
                        st.metric("Characters", stats['characters'])
                    with col_stat2:
                        st.metric("Lines", stats['lines'])
                        st.metric("Sentences", stats['sentences'])
                    with col_stat3:
                        st.metric("Paragraphs", stats['paragraphs'])
                        st.metric("Avg Word Length", f"{stats['avg_word_length']:.1f}")
                    
                    st.write("**Most Frequent Words:**")
                    common_df = pd.DataFrame(common_words, columns=['Word', 'Frequency'])
                    st.dataframe(common_df, use_container_width=True)
                
                with tab3:
                    if generate_summary:
                        summary = summarize_text(extracted_text)
                        st.write("**Summary:**")
                        st.info(summary)
                    else:
                        st.info("Enable 'Generate Summary' in settings")
                
                st.download_button(
                    label="📥 Download Text",
                    data=extracted_text,
                    file_name=f"extracted_{file_name.split('.')[0]}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("❌ No text detected. Try adjusting language settings or image quality.")
    
    elif file_extension == 'pdf':
        st.info("📄 PDF processing may take a moment...")
        with st.spinner('Processing PDF pages...'):
            # Pass raw bytes directly to convert_from_bytes
            pdf_texts = extract_text_from_pdf(bytes_data, ocr_language)
        
        if pdf_texts:
            st.success(f"✅ Processed {len(pdf_texts)} page(s)!")
            
            all_text = "\n\n".join([f"Page {p['page']}:\n{p['text']}" for p in pdf_texts])
            
            # PDF results in glass card
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            tab1, tab2 = st.tabs(["📄 Pages", "📊 Analysis"])
            
            with tab1:
                for page in pdf_texts:
                    with st.expander(f"Page {page['page']} - {len(page['text'])} characters"):
                        st.text(page['text'])
            
            with tab2:
                stats, common_words = analyze_text(all_text)
                col_pdf1, col_pdf2, col_pdf3 = st.columns(3)
                with col_pdf1:
                    st.metric("Total Pages", len(pdf_texts))
                    st.metric("Total Words", stats['words'])
                with col_pdf2:
                    st.metric("Total Characters", stats['characters'])
                    st.metric("Vocabulary Diversity", f"{stats['vocabulary_diversity']:.2%}")
                with col_pdf3:
                    st.metric("Sentences", stats['sentences'])
                    st.metric("Paragraphs", stats['paragraphs'])
                
                if generate_summary:
                    summary = summarize_text(all_text)
                    st.write("**Document Summary:**")
                    st.info(summary)
            
            st.download_button(
                label="📥 Download PDF Text",
                data=all_text,
                file_name=f"pdf_extracted_{file_name.split('.')[0]}.txt",
                mime="text/plain",
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("❌ No text extracted from PDF. Try different language settings.")

# Footer
st.divider()
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.write(f"**Using {OCR_ENGINE.upper()} OCR Engine**")
st.markdown(f"""
**Performance:** {' Fast' if OCR_ENGINE == 'tesseract' else '🐢 Slow but reliable'}

**Available Languages:** {', '.join([opt[1] for opt in available_options])}
""")
st.markdown('</div>', unsafe_allow_html=True)
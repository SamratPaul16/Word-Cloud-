import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PyPDF2 as p
from docx import Document
from wordcloud import WordCloud, STOPWORDS
import base64
from io import BytesIO
from PIL import Image
import os

# Function to read text file
def read_txt(file):
    return file.getvalue().decode('utf-8')

# Function to read pdf file
def read_pdf(file):
    pdf = p.PdfReader(file)
    return " ".join([page.extract_text() for page in pdf.pages])

# Function to read docx file
def read_docx(file):
    doc = Document(file)
    return " ".join([para.text for para in doc.paragraphs])

# Function to filter stopwords
def filter_stopwords(text, additional_stopwords=[]):
    words = text.split()
    all_stopwords = STOPWORDS.union(set(additional_stopwords))
    filtered_words = [word for word in words if word.lower() not in all_stopwords]
    return " ".join(filtered_words)

# Function to create download link for plot
def get_image_download_link(buffered, format_):
    image_base64 = base64.b64encode(buffered).decode()
    return f'<a href="data:image/{format_};base64,{image_base64}" download="plot.{format_}">Download Plot ({format_.upper()})</a>'

# Function to generate download link for a DataFrame
def get_table_download_link(df, filename, file_label):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:text/csv;base64,{b64}" download="{filename}">{file_label}</a>'

# Streamlit UI
st.title('Word Cloud Application')
st.subheader("📁Upload a PDF, DOCX, or TXT file to generate a word cloud")

upload_file = st.file_uploader("Choose a file to upload", type=["pdf", "docx", "txt"])

if upload_file:
    file_details = {"Filename": upload_file.name, "File Type": upload_file.type, "File Size": upload_file.size}
    st.write("File Details:", file_details)

    try:
        if upload_file.type == "text/plain":
            text = read_txt(upload_file)
        elif upload_file.type == "application/pdf":
            text = read_pdf(upload_file)
        elif upload_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = read_docx(upload_file)
        else:
            st.error("File type not supported. Please upload a TXT, PDF, or DOCX file.")
            st.stop()
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        st.stop()

    # Generate word count table
    words = text.split()
    word_count = (pd.DataFrame({"word": words})
                  .groupby('word').size()
                  .reset_index(name="count")
                  .sort_values('count', ascending=False))

    # Sidebar: Settings
    st.sidebar.header("Settings")
    stopwords_checkbox = st.sidebar.checkbox("Remove default stopwords")
    additional_stopwords = st.sidebar.multiselect(
        "Add additional stopwords", ["the", "and", "a", "an", "is", "in", "it", "of", "to","The","An","and","A","this","This","that"])

    width = st.sidebar.slider("Width", 100, 1000, 800)
    height = st.sidebar.slider("Height", 100, 1000, 400)
    max_font_size = st.sidebar.slider("Max Font Size", 10, 110, 110)
    resolution = st.sidebar.selectbox("Resolution", ["Low", "Medium", "High"], index=1)
    dpi = {"Low": 72, "Medium": 144, "High": 288}[resolution]
    file_format = st.sidebar.selectbox("File Format", ["png", "jpeg", "pdf", "svg"], index=0)

    # Mask file selection
    mask_file = st.sidebar.file_uploader("Upload a mask image (optional)", type=["png", "jpg", "jpeg"])

    # Default mask dropdown
    # Path to masks folder (relative to script location)
    default_mask_path = os.path.join(os.path.dirname(__file__), "masks-wordclouds")
    default_masks = [f for f in os.listdir(default_mask_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    selected_mask_name = st.sidebar.selectbox("Or select a default mask", [None] + default_masks)

    if stopwords_checkbox:
        text = filter_stopwords(text, additional_stopwords)

    # Generate word cloud
    try:
        mask = None
        if mask_file:
            mask = np.array(Image.open(mask_file))
        elif selected_mask_name:
            mask = np.array(Image.open(os.path.join(default_mask_path, selected_mask_name)))

        wordcloud = WordCloud(
            width=width, height=height, max_font_size=max_font_size, 
            mask=mask, background_color="white"
        ).generate(text)

        plt.figure(figsize=(10, 8))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")

        # Save the plot to a bytes buffer
        buffered = BytesIO()
        plt.savefig(buffered, format=file_format, dpi=dpi, bbox_inches='tight')
        buffered.seek(0)

        # Display word cloud
        st.pyplot(plt)

        # Download word cloud
        st.markdown(get_image_download_link(buffered.read(), file_format), unsafe_allow_html=True)

        # Display and download word count table
        st.subheader("Word Count Table")
        st.dataframe(word_count)
        st.markdown(get_table_download_link(word_count, "word_count.csv", "Download Word Count Table"), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error generating word cloud: {e}")

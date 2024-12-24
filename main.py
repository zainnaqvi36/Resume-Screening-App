import streamlit as st
import pandas as pd
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model and TF-IDF vectorizer
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# Load the category mapping CSV (replace with the path to your CSV)
category_mapping_df = pd.read_csv('category_mapping.csv')

# Define text cleaning functions
def cleanresume(txt):
    # Clean the text: remove URLs, mentions, hashtags
    cleantxt = re.sub(r'http\s+\S+', ' ', txt)
    cleantxt = re.sub(r'@\S+', ' ', cleantxt)
    cleantxt = re.sub(r'#\S+', ' ', cleantxt)
    return cleantxt

def clean_special_chars(txt):
    # Remove special characters like *, ,, ., -, :, &, etc.
    return re.sub(r'[^\w\s,]', '', txt)

# Function to predict category from resume text
def predict_category(resume_text):
    # Clean the resume text
    cleaned_resume = cleanresume(resume_text)
    cleaned_resume = clean_special_chars(cleaned_resume)

    # Transform the cleaned text using the same TF-IDF vectorizer
    input_features = tfidf.transform([cleaned_resume])

    # Get the predicted numeric label
    predicted_label = clf.predict(input_features)[0]

    # Map the predicted numeric label to the original category
    predicted_category = category_mapping_df[category_mapping_df['Numeric_Label'] == predicted_label]['Original_Category'].values[0]

    return predicted_category

# Streamlit UI
st.title("Resume Category Prediction")

st.write(
    """
    Upload your resume (in text format), and the system will predict the category it belongs to.
    """
)

# Upload resume
uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "docx"])

if uploaded_file is not None:
    # If the file is uploaded, read and display the file content
    if uploaded_file.type == "text/plain":
        resume_text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        import PyPDF2
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        resume_text = ""
        for page in pdf_reader.pages:
            resume_text += page.extract_text()
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        from docx import Document
        doc = Document(uploaded_file)
        resume_text = ""
        for para in doc.paragraphs:
            resume_text += para.text

    # Show the resume text
    st.subheader("Resume Content:")
    st.text(resume_text)

    # Predict the category
    if resume_text:
        predicted_category = predict_category(resume_text)
        st.subheader(f"Predicted Category: {predicted_category}")


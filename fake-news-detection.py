import streamlit as st
import pickle
from sklearn.linear_model import LogisticRegression
import base64

# Load the pre-trained model and vectorizer
model = pickle.load(open('pred.pkl', 'rb'))
vector = pickle.load(open('tfidf.pkl', 'rb'))

# Set Streamlit page config
st.set_page_config(page_title="Fake News Detection", layout="centered")

# Function to add background image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover;
        background-attachment: fixed;
    }}
    .main {{
        background-color: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 15px;
        margin-top: 2rem;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

# Add background image (replace with your image path)
add_bg_from_local('news_background.jpg')  # You need to have this image in your directory

# Apply Custom CSS for Styling
st.markdown("""
    <style>
        .title {
            font-size: 42px !important;
            color: #2c3e50 !important;
            text-align: center;
            margin-bottom: 20px;
            color: #00f108 !important;
        }
        .description {
            font-size: 18px;
            color: #34495e;
            text-align: center;
            margin-bottom: 30px;
            color: #00f108 !important;
        }
        .result-true { 
            color: #27ae60; 
            font-size: 28px; 
            font-weight: bold;
            text-align: center;
            padding: 15px;
            background-color: rgba(39, 174, 96, 0.1);
            border-radius: 10px;
            margin-top: 20px;
        }
        .result-fake { 
            color: #e74c3c; 
            font-size: 28px; 
            font-weight: bold;
            text-align: center;
            padding: 15px;
            background-color: rgba(231, 76, 60, 0.1);
            border-radius: 10px;
            margin-top: 20px;
        }
        .stTextArea textarea {
            border: 2px solid #3498db !important;
            border-radius: 10px !important;
            padding: 10px !important;
        }
        .stButton>button {
            background-color: #3498db !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 10px 24px !important;
            font-size: 18px !important;
            border: none !important;
            width: 100%;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            background-color: #2980b9 !important;
            transform: scale(1.02);
        }
        .st-emotion-cache-mtjnbi {
            background-color: #00000087 !important;
            border-radius: 20% !important;
        }
    </style>
""", unsafe_allow_html=True)

# Main content container
st.markdown("<div class='main'>", unsafe_allow_html=True)

# Title and Description
st.markdown("<h1 class='title'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='description'>Analyze news articles or headlines to determine their authenticity using AI</p>", unsafe_allow_html=True)

# Input text area
message = st.text_area("", placeholder="Paste news content here...", height=200)

# Predict button
if st.button("Analyze News Authenticity"):
    if message.strip():
        transformed_message = vector.transform([message])
        output = model.predict(transformed_message)[0]
        
        # Display result with animation
        if output == 1:
            st.markdown("<p class='result-true'>✅ Genuine News Content</p>", unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown("<p class='result-fake'>❌ Potential Fake News</p>", unsafe_allow_html=True)
            st.snow()
    else:
        st.warning("⚠ Please enter news content before analyzing")

st.markdown("</div>", unsafe_allow_html=True)
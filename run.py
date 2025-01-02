import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

 
model = tf.keras.models.load_model('model_plant.h5')


class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
    'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]


disease_descriptions = {
    'Apple___Apple_scab': 'Apple scab is a fungal disease that causes lesions on apple leaves and fruit.',
    'Apple___Black_rot': 'Black rot is a fungal disease that causes black lesions and can lead to fruit decay.',
    'Apple___Cedar_apple_rust': 'Cedar apple rust causes orange spots on the leaves and fruit of apple trees.',
    'Apple___healthy': 'This apple plant is healthy with no signs of disease.',
    'Blueberry___healthy': 'This blueberry plant is healthy with no signs of disease.',
    'Cherry_(including_sour)___Powdery_mildew': 'Powdery mildew causes a white, powdery coating on cherry leaves.',
    'Cherry_(including_sour)___healthy': 'This cherry plant is healthy with no signs of disease.',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Gray leaf spot is a fungal disease that causes gray lesions on corn leaves.',
    'Corn_(maize)___Common_rust_': 'Common rust causes orange-red pustules on corn leaves.',
    'Corn_(maize)___Northern_Leaf_Blight': 'Northern leaf blight causes long lesions on corn leaves.',
    'Corn_(maize)___healthy': 'This corn plant is healthy with no signs of disease.',
    'Grape___Black_rot': 'Black rot is a fungal disease that affects grapevines, causing black lesions and decaying fruit.',
    'Grape___Esca_(Black_Measles)': 'Esca, or black measles, causes wilting and dark streaks in grapevines.',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Leaf blight causes dark spots and lesions on grape leaves.',
    'Grape___healthy': 'This grape plant is healthy with no signs of disease.',
    'Orange___Haunglongbing_(Citrus_greening)': 'Citrus greening causes yellowing of leaves and fruit drop in orange trees.',
    'Peach___Bacterial_spot': 'Bacterial spot causes lesions and spots on peach leaves and fruit.',
    'Peach___healthy': 'This peach plant is healthy with no signs of disease.',
    'Pepper,_bell___Bacterial_spot': 'Bacterial spot causes lesions and yellowing of bell pepper leaves.',
    'Pepper,_bell___healthy': 'This bell pepper plant is healthy with no signs of disease.',
    'Potato___Early_blight': 'Early blight causes dark spots and lesions on potato leaves.',
    'Potato___Late_blight': 'Late blight causes lesions and mold on potato plants, leading to rotting.',
    'Potato___healthy': 'This potato plant is healthy with no signs of disease.',
    'Raspberry___healthy': 'This raspberry plant is healthy with no signs of disease.',
    'Soybean___healthy': 'This soybean plant is healthy with no signs of disease.',
    'Squash___Powdery_mildew': 'Powdery mildew causes a white, powdery coating on squash leaves.',
    'Strawberry___Leaf_scorch': 'Leaf scorch causes yellowing and scorched edges on strawberry leaves.',
    'Strawberry___healthy': 'This strawberry plant is healthy with no signs of disease.',
    'Tomato___Bacterial_spot': 'Bacterial spot causes lesions and spots on tomato leaves.',
    'Tomato___Early_blight': 'Early blight causes dark lesions on tomato leaves and fruit.',
    'Tomato___Late_blight': 'Late blight causes water-soaked lesions and decaying fruit in tomato plants.',
    'Tomato___Leaf_Mold': 'Leaf mold causes fuzzy mold growth on tomato leaves.',
    'Tomato___Septoria_leaf_spot': 'Septoria leaf spot causes round, dark lesions on tomato leaves.',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Two-spotted spider mites cause yellowing and spotting on tomato leaves.',
    'Tomato___Target_Spot': 'Target spot causes round, concentric lesions on tomato leaves.',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Yellow leaf curl virus causes yellowing and curling of tomato leaves.',
    'Tomato___Tomato_mosaic_virus': 'Tomato mosaic virus causes mottled patterns and discoloration of tomato leaves.',
    'Tomato___healthy': 'This tomato plant is healthy with no signs of disease.'
}


st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="wide")


st.markdown("""
    <h1 style="text-align:center; color:#4CAF50;">Plant Disease Detection</h1>
    <p style="text-align:center; color:#555555;">Upload an image of a plant leaf to detect any potential disease using AI.</p>
""", unsafe_allow_html=True)


st.markdown("<h3 style='color:#388E3C;'>Supported Plant Disease Classes</h3>", unsafe_allow_html=True)
st.write("""
    The model can detect diseases in the following plants:
    - Apple
    - Blueberry
    - Cherry
    - Corn
    - Grape
    - Orange
    - Peach
    - Pepper (Bell)
    - Potato
    - Raspberry
    - Soybean
    - Squash
    - Strawberry
    - Tomato
""")


st.markdown("<h3 style='color:#388E3C;'>Upload a Plant Image</h3>", unsafe_allow_html=True)
uploaded_img = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_img is not None:
    try:

        img = Image.open(uploaded_img)

        if img.mode != 'RGB':
            img = img.convert('RGB')

        img_np = np.array(img)


        st.write(f"Image shape: {img_np.shape}")
        if img_np.shape == ():
            raise ValueError("The uploaded image has an invalid shape.")


        st.image(img, caption='Uploaded Image', use_column_width=True)


        image = img.resize((128, 128))


        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])


        prediction = model.predict(input_arr)
        result_index = np.argmax(prediction)
        model_prediction = class_names[result_index]


        st.write(f"### Predicted Disease: **{model_prediction}**")


        description = disease_descriptions.get(model_prediction, 'No description available.')
        st.write(f"### Disease Description: {description}")


        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.set_title(f"Disease: {model_prediction}", fontsize=15)
        ax.axis('off')
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")


st.markdown("""
    <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 10px 24px;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stFileUploader {
            background-color: #f1f1f1;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

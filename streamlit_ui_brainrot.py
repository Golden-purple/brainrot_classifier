import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("models/breinrot_classifier_vgg16.keras")
    return model

model = load_model()

st.title("Brainrot Classifier \n Can identify the following : Tung Tung Tung Sahur , Ballerina Cappucina , Lirilli Larilla , " \
"Tralelero Tralala")
st.write("This is using VGG16 architecture to classify the images! \n Upload an image !")

uploaded_file = st.file_uploader("Choose an image : ", type = ["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image , caption = "Uploaded Image" , use_column_width = True)

    img_resized = image.resize((224, 224)) 
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array , axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    ans = predicted_class

    if(ans == 0) :
        st.write("Found Ballerina Cappucina  in the image !")
    elif (ans == 1) :
        st.write("Found Brr Brr Patapim  in the image !")
    elif (ans == 2) :
        st.write("Found Lirilli Larilla  in the image !")
    elif (ans == 3) :
        st.write("Found Tralalero Tralala  in the image !")
    elif (ans == 4) :
        st.write("Found Tung Tung Tung Tung Tung Tung Tung Sahur  in the image !")

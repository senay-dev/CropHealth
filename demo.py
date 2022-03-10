import pandas as pd
import numpy as np
import streamlit as st
from os import listdir
from os.path import isfile, join
from PIL import Image
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from PIL import Image
from tensorflow.keras.utils import img_to_array

showpred = 0
try:
	model_path = 'model_80.h5'
except: 
	print("Need to train model")
test_path = 'Data/Test'

#Load the pre-trained models
model = load_model(model_path)
st.sidebar.title("About")

st.sidebar.info(
    "This application identifies the crop health in the picture.")

#onlyfiles = [f for f in listdir("Data/Test") if isfile(join("Data/Test", f))]

#st.sidebar.title("Train Neural Network")
#if st.sidebar.button('Train CNN'):
	#Training.train()

#st.sidebar.title("Predict Images")
#imageselect = st.sidebar.selectbox("Pick an image.", onlyfiles)
#if st.sidebar.button('Predict Crop Health'):
    #showpred = 1
#prediction = Testing.predict((model),"Data/Test/" + imageselect)


st.title('Crop Health Identification')
st.write("Pick an image.")
st.write("When you're ready, submit a prediction.")

uploaded_file = st.file_uploader("Choose an image...")
if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((480,480))
    image_arr = img_to_array(image)
    image_arr= np.reshape(image_arr, [1,480,480,3])
    
    label = model.predict(image_arr)
    prediction = np.argmax(label)
    if prediction == 0:
    	st.write("This is a **healthy crop!**")
    if prediction == 1:
    	st.write("This is a **crop that has leaf rust!**")
    if prediction == 2:
    	st.write("This is a **crop that has stem rust!**")
    #st.image(image, caption='Uploaded Image.', use_column_width=True)
#st.write("")
#image = Image.open("Data/Test/" + imageselect)
#st.image(image, caption="Let's predict the health of this crop!", use_column_width=True)




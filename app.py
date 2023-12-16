import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import cv2
import os
import streamlit as st
from tensorflow.keras.models import load_model

st.title("Picture2Sketch Generator")
st.subheader("Using Convolutional Autoencoder")
st.image("https://i0.wp.com/sefiks.com/wp-content/uploads/2018/03/convolutional-autoencoder.png?fit=1818%2C608&ssl=1", caption='Your Image Caption', use_column_width=True)

model = load_model('AE-1.0.h5')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image,(180,180),interpolation=cv2.INTER_AREA)
    image = (image-image.min())/(image.max()-image.min())
    st.image(image, caption="Uploaded Image.", width=180)
    print(image.shape)
    print(image.max())
    print(image.min())
    image = np.reshape(image,[1,180,180,3])
    res = model.predict(image)
    res = res.reshape([180,180,3])
    #res = cv2.resize(res,(300,300),interpolation=cv2.INTER_AREA)
    st.image(res, width=180, caption="Sketched Image")

st.subheader("Created by Jibitesh Chakraborty")

import tensorflow as tf
from tensorflow import _keras
from keras.models import load_model
import streamlit as st
import numpy as np
# import cv2
import PIL
import matplotlib.pyplot as plt

model=load_model("tomato_disease.h5")

st.set_page_config(layout='wide')

st.markdown("<h1 style ='text-align: center;color: red'>**Tomato Plant Disease Prediction**</h1>",unsafe_allow_html=True)

img=st.file_uploader("Please upload the image of Tomato leaf in JPG or PNG format",type=['jpg','png'])

label={0: 'Tomato___Bacterial_spot',1: 'Tomato___Early_blight',2: 'Tomato___Late_blight',3: 'Tomato___Leaf_Mold',4: 'Tomato___Septoria_leaf_spot',\
    5: 'Tomato___Spider_mites Two-spotted_spider_mite',6: 'Tomato___Target_Spot',7: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',8: 'Tomato___Tomato_mosaic_virus',\
        9: 'Tomato___healthy'}
st.text(label)

if img is not None:

    col1, col2 = st.columns(2)

    with col1:
        st.write("""
        
        # Image to be Predicted
        
        """
        )

    image=PIL.Image.open(img)
    img_arr = np.array(image)
    scaled_img=img_arr/255
    st.image(image)
    st.text(f"Uploaded Image size is {img_arr.shape}")
    def image_processing(image_array):
        return image_array.reshape((1,)+(256,256,3))
    
    if img_arr.shape[0:2] != (256,256):
        re_sized_img=image.resize((256,256))
        re_sized_arr=np.array(re_sized_img)
        # re_sized_arr=cv2.resize(scaled_img, (256,256),interpolation=cv2.INTER_CUBIC)
        st.text(f"Uploaded Image is re-sized to {re_sized_arr.shape}")
        test=image_processing(re_sized_arr)
    else:
        test=image_processing(scaled_img)

    with col2:

        if(st.button("Convert")):
            pred=model.predict(test)
            st.text(label[np.argmax(pred)])

else:
    st.text("Please upload the image")



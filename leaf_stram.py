import tensorflow as tf
from tensorflow import _keras
from keras.models import load_model
import streamlit as st
import numpy as np
import cv2
import PIL
import matplotlib.pyplot as plt

model=load_model("tomato_disease.h5")
#model=load_model("model.h5")

st.set_page_config(layout='wide')

st.markdown("<h1 style ='text-align: center;color: red'>**ทำนายโรคใบมะเขือเทศ จากภาพด้วย CNN**</h1>",unsafe_allow_html=True)

img=st.file_uploader("Please upload the image in JPG or PNG format",type=['jpg'])


label={0: 'Bacterial spot',1: 'Early blight',2: 'Late_blight',3: 'Leaf_Mold',4: 'Septoria_leaf_spot',\
    5: 'Spider_mites Two-spotted_spider_mite',6: 'Target_Spot',7: 'Yellow_Leaf_Curl_Virus',8: 'mosaic_virus',\
        9: 'healthy'}
st.text(label)

if img is not None:

    col1, col2 = st.columns(2)

    with col1:
        st.write("""
        
        # รูปภาพ
        
        """
        )
    image=PIL.Image.open(img)
    newsize = (256,256)
    img_arr = np.array(image)
    scaled_img=img_arr/255
    st.image(image)
    st.text(f"Uploaded Image size is {img_arr.shape}")
    def image_processing(image_array):
        return image_array.reshape((1,)+(256,256,3))
    
    if img_arr.shape[0:2] != newsize:
        #re_sized_img=image.resize(newsize)
        #re_sized_arr=np.array(re_sized_img)
        re_sized_arr=cv2.resize(scaled_img, newsize,interpolation=cv2.INTER_CUBIC)
        st.text(f"Uploaded Image is re-sized to {re_sized_arr.shape}")
        test=image_processing(re_sized_arr)
    else:
        test=image_processing(scaled_img)
#     image_ori=PIL.Image.open(img)
#     newsize = (128, 128)
#     image = cv2.resize(image_ori,(newsize),interpolation=cv2.INTER_CUBIC)
#     img_arr = np.array(image)
#     scaled_img=img_arr/255
#     st.image(image_ori)
    
#     scaled_img.reshape((1,)+(128,128,3))
    
    

    with col2:

        if(st.button("ทำนายเลย")):
            pred=model.predict(test)
            result = label[np.argmax(pred)]
            st.write("""
            # {result}
            """)

else:
            st.text("Please upload the image")



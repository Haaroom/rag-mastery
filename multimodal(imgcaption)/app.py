from transformers import (BlipProcessor,BlipForConditionalGeneration)
from PIL import Image
import streamlit as st
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base")
    return processor, model
processor, model = load_model()
def generate_caption(img):
    inputs = processor(img,return_tensors="pt")
    outputs = model.generate(**inputs,max_new_tokens=50)
    caption = processor.decode(outputs[0],skip_special_tokens=True)
    return caption
st.set_page_config(page_title="Caption Generator")
st.title("Image Caption Generator")
uploaded_file = st.file_uploader("Choose an image",type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img,caption="Uploaded Image",use_container_width=True)
    if st.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            caption = generate_caption(img)
        st.success("Caption Generated")
        st.write("Caption")
        st.write(caption)
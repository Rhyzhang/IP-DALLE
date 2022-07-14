import streamlit as st
from PIL import Image
import io

def image_to_byte_array(image: Image) -> bytes:
    """Converts a PIL image to a byte array."""
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format='png')
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr

st.set_page_config(
     page_title="Cropping",
     page_icon="✂️",
     initial_sidebar_state="expanded",
     menu_items={
         'About': "Cropping images for Dalle"
     }
)
st.sidebar.subheader("About")
st.sidebar.markdown("""
    This is a community streamlit app for streamlining the creative process for Dalle 🥰.
    It will also serve as a collection of useful and creative Dalle techniques to bring
    your creations to the next level 🤩. Please feel free to use this app for your own purposes.
""")
st.sidebar.info("""
    This app is made by [@Rhyzhang](https://github.com/Rhyzhang) and is on [GitHub](https://github.com/Rhyzhang/IP-DALLE).
""")

st.title("Zoom Out Cropping Effect")

uploaded_img = st.file_uploader("Upload your Dalle Generation", type=["png"])
if uploaded_img is not None:
    # Get img file name
    uploaded_img_name = uploaded_img.name

    # Load the image
    uploaded_img = Image.open(uploaded_img)

    # Figure out the size of the image
    width, height = uploaded_img.size

    # Dimensions of the crop and crop the image
    left = -(st.number_input('Crop how many pixels to the left?', min_value=100, max_value=10000, value=100, step=100))
    right = width + (st.number_input('Crop how many pixels to the right?', min_value=100, max_value=10000, value=100, step=100))
    top = -(st.number_input('Crop how many pixels to the top?', min_value=100, max_value=10000, value=100, step=100))
    bottom = height + (st.number_input('Crop how many pixels to the bottom?', min_value=100, max_value=10000, value=100, step=100))
    cropped_img = uploaded_img.crop((left, top, right, bottom))

    # Replace all black pixels with transparent pixels
    rgba = cropped_img.convert("RGBA")
    datas = rgba.getdata()
    newData = []
    for item in datas:
        if item[0] == 0 and item[1] == 0 and item[2] == 0:  # finding black colour by its RGB value
            # storing a transparent value when we find a black colour
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)  # other colours remain unchanged
    rgba.putdata(newData)


    # Show the image
    col1, col2 = st.columns(2)
    col1.image(uploaded_img, caption="Original Image",use_column_width=True)
    col2.image(cropped_img, caption="Cropped Image", use_column_width=True)
    
    # Download the cropped 
    cropped_img_bytes = image_to_byte_array(image=rgba)
    btn = col2.download_button(
             label="Download image",
             data=cropped_img_bytes,
             file_name=uploaded_img_name.split(".")[0] + "_cropped.png",
             mime="image/png"
            )
else:
    st.warning("No image uploaded")

    st.header("What is Zoom Out Cropping?")
    st.markdown("""
        I don't know what to call this effect, but it's a way to crop the image outwards as if you were zooming out.
        Dalle would then be prompted to fill in the transparent pixels.
    """)
    st.markdown("Example: 'futuristic bridge across the Grand Canyon, digital art'")
    col1, col2, col3 = st.columns(3)
    col1.image(r"./images/ZOC/bridge_original.png", caption="From this", use_column_width=True)
    col2.image(r"./images/ZOC/bridge_cropped.png", caption="To this", use_column_width=True)
    col3.image(r"./images/ZOC/bridge_final.png", caption="To this", use_column_width=True)
    
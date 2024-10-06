import numpy as np
import streamlit as st
import requests
from PIL import Image
import io

def main():
    st.title("Pathfinding with Image and Coordinates")

    uploaded_image = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

    st.write("Select the Start Point:")
    x_start = st.number_input("X coordinate for Start (0-31)", min_value=0, max_value=31)
    y_start = st.number_input("Y coordinate for Start (0-31)", min_value=0, max_value=31)

    st.write("Select the Goal Point:")
    x_goal = st.number_input("X coordinate for Goal (0-31)", min_value=0, max_value=31)
    y_goal = st.number_input("Y coordinate for Goal (0-31)", min_value=0, max_value=31)

    if st.button("Submit Coordinates"):
        if uploaded_image is not None:
            img = Image.open(uploaded_image)
            buf = io.BytesIO()
            img.save(buf, format='JPEG')
            img_bytes = buf.getvalue()

            files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
            data = {
                "x_start": x_start,
                "y_start": y_start,
                "x_goal": x_goal,
                "y_goal": y_goal
            }

            response = requests.post("http://localhost:8000/pathfinding/", files=files, data=data)

            if response.status_code == 200:
                result = response.json()
                st.success(f"Valid Start Point: {result['start']}, Valid Goal Point: {result['goal']}")
                img_array = np.array(result['path'], dtype=np.uint8)
                st.image(img_array, caption="Path Found", use_column_width=True)
            else:
                st.error("Failed to get a valid response from the backend.")
        else:
            st.warning("Please upload an image.")

if __name__ == "__main__":
    main()

import streamlit as st
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import cnn_model
import numpy as np
import pandas as pd

# Load the trained plant disease prediction model
model = cnn_model.CNN(39)
model.load_state_dict(torch.load("disease_model.pt"))

# Load the disease information from the CSV file
disease_info = pd.read_csv("disease_info.csv", encoding='latin-1')  # Replace with your actual CSV file name

# Plant Disease Prediction Page
def plant_disease_prediction():
    st.title("Plant Disease Detection")
    st.sidebar.subheader("Inputs")
    st.sidebar.image('plants-7679.gif', width=300)
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    # # Display user inputs for plant disease prediction
    # if uploaded_file is not None:
    #     st.subheader("User Inputs for Disease Prediction")
        # st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Plant Disease Prediction Logic
    if uploaded_file is not None and st.button("Predict Disease"):
        # Load the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        st.write("Classifying...")

        def predict_plant_disease(image):
            # Function for plant disease prediction
            image = image.resize((224, 224))
            input_data = TF.to_tensor(image)
            input_data = input_data.view((-1, 3, 224, 224))

            with torch.no_grad():
                output = model(input_data)
                output = output.detach().numpy()
                index = np.argmax(output)

            return index

        # Predict disease index
        prediction_index = predict_plant_disease(image)
        predicted_disease = disease_info.loc[prediction_index, 'disease_name']

        if "Healthy" in predicted_disease:
            st.success(f"The plant is healthy.")
        else:
            # Fetch additional information from the CSV based on the predicted disease
            disease_info_row = disease_info[disease_info['disease_name'] == predicted_disease].iloc[0]

            st.success(f"The plant is predicted to have: {predicted_disease}")

            # Display disease information
            st.subheader("Disease Information")
            st.write(f"**Description:** {disease_info_row['description']}")
            st.write(f"**Possible Steps:** {disease_info_row['Possible Steps']}")
            st.image(disease_info_row['image_url'], caption="Reference Image", use_column_width=True)

# Run the app
if __name__ == "__main__":
    plant_disease_prediction()

import streamlit as st
import google.generativeai as genai
import os
import cv2
import numpy as np
import PIL.Image

# Set API Key for Google Gemini
os.environ["GOOGLE_API_KEY"] = "your api key"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Load the Gemini Model
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

# Function to detect faces using OpenCV (only returns the largest face)
def detect_faces(image):
    # Convert PIL image to OpenCV format
    img_cv = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

    # Load pre-trained face detection model (Haar cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None, img_cv

    # Select the largest face (assuming it's the main subject)
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])  # Sort by area (w*h)
    return [largest_face], img_cv  # Return only the largest face


# Function to draw bounding boxes on faces
def draw_faces(image, faces, player_name):
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)

        # Adjust font scale dynamically
        font_scale = max(0.5, min(1.5, 30 / len(player_name)))  # Scales based on name length

    return image


# Function to generate detailed player info
def get_player_info(image):
    prompt = """
    You are an expert AI trained in recognizing professional football (soccer) players with high accuracy. 
    Carefully analyze the given image and identify the player with certainty. 
    If a well-known player is present, return the following details in a structured format:

    - **Full Name**: (Only the player's full name, no extra text)
    - **Nationality**: (Country of origin)
    - **Club**: (Current club name)
    - **Position**: (Primary playing position)
    - **Achievements**: (Notable titles or awards, max 2-3)

    If multiple players are detected, return only the **most famous one**.
    If no player is recognized, respond with "Unknown".
    """

    response = model.generate_content([prompt, image])
    return response.text.strip()


# Streamlit App
st.title("Football Player Detection")
st.write("Upload an image, and the app will detect the player and provide details.")

# Image Upload
uploaded_image = st.file_uploader("Upload a Player's Image", type=['png', 'jpg', 'jpeg'])

if uploaded_image:
    # Open Image
    img = PIL.Image.open(uploaded_image)

    # Detect Faces
    faces, img_cv = detect_faces(img)

    if faces:
        # Get Player Info from Gemini
        player_info = get_player_info(img)

        # Extract the player's name
        player_name = player_info.split("\n")[0].replace("**Full Name**: ", "").strip()

        # Draw Bounding Box and Name
        processed_img = draw_faces(img_cv, faces, player_name)

        # Convert OpenCV image back to PIL for display
        processed_pil_img = PIL.Image.fromarray(processed_img)

        # Resize image before displaying
        display_width = 340  # Set desired width
        display_height = 360  # Set desired height

        # Resize the image while maintaining aspect ratio
        processed_pil_img = processed_pil_img.resize((display_width, display_height))

        # Display the resized image

        col1, col2 = st.columns(2)

        with col1:
            st.image(processed_pil_img, caption="Detected Player", width=display_width)

        with col2:
            st.write(player_info)  # Display structured player info

    else:
        st.warning("No faces detected. Try another image.")



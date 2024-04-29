import requests
from PIL import Image
import gradio as gr

# Define the API endpoint URL
api_url = "http://127.0.0.1:5000/predict"

# Define the predict function
def predict(img):
    # Open and preprocess the image
    img = Image.fromarray(img)
    img.save("temp_image.jpg")  # Save the image temporarily
    
    # Send a POST request to the API endpoint
    files = {"image": open("temp_image.jpg", "rb")}
    response = requests.post(api_url, files=files)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Get the prediction results
        emotions = response.json()
        return emotions
    else:
        return {"Error": "Failed to get prediction from the API"}

# Define Gradio interface
title = "Facial Emotion Detector"
description = gr.Markdown(
    """Upload a photo and discover the emotions captured in the moment!
      **Tip**: Be sure to only include face to get best results. Check some sample images
                 below for inspiration!""").value
examples = ['happy1.jpg', 'happy2.jpg', 'angry1.png', 'angry2.jpg', 'neutral1.jpg', 'neutral2.jpg']

gr.Interface(fn = predict, 
             inputs = gr.inputs.Image(shape=(48,48), image_mode='L'), 
             outputs = gr.outputs.Label(label='Emotion'), 
             title = title,
             examples = examples,
             description = description,
             allow_flagging='never').launch(debug=True, share=True)



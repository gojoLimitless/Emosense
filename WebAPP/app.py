# Facial expression classifier
import os
from fastai.vision.all import *
import gradio as gr
import pathlib   
temp = pathlib.PosixPath   
pathlib.PosixPath = pathlib.WindowsPath

# Emotion
learn_emotion = load_learner('emotion_model.pkl')
learn_emotion_labels = learn_emotion.dls.vocab

# Predict
def predict(img):
    img = PILImage.create(img)
    
    pred_emotion, pred_emotion_idx, probs_emotion = learn_emotion.predict(img)
    
    emotions = {learn_emotion_labels[i]: float(probs_emotion[i]) for i in range(len(learn_emotion_labels))}
    
    return emotions

# Gradio
title = "Facial Emotion Detector"
description = gr.Markdown(
    """Upload a photo and discover the emotions captured in the moment!
      **Tip**: Be sure to only include face to get best results. Check some sample images
                 below for inspiration!""").value


examples = ['happy1.jpg', 'happy2.jpg', 'angry1.png', 'angry2.jpg', 'neutral1.jpg', 'neutral2.jpg']

gr.Interface(fn = predict, 
             inputs = gr.Image(shape=(48,48),image_mode='L'), 
             outputs = gr.Label(label='Emotion'), 
             title = title,
             examples = examples,
             description = description,
             allow_flagging='never').launch(debug=True, share=True)

from flask import Flask, request, jsonify
from fastai.vision.all import *
import pathlib   
temp = pathlib.PosixPath   
pathlib.PosixPath = pathlib.WindowsPath

# Load the Fastai model
learn_emotion = load_learner('emotion_model.pkl')
learn_emotion_labels = learn_emotion.dls.vocab

# Create the Flask app
app = Flask(__name__)

# Define a route for the API
@app.route('/predict-emotion', methods=['POST'])
def predict_emotion():
    # Get the image file from the request
    img_file = request.files['image']
    
    # Open the image file
    img = PILImage.create(img_file)
    
    # Make a prediction
    pred_emotion, pred_emotion_idx, probs_emotion = learn_emotion.predict(img)
    
    # Format the prediction
    emotions = {learn_emotion_labels[i]: float(probs_emotion[i]) for i in range(len(learn_emotion_labels))}
    
    # Return the prediction as JSON
    return jsonify(emotions)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, )

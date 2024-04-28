from flask import Flask, render_template, request
from lime import lime_image
import numpy as np
from PIL import Image
from skimage.segmentation import mark_boundaries
import tensorflow as tf

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('efficientnet_model.h5')

# Initialize the LIME explainer
explainer_lime = lime_image.LimeImageExplainer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image
    uploaded_image = request.files['image']
    # Read the image
    image = Image.open(uploaded_image)
    # Preprocess the image
    image = preprocess_image(image)
    # Perform prediction using the model
    prediction = model.predict(np.expand_dims(image, axis=0))
    
    # Explain the prediction using LIME
    explanation_lime = explain_with_lime(image)
    
    return render_template('result.html', 
                           prediction=prediction, 
                           explanation_lime=explanation_lime)

def preprocess_image(image):
    image = image.resize((256, 256))  # Resize to match model input size
    image = np.array(image) / 255.0  # Normalize
    return image

def explain_with_lime(image):
    explanation_lime = explainer_lime.explain_instance(image, model.predict, top_labels=5, hide_color=0, num_samples=1000)
    temp, mask = explanation_lime.get_image_and_mask(explanation_lime.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    explanation_img = mark_boundaries(temp / 2 + 0.5, mask)
    return explanation_img

if __name__ == '__main__':
    app.run(debug=True)

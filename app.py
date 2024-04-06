from flask import Flask, render_template, request

from flask_caching import Cache

import numpy as np

import os
import matplotlib.pyplot as plt

from PIL import Image
from tensorflow.keras.preprocessing import image

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions

import tensorflow as tf

model_path = 'KAGGLEbone_marrow_cell_classification_13classes_0acc_0v.acc.h5'

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

app = Flask(__name__)
cache = Cache(app)

class_names = ["ART", "BLA", "EBO", "EOS", "LYT", "MMZ", "MON", "MYB", "NGS", "NIF", "PEB", "PLM", "PMO"]


def generate_prediction_report(folder_path, model, class_names):
    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

    # Sort the image files by size (in bytes)
    image_files.sort(key=lambda f: os.path.getsize(os.path.join(folder_path, f)))

    # Initialize a dictionary to store the count of predicted classes
    class_count = {class_name: 0 for class_name in class_names}
    
    # Iterate over the images in the folder
    for filename in image_files:
        img_path = os.path.join(folder_path, filename)

        # Load and preprocess the image
        img = Image.open(img_path)
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize the image

        # Predict the classes
        preds = model.predict(img_array)
        predicted_class_index = np.argmax(preds, axis=1)[0]
        predicted_class = class_names[predicted_class_index]

        # Increment the count for the predicted class
        class_count[predicted_class] += 1

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(class_count.keys(), class_count.values(), color='skyblue', alpha=0.7)
    plt.xlabel('Predicted Class')
    plt.ylabel('Count')
    plt.title('Distribution of Predicted Classes')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('images/graph/prediction_report.jpg')  # Save the graph image



# Route for the graph page
@app.route('/graph', methods=['GET', 'POST'])
def graph():
    if request.method == 'POST':
        
        # Save the uploaded files
        for file in request.files.getlist('imagefile'):
            if file.filename != '':
                file.save(os.path.join('images/uploads', file.filename))

        # Generate the prediction report
        generate_prediction_report('images/uploads', model, class_names)
        
        return render_template('graph.html', prediction_report='images/graph/prediction_report.jpg')
    else:
        # Render the graph template for GET requests
        return render_template('graph.html')



@app.route('/', methods = ['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)
    
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    
    #image = preprocess_input(image)
    #image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    
    image = np.expand_dims(image, axis=0)
    cell_image = image/255.0 #Normalize the image
    
    #Make predictions
    
    #yhat = model.predict(cell_image)
    #label = decode_predictions(yhat) #compatibility problem :/ model <-/->current app's requirements/version
    #label = label[0][0]
    
    predictions = model.predict(cell_image)
    
    print(predictions)
    
    #classification = '%s (%.2f%%)' % (label[1], label[2]*100) #FOR SIMPLE KERAS CNN MODELS
    
    #Classification for our custom model:
    
    
    # Get the index of the class with the highest probability
    top_predicted_class_idx = np.argmax(predictions[0])
    
    # Get the class name corresponding to the top predicted class
    class_name = class_names[top_predicted_class_idx]

    # Get the confidence in percentage for the top predicted class
    confidence = predictions[0][top_predicted_class_idx] * 100

    # Construct the prediction statement
    prediction_statement = f" {class_name} ({confidence:.2f}%)"
    
    
    return render_template('index.html', imagefile=imagefile, prediction=prediction_statement)
   
   


# Route for the index page
@app.route('/',  methods = ['GET'])
def load_index():
    return render_template('index.html')



if __name__ == '__main__':
    app. run(debug=True)
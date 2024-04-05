from flask import Flask, render_template, request

from flask_caching import Cache

import numpy as np

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


@app.route('/',  methods = ['GET'])
def load_index():
    return render_template('index.html')


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
    
    class_names = ["ART", "BLA", "EBO", "EOS", "LYT", "MMZ", "MON", "MYB", "NGS", "NIF", "PEB", "PLM", "PMO"]
    
    # Get the index of the class with the highest probability
    top_predicted_class_idx = np.argmax(predictions[0])
    
    # Get the class name corresponding to the top predicted class
    class_name = class_names[top_predicted_class_idx]

    # Get the confidence in percentage for the top predicted class
    confidence = predictions[0][top_predicted_class_idx] * 100

    # Construct the prediction statement
    prediction_statement = f" {class_name} ({confidence:.2f}%)"
    
    
    return render_template('index.html', prediction=prediction_statement)
    



if __name__ == '__main__':
    app. run(port = 5000, debug=True)
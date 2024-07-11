from flask import Flask, render_template, request
from keras.src.utils.module_utils import tensorflow

from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)
model =tensorflow.keras.models.load_model("C:/Users/Gobi/Downloads/Brain_Tumor.h5")


@app.route('/', methods=['Get'])
def hello_world():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    img = image.load_img(image_path, target_size=(256, 256))

    # Convert the image to a numpy array
    img_array = image.img_to_array(img)

    # Expand the dimensions to match the input shape expected by the model
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess the image (same as how you preprocessed the training data)
    img_array = img_array / 255.0  # Rescale pixel values to [0, 1]

    # Make predictions
    prediction = model.predict(img_array)

    class_index = np.argmax(prediction)

    confidence = prediction[0][class_index] * 100

    # Define the class labels (update these labels according to your model's output)
    class_labels = [
        'Astrocitoma T1', 'Astrocitoma T1C+', 'Astrocitoma T2',
        'Carcinoma T1', 'Carcinoma T1C+', 'Carcinoma T2',
        'Ependimoma T1', 'Ependimoma T1C+', 'Ependimoma T2',
        'Ganglioglioma T1', 'Ganglioglioma T1C+', 'Ganglioglioma T2',
        'Germinoma T1', 'Germinoma T1C+', 'Germinoma T2',
        'Glioblastoma T1', 'Glioblastoma T1C+', 'Glioblastoma T2',
        'Granuloma T1', 'Granuloma T1C+', 'Granuloma T2',
        'Meduloblastoma T1', 'Meduloblastoma T1C+', 'Meduloblastoma T2',
        'Meningioma T1', 'Meningioma T1C+', 'Meningioma T2',
        'Neurocitoma T1', 'Neurocitoma T1C+', 'Neurocitoma T2',
        'Oligodendroglioma T1', 'Oligodendroglioma T1C+', 'Oligodendroglioma T2',
        'Papiloma T1', 'Papiloma T1C+', 'Papiloma T2',
        'Schwannoma T1', 'Schwannoma T1C+', 'Schwannoma T2',
        'Tuberculoma T1', 'Tuberculoma T1C+', 'Tuberculoma T2',
        '_NORMAL T1', '_NORMAL T2'
    ]

    # Get the predicted class label
    predicted_label = class_labels[class_index]

    return render_template('index.html', prediction=predicted_label, confidence=confidence)


if __name__ == '__main__':
    app.run(port=3000, debug=True)
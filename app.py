import numpy
from tensorflow.keras.models import load_model
import keras
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

app = Flask(__name__, template_folder='./templates/', static_folder='./templates/static')
classes = ['Bougainvillea' ,
           'Daisies', 
           'Garden Roses' , 
           'Gardenias' , 
           'Hibiscus' ,
           'Hydra' ,
           'Lilies', 
           'Orchid' ,
           'Peonies' ,
           'Tulips']

@app.route('/')
def home():
    return render_template("index.html")

def load_model_class(path:str)->keras.engine.sequential.Sequential:
    model = load_model(path)

    return model

def load_image(img_file_path:str)->numpy.ndarray:
    img_shape = (150,150)
    img = cv2.imread(img_file_path)
    if img is None:
        print("No Image :(")
    resized_img = cv2.resize(img,img_shape)
    re_img = numpy.asarray(resized_img)
    img_batch = numpy.expand_dims(re_img,axis=0)
    return img_batch

def predict_result(result:numpy.ndarray)->str:
    dict_result = {}
    for i in range(10):
        dict_result[result[0][i]] = classes[i]
    res = result[0]
    res.sort()
    res = res[::-1]
    prob = res[:3]
    
    prob_result = []
    class_result = []
    for i in range(3):
        prob_result.append((prob[i]*100).round(2))
        class_result.append(dict_result[prob[i]])
    
    return class_result,prob_result

def get_image (req_met:str) -> str:
    predict_img_path="./templates/static/image"
    if req_met == "POST":
        image = request.files['file']
        file_name = secure_filename(image.filename)
        image.save(os.path.join(predict_img_path,file_name))
        
        return os.path.join(predict_img_path,file_name),file_name

@app.route('/predict', methods=['POST'])
def predict() -> render_template:
    file_path = "./model/flower-classifier.h5"
    uploaded_img_path, filename = get_image(request.method)
    image = load_image(uploaded_img_path)
    model = load_model_class(file_path)
    result = model.predict(image)
    class_result, prob_result = predict_result(result)

    return render_template("res_page.html",result=class_result,prob=prob_result,img=filename)  


if __name__ == "__main__":
    app.run(debug=True)
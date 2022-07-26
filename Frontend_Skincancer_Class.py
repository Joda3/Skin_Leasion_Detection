from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
#from settings import PROJECT_ROOT
import os

try:
	import shutil
	shutil.rmtree('uploaded/image')
	os.chdir('C:/Users/proxi/PycharmProjects/Aktuelle DS/uploaded')
	os.mkdir("C:/Users/proxi/PycharmProjects/Aktuelle DS/uploaded/images")
	os.chdir("C:/Users/proxi/PycharmProjects/Aktuelle DS/uploaded")
	print()
except:
	pass





mein_model="C:/Users/proxi/PycharmProjects/Aktuelle DS/model/model-2.h5"
mein_model_opt="C:/Users/proxi/PycharmProjects/Aktuelle DS/model/mode_optl.h5"


model = tf.keras.models.load_model(mein_model)
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'C:/Users/proxi/PycharmProjects/Aktuelle DS/uploaded/images/'

@app.route('/')
def upload_f():
	print("upload")
	return render_template('upload.html')

def finds():
	test_datagen = ImageDataGenerator(rescale = 1./255)
	print("find")
	vals = ['Actinic keratoses',
              'Basal cell carcinoma',
              'Benign keratosis-like lesions ', 
              'Dermatofibroma', 
              'Melanocytic nevi', 
              'Melanoma',
              'Vascular lesions']

	test_dir = 'C:/Users/proxi/PycharmProjects/Aktuelle DS/uploaded/'
	test_generator = test_datagen.flow_from_directory(
			test_dir,
			target_size =(224, 224),
			color_mode ="rgb",
			shuffle = False,
			class_mode='categorical',
			batch_size = 1)

	pred = model.predict(test_generator)
	pred = pred[0]
	print(len(pred))
	print(pred)
	print([np.argmax(pred)])
	return str(vals[np.argmax(pred)])


@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		f = request.files['file']
		f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
		val = finds()
		return render_template('pred.html', ss = val)

if __name__ == '__main__':
	app.run()

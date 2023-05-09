from flask import Flask, render_template, request
from keras.models import load_model
#from flask_ngrok import run_with_ngrok
# from keras.preprocessing import image
from keras.utils import load_img, img_to_array
import numpy as np

app = Flask(__name__)
#run_with_ngrok(app)
dic = {0 : 'RDR', 1 : 'NRDR'}

model = load_model('model/mobilenetv2.h5')

model.make_predict_function()

# def predict_label(img_path):
# 	i = load_img(img_path, target_size=(224,224))
# 	i = i / 255.0
# 	i = np.array(i)
# 	i = i.reshape(1, 224,224,3)
# 	p = model.predict(i)
# 	return dic[p[0]]

def predict_label(img_path):
	img=load_img(img_path, target_size=(224,224))
	img = np.array(img)
	img = img / 255.0
	img = img.reshape(1,224,224,3)
	label = model.predict(img)
	if label[0][0]>0.5:
		print(label[0][0])
		return "RDR"
	else:
		print(label[0][0])
		return "NRDR"

   
# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Please subscribe Artificial Intelligence Hub..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']
		img_path = "static/" + img.filename	
		img.save(img_path)

		p= predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug=True)
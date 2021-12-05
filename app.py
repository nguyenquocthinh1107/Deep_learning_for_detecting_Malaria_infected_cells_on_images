from flask import Flask,render_template,request
from flask_cors import cross_origin
from detection import Decode, detectWithYOLOv4, detectWithResnet, detectWithUNET
app = Flask(__name__)

@app.route('/',methods=['GET'])
@cross_origin()
def index():
	return render_template('index.html')

@app.route('/result',methods=['GET','POST'])
@cross_origin()
def result():
	if request.method == 'POST':
		image = request.json['image']
		model = int(request.json['model'])
		img = Decode(image).copy()

		# print("MODEL: ", model)

		if model == 1:
			detectWithYOLOv4(img)
		elif model == 2:
			detectWithResnet(img)
		elif model == 3:
			detectWithUNET(img)

		return render_template('index.html')
	return render_template('index.html')

if __name__=="__main__":
	app.run(host="0.0.0.0",port="5000")


from flask import Flask, jsonify, render_template, request
app = Flask (__name__)

@app.route('/')
def index():
	return render_template('jumbled.html')

@app.route('/JumbledAppendToFile')
def JumbledAppendToFile(fileName):
	
	print("JUMBLE APPEND TO FILE IS WORKING")
	return "Nothing"

if __name__ == '__main__':
	app.run()

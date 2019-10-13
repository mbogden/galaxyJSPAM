from os import \
        path, \
        listdir

from flask import \
        Flask, \
        render_template, \
        send_file

app = Flask(__name__)

@app.route("/")
def index():
    #open file, read contents
    with open ("test.txt","rt") as myfile:
        for myline in myfile:

            #find sdss
            location = myline.find("sdss_name")
            if location != -1:
                sdss = myline[location+1:]

    #read from string to find sdss, gen#, run#, and humanScore 
    textdata = "TEST"
    imgName = 'test.png'
    message = "GALAXY DISPLAY TEXT"
    page_contents = render_template("displayGalaxy.html", imgName = imgName, message = textdata)
    return page_contents

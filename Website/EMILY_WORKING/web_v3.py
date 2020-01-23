from os import \
        path, \
        listdir

from flask import \
        Flask, \
        render_template, \
        send_file, \
	Response, \
	request, \
	redirect, \
	url_for

app = Flask(__name__)

@app.route('/background')
def background():
    print("Hello from background")
    return 'nothing'

@app.route("/",methods=['GET','POST'])  # change

def IterateFiles():


    runLoc = "test_data_SE/587722984435351614/"
    sLoc = "static/" + runLoc
    fileName = sLoc + "run_0_0/info.txt"
    imgLocation = runLoc + 'run_0_0/param_1_model.png'
    pageContents = runBlock(fileName, imgLocation)

    for fileRunNumber in range(1,10):
        fileName = sLoc + "run_0_%d/info.txt" % fileRunNumber
        imgLocation = runLoc + 'run_0_%d/param_1_model.png' % fileRunNumber
        pageContents += runBlock(fileName, imgLocation)
        #jumbled.JumbledAppendToFile(fileName)

    if request.method == "POST":

        print('Post: ', request.form)
        #print("Run: %s - %s" % ( request.form[0], request.form[1] ) )

   
    return pageContents

def runBlock(fileName, imgLocation):
    #open file, read contents
    with open (fileName,"rt") as myfile:

        #Create variables
        sdss = ""
        genNumber = 0
        runNumber = 0
        humanScore = 0.0

        for myline in myfile:
            
            if "sdss_name" in myline:
                sdss = myline.split()[1].strip()
            elif "generation" in myline:
                genNumber = int(myline.split()[1].strip()) 
            elif "run_number" in myline:
                runNumber = int(myline.split()[1].strip())
            elif "human_score" in myline:
                humanScore = float(myline.split()[1].strip())

    # end going through file
    

    #read from string to find sdss, gen#, run#, and humanScore  
    page_contents = render_template("displayGalaxy_2.html", imgLoc = imgLocation, sdss = sdss, genNumber = genNumber, runNumber = runNumber, humanScore = humanScore) 
    return page_contents

def PrintInfo(fileName, imgLocation):
    #open file, read contents
    with open (fileName,"rt") as myfile:

        #Create variables
        sdss = ""
        genNumber = 0
        runNumber = 0
        humanScore = 0.0

        for myline in myfile:
            
            if "sdss_name" in myline:
                sdss = myline.split()[1].strip()
            elif "generation" in myline:
                genNumber = int(myline.split()[1].strip()) 
            elif "run_number" in myline:
                runNumber = int(myline.split()[1].strip())
            elif "human_score" in myline:
                humanScore = float(myline.split()[1].strip())
    

    #read from string to find sdss, gen#, run#, and humanScore  
    page_contents = render_template("displayGalaxy.html", imgLoc = imgLocation, sdss = sdss, genNumber = genNumber, runNumber = runNumber, humanScore = humanScore) 
    return page_contents


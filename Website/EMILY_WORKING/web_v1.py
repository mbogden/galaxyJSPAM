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
    return "Hello From Main"


@app.route("/test1")
def test1():
    mes  = '1: Hello from test\n'
    mes += '2: New line char \\n does not work'  # newline does not work
    mes += '<br> 3: use html newline\n'

    mes += '<br>4: Testing loop<br>'
    for i in range(3):
        mes += '\tloop %d<br>' % i # Note tabs \t do not work in html

    mes += 'Example %3d with string %s' % (34 , "'some sting'") 
    return mes


@app.route("/test2")
def test2():

    imgLoc = 'test.png'
    mes = send_file(imgLoc)
    print("img stuff below")
    print(mes)

    return mes


@app.route("/test3")
def test3():
    strMes = "hello from test 3"
    return render_template("string.html", message=strMes)


@app.route("/test4")
def test4():
    page_contents = render_template("string.html", message="message 1 to template")
    page_contents += render_template("string.html", message="message 2 to template")
    return page_contents

@app.route("/test5")

def test5():
    imgName = 'test.png'
    page_contents = render_template("img.html", imgName = imgName)

    return page_contents


if __name__ == '__main__':
    print("Hello ")
    app.run(debug=True)


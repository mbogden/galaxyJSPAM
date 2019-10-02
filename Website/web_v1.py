from flask import \
        Flask, \
        render_template

app = Flask(__name__)


@app.route("/")

def hello():
    message = "Hello\n\n"
    return message

def soo():
    mes = "sooooo....\n\n"
    return render_template('index.html', message=mes)


if __name__ == '__main__':
    print("Hello ")
    app.run(debug=True)


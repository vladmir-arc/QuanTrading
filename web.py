import flask
from flask import Flask, render_template, request

app = Flask(__name__, template_folder='templates')


@app.route('/')
def index():
    return 'Hello World!'


@app.route('/user/<name>')
def user(name):
    return render_template('user.html', name=name)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        return flask.redirect('/')
    else:
        return render_template('login.html')


if __name__ == '__main__':
    app.run(debug=True)

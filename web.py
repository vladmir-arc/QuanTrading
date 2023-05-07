import flask
from test import sum_2
from LSTM import main, get_stocks
from flask import Flask, render_template, request


app = Flask(__name__, template_folder='templates')


@app.route('/')
def index():
    return '''
            <form method="POST" action="/calculate">
                <label for="num1">Enter first number:</label>
                <input type="text" id="num1" name="num1"><br><br>
                <label for="num2">Enter second number:</label>
                <input type="text" id="num2" name="num2"><br><br>
                <input type="submit" value="Calculate">
            </form>
        '''


@app.route('/stock_list')
def stock_list():
    slist = get_stocks('Database')
    stock_EtoC = {'ShanMeiGuoJi': ['山煤国际', '600546.SS'],
                  'PetroChina': ['中国石油', '601857.SS'],
                  'ZhongGuoShenHua': ['中国神华', '601088.SS'],
                  'ZhongMeiNengYuan': ['中煤能源', '601898.SS'],
                  'ShanXiMeiYe': ['陕西煤业', '601225.SS'],
                  'PetroChemical': ['中国石化', '600028.SS']}

    length = len(slist)
    return render_template('stock_list.html', list=slist, dict=stock_EtoC, length=length)


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


@app.route('/calculate', methods=['POST'])
def calculate():
    num1 = int(request.form['num1'])
    num2 = int(request.form['num2'])
    result = sum_2(num1, num2)
    return f'The sum of {num1} and {num2} is {result}.'


if __name__ == '__main__':
    app.run(debug=True)

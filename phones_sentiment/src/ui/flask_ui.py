import os

from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import SubmitField, TextAreaField
from wtforms.validators import DataRequired

from estimator.model_predictor import ModelPredictor

# SENTIMENT_URL = 'http://127.0.0.1:80/estimator'

app = Flask(__name__)
bootstrap = Bootstrap(app)
model = ModelPredictor('./model')

app.config['SECRET_KEY'] = os.urandom(32)

label_classes = {-1: 'default', 0: 'danger', 1: 'warning', 2: 'info', 3: 'primary', 4: 'success'}
label_texts = {-1: 'None', 0: 'Certain negative', 1: 'Probably negative',
               2: 'Neutral', 3: 'Probably positive', 4: 'Certain positive'}


class EstimatorForm(FlaskForm):
    text = TextAreaField('', render_kw={'rows': 10},  validators=[DataRequired()])
    submit = SubmitField('Classify')


@app.route('/estimator', methods=['POST', 'GET'])
def index_page(text='', prediction_message=''):
    if request.method == "POST":
        text = request.form["text"]
        pred = model.predict(text)
        label_class = label_classes[pred]
        label_text = label_texts[pred]
    else:
        text = ''
        label_class = 'default'
        label_text = 'None'

    return render_template('estimator.html', form=EstimatorForm(),
                           label_class=label_class, label_text=label_text, text=text)


@app.route('/about', methods=['GET'])
def bootstrap_test_page():
    return render_template('about.html')


if __name__ == '__main__':
    # threading.Thread(target=open_browser).start()
    app.run(host='0.0.0.0', port=80, debug=False)

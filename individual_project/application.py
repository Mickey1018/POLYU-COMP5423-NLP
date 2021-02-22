from flask import Flask, render_template, request
from emotion_classification import *
from text_processor import *
from feature_extraction import *
from classification_model import *
app = Flask(__name__)


@app.route('/')
def get_sentence():
   return render_template('predict.html')


@app.route('/result', methods=['POST', 'GET'])
def result():
   if request.method == 'POST':
      sentence = request.form['Emotion Predicted']
      processed_sentece = text_processing([sentence, ' '])
      features_sentence = extract_features(processed_sentece)
      print(features_sentence)
      model = pickle.load(open('trained_model.sav', 'rb'))
      result = predict_emotion(model, features_sentence)[0]
      return render_template("result.html", result=result)


if __name__ == '__main__':
   app.run(debug=True)



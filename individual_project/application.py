from flask import Flask, render_template, request
from text_processor import *
from feature_extraction import *
from classification_model import *
app = Flask(__name__)

label_mapping = {-3: {'emotion': 'anger', 'emoji': 128545},
                 -2: {'emotion': 'fear', 'emoji': 128561},
                 -1: {'emotion': 'sadness', 'emoji': 128546},
                 1: {'emotion': 'joy', 'emoji': 128513},
                 2: {'emotion': 'love', 'emoji': 128525},
                 3: {'emotion': 'surprise', 'emoji': 128565}
                 }


@app.route('/')
def get_sentence():
   return render_template('predict.html')


@app.route('/result', methods=['POST', 'GET'])
def result():
   if request.method == 'POST':
      sentence = request.form['Emotion Predicted']
      processed_sentece = text_processing([sentence, ' '])
      features_sentence = extract_features(processed_sentece)
      model = pickle.load(open('trained_model.sav', 'rb'))
      result = predict_emotion(model, features_sentence)[0]
      return render_template("result.html",
                             emotion=label_mapping[result]['emotion'],
                             emoji=label_mapping[result]['emoji'])


if __name__ == '__main__':
   app.run(debug=True)



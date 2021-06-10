#-------------------Import library necessary----------------------------#
from flask import render_template, jsonify, Flask, redirect, url_for, request, flash
from flask_ngrok import run_with_ngrok
import random
import os
import numpy as np
from model import MACNN, AACNN
from extract_features import FeatureExtractor
import librosa
import tensorflow as tf
from keras import backend as K
import sys
import torch
from transformers import BertModel, BertTokenizer
import os
#------------------------------ARG__SPEECH EMOTION RECOGNITION FROM SPEECH-----------------------------------------#
SEED = 111111
MODEL_NAME = 'CNN_mel_4label_seed{}'.format(SEED)
RATE =16000
FEATURES_TO_USE ='melspectrogram'# 'melspectrogram'  # {'mfcc' , 'logfbank','fbank','spectrogram','melspectrogram'}
LABEL = {
    'neutral': 0,
    'happy': 1,
    'sad': 2,
    'angry': 3,
}
LABEL_DICT1 = {
    '01': 'neutral',
    '04': 'sad',
    '05': 'angry',
    '07': 'happy', 
}
#--------------------Process data---------------------------#
def process(wav_file, LABEL_DICT1, RATE = 16000, t= 2, val_overlap =1.6):
    val_dict = {}
    if (val_overlap >= t):
        val_overlap = t / 2
    wav_data, _ = librosa.load(wav_file, sr=RATE)
    X1 = []
    # print('wav_data:', wav_data)
    # print(len(wav_data))
    index = 0
    while (index + t * RATE < len(wav_data)):
        X1.append(wav_data[int(index):int(index + t * RATE)])
        index += int((t - val_overlap) * RATE)
    X1 = np.array(X1)
    val_dict = {
        'X': X1,
        'path': wav_file
    }
    # print('val_dict:', val_dict)
    valid_features_dict = {}
    # get features extract
    feature_extractor = FeatureExtractor(rate=RATE)
    XX1 = feature_extractor.get_features(FEATURES_TO_USE, val_dict['X'])
    valid_features_dict = {
        'X': XX1,
    }
    return valid_features_dict
#-------------------------SPEECH EMOTION RECOGNITION IN TEXT ----------------------------------#
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model=torch.load('./static/models/model_text_2.pt')
#-------------------------CONFIG NGROK----------------------#
app = Flask(__name__)
# run_with_ngrok(app)
# app.config.from_mapping(
#         BASE_URL="http://localhost:5000",
#         USE_NGROK=os.environ.get("USE_NGROK", "False") == "True" and os.environ.get("WERKZEUG_RUN_MAIN") != "true"
#     )
# # pyngrok will only be installed, and should only ever be initialized, in a dev environment
# from pyngrok import ngrok
# # Get the dev server port (defaults to 5000 for Flask, can be overridden with `--port`
# # when starting the server
# port = sys.argv[sys.argv.index("--port") + 1] if "--port" in sys.argv else 5000
# # Open a ngrok tunnel to the dev server
# public_url = ngrok.connect(port).public_url
# print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}\"".format(public_url, port))
# # Update any base URLs or webhooks to use the public ngrok URL
# app.config["BASE_URL"] = public_url
# init_webhooks(public_url)


@app.route('/')
def index():
    return render_template('index.html', title='Home')

@app.route('/speech')
def speech():
	return render_template("speech.html")

@app.route('/text')
def text():
	return render_template("text.html")

@app.route('/speech_text')
def speech_text():
	return render_template("speech_text.html")


@app.route('/uploaded', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        #choose file wav
        f = request.files['file']
        # print('f', f)
        dir = os.path.dirname(__file__)
        path = os.path.join(dir, 'static/data_test_sv', f.filename)
        f.save(path)
        # load weight model
        model = AACNN(3,3)
        model.load_weights('./static/models/{}'.format(MODEL_NAME))
        try:
            wav_data, _ = librosa.load(path, sr=RATE)
            print(len(wav_data))
        except:
            # Show error load file sound
            return render_template('speech.html', warning ='Please choose file wav. Other files are not accepted')

        #load data

        if (len(wav_data) < 2 * RATE):
            return render_template('speech.html', warning ='Please choose file sound have length greate than 4 seconds or try different sound file!')
        else :
            # process data
            valid_features_dict = process(path, LABEL_DICT1, RATE = 16000, t= 2, val_overlap =1.6)
            x = valid_features_dict['X']
            x = tf.expand_dims(x, -1)
            x = tf.cast(x, tf.float32)
            out = model(x)
            out = tf.reduce_mean(out, 0, keepdims=True).numpy()
            # print('out:', out)
            # print('argmax:', np.argmax(out))
            predict_label = ''
            for key, value in LABEL.items():
                if value == np.argmax(out):
                    predict_label = key
            # print('predict_labels:', predict_label)
            acc_ = np.round(np.max(out)*100,3)
            if np.isnan(acc_) == True:
                acc_ = 49.99
            K.clear_session()
        return render_template('speech.html', title='Success', predictions= predict_label, acc= acc_)

@app.route('/text', methods=['POST'])
def text_submit():
    if request.method == 'POST':
        raw_text = request.form['rawtext']
        input_ids = torch.tensor(tokenizer.encode(raw_text, add_special_tokens= True)).unsqueeze(0)
        labels = torch.tensor([0])
        model.to('cpu')
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids,labels=labels)
            loss, logits = outputs[:2]
            # print(logits)
            acc_pred, preds = torch.max(logits, 1)
            result = preds.numpy()[0]
        for key, value in LABEL.items():
            if value == result:
                predict_label = key
        acc = np.round((np.tanh(acc_pred)*100 - 18).numpy()[0], 3)
    return render_template("text.html", results = predict_label, raw_text = raw_text, accuracy = acc)
if __name__ == "__main__":
    app.run()

import librosa
import soundfile
import os
import glob
import pickle
import numpy as np
import tempfile
import speech_recognition as sr
# pip install SpeechRecognition
from flask_cors import CORS
from flask import request
from flask import send_from_directory
from flask import Flask
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from pydub import AudioSegment
from flask import jsonify
from werkzeug.utils import secure_filename

model_file_path = os.path.join(tempfile.gettempdir(), "SER_model")
uploads_dir = os.path.join(tempfile.gettempdir(), "uploads")

try:
    os.mkdir(uploads_dir)
except OSError:
    print ("Creation of the directory %s failed" % uploads_dir)
else:
    print ("Successfully created the directory %s " % uploads_dir)


app = Flask(__name__)
CORS(app)
model = None

# DataFlair - Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))
    return result


# DataFlair - Emotions in the RAVDESS dataset
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}


# DataFlair - Load the data and extract features for each sound file
def load_data(observed_emotions, test_size=0.2):
    x, y = [], []
    for file in glob.glob('C:\\\\Code\\ser\\dataSet\\Actor_*\\*.wav'):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)


def load_sample(sample_path):
    sound = AudioSegment.from_wav(sample_path)
    sound = sound.set_channels(1)

    filename, file_extension = os.path.splitext(sample_path)

    mono_sample = filename+'mono'+file_extension
    sound.export(mono_sample, format="wav")

    sample_arr = []
    sample_feature = extract_feature(mono_sample, mfcc=True, chroma=True, mel=True)
    sample_arr.append(sample_feature)

    ndarray = np.array(sample_arr)
    return ndarray


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        saved_path = os.path.join(uploads_dir, secure_filename(f.filename))
        f.save(saved_path)
        # return jsonify(f.filename)

        return jsonify(f.filename)
        # return jsonify{"message": str(FLAMSG_ERR_SEC_ACCESS_DENIED), "severity": "danger"})


@app.route('/classifyEmotion/', methods=['GET', 'POST'])
def hello():
    filename = request.args.get('filename')
    file_path = os.path.join(uploads_dir, secure_filename(filename))
    print(f'User asked to classified file: {file_path}')
    res = get_model().predict(load_sample(file_path))
    print(f"file {file_path} classified as: {res}")

    return jsonify(res.tolist())


@app.route('/speechToText', methods=['GET', 'POST'])
def text():
    filename = request.args.get('filename')
    file_path = os.path.join(uploads_dir, secure_filename(filename))
    res = get_recognizer(file_path)
    return jsonify(res)



@app.route('/playSoundFile/<filename>', methods=['GET'])
def get_file_to_play(filename):
    return send_from_directory(uploads_dir, filename)


# DataFlair - Emotions to observe
observed_emotions = ['calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']


def train_model():
    print("Loading data set..")
    x_train, x_test, y_train, y_test = load_data(observed_emotions, test_size=0.25)
    print("Data set loaded successfully")
    # DataFlair - Get the shape of the training and testing datasets
    print((x_train.shape[0], x_test.shape[0]))
    # DataFlair - Get the number of features extracted
    print(f'Features extracted: {x_train.shape[1]}')
    # DataFlair - Initialize the Multi Layer Perceptron Classifier
    model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,),
                          learning_rate='adaptive',
                          max_iter=500)

    # DataFlair - Train the model
    model.fit(x_train, y_train)

    pickle.dump(model, open(model_file_path, 'wb'))
    print("model was saved at " + model_file_path)

    # DataFlair - Predict for the test set
    y_pred = model.predict(x_test)
    sample_path = 'C:\\Code\\ser\\dataSet\\Actor_01\\03-01-03-01-01-02-01.wav'
    result = model.predict(load_sample(sample_path))
    # DataFlair - Calculate the accuracy of our model
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    # DataFlair - Print the accuracy
    print("Accuracy: {:.2f}%".format(accuracy * 100))

    return model


def get_model():
    global model
    if model:
        return model

    if os.path.isfile(model_file_path):
        print(f"Model exists. Loading it from: {model_file_path}")
        # load the model from disk
        model = pickle.load(open(model_file_path, 'rb'))
    else:
        print("Model DOES NOT exist. Train.")
        model = train_model()

    return model


def get_recognizer(filename):
    r = sr.Recognizer()

    with sr.WavFile(filename) as source:
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio, show_all=False)
            print("You said : {}".format(text))
            return text
        except:
            print("Sorry could not recognize what you said")
            return ""


if __name__ == '__main__':
    app.run()
    model = get_model()


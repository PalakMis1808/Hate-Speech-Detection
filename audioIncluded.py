# importing all necessary libraries
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import nltk
import speech_recognition as sr
import re
from nltk.corpus import stopwords

stopword = set(stopwords.words('english'))

# stemming algorithm being used
stemmer = nltk.SnowballStemmer("english")

# Load data

data = pd.read_csv("E:/E drive Downloads/labeled_data.csv (1)/labeled_data.csv")


# Map labels to human-readable categories
data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive Speech", 2: "No Hate and Offensive Speech"})

data = data[["tweet", "labels"]]

# Data cleaning function
def clean(text):
    #convert text to lowercase
    text = str(text).lower()
    # Remove periods and question marks
    text = re.sub('[.?]','',text)
    #Remove Urls
    text = re.sub('https?://\S+|www.\S+', '', text)
    #Remove HTML tags
    text = re.sub('<.?>+','',text)
    #Remove non-alphanumeric characters
    text = re.sub(r'[^\w\s]','',text)
    #Remove newline characters
    text = re.sub('\n','',text)
    #Remove alphanumeric words
    text = re.sub('\w\d\w','',text)
    #Remove stopwords
    text = [word for word in text.split(' ') if word not in stopword]
    #join words in a single string
    text = " ".join(text)
    #apply stemming
    text = [stemmer.stem(word) for word in text.split(' ')]
    #join the stemmed words back into a single string
    text = " ".join(text)
    return text


# Apply data cleaning function
data["tweet"] = data["tweet"].apply(clean)

# Features and labels
x = np.array(data["tweet"])
y = np.array(data["labels"])

# Vectorize the text data
cv = CountVectorizer()
X  = cv.fit_transform(x)




# Split the data into training and testing sets
X_train, X_text, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_text)

# Print accuracy
from sklearn.metrics import accuracy_score
print("The accuracy is", accuracy_score(y_test, y_pred))

# Classification report
from sklearn.metrics import classification_report
print("The classification report is", classification_report(y_test, y_pred))


# Print accuracy
from sklearn.metrics import accuracy_score
print("The accuracy is", accuracy_score(y_test, y_pred))

# Classification report
from sklearn.metrics import classification_report
print("The classification report is", classification_report(y_test, y_pred))

# Flask app
app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('indexHate.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'audio' in request.files:
            # ... (rest of the code remains unchanged)

            audio_file = request.files['audio']
            recognizer = sr.Recognizer()

            try:
                with sr.AudioFile(audio_file) as source:
                    audio = recognizer.record(source)

                input_text = recognizer.recognize_google(audio)
                print("Input Text:", input_text)

                # Clean and vectorize the input text
                cleaned_text = clean(input_text)
                # Vectorize the cleaned input text
                input_text_vectorized = cv.transform([cleaned_text]).toarray()

                # Map the prediction to a human-readable label
                pred_label = model.predict(input_text_vectorized)[0]

                if pred_label == "Hate Speech":
                    result_label = "Hate Speech"
                elif pred_label == "Offensive Speech":
                    result_label = "Offensive Speech"
                else:
                    result_label = "No Hate and Offensive Speech"

                print("Prediction Label:", result_label)

                return render_template('indexHate.html', prediction=result_label, input_text=input_text)

            except sr.UnknownValueError:
                return render_template('indexHate.html', prediction="Error: Could not understand audio.")
            except sr.RequestError as e:
                return render_template('indexHate.html', prediction=f"Error: Could not request results from Google Speech Recognition service; {e}")

        else:
            input_text = request.form['sentence']
            print("Input Text:", input_text)

             # Clean and vectorize the input text
            cleaned_text2 = clean(input_text)
            # Vectorize the cleaned input text
            input_text_vectorized2 = cv.transform([cleaned_text2]).toarray()

            # Map the prediction to a human-readable label
            pred_label = model.predict(input_text_vectorized2)[0]

            if pred_label == "Hate Speech":
                result_label = "Hate Speech"
            elif pred_label == "Offensive Speech":
                result_label = "Offensive Speech"
            else:
                result_label = "No Hate and Offensive Speech"

            print("Prediction Label:", result_label)

            return render_template('indexHate.html', prediction=result_label, input_text=input_text)

if __name__ == "__main__":
    app.run(debug=True)

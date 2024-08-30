# importing all necessary libraries

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
# Natural Language Toolkit
import nltk

from nltk.corpus import stopwords
stopword = set(stopwords.words('english'))

# stemming algorithm being used

stemmer = nltk.SnowballStemmer("english")

# Load data

data = pd.read_csv("E:/E drive Downloads/labeled_data.csv (1)/labeled_data.csv")
print(data.head())

# Map labels to human-readable categories

data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive Speech", 2: "No Hate and Offensive Speech"})
data = data[["tweet", "labels"]]

import re

# Data cleaning function
def clean(text):
    text = str(text).lower()
    text = re.sub('[.?]','',text)
    text = re.sub('https?://\S+|www.\S+', '', text)
    text = re.sub('<.?>+','',text)
    text = re.sub(r'[^\w\s]','',text)
    text = re.sub('\n','',text)
    text = re.sub('\w\d\w','',text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text
data["tweet"] = data["tweet"].apply(clean)

# Features and labels
x = np.array(data["tweet"])
y = np.array(data["labels"])

# Vectorize the text data
cv = CountVectorizer()
x = cv.fit_transform(y)

# Split the data into training and testing sets
X_train, X_text, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
# Predict on the test set
y_pred = model.predict(X_text)

# Print accuracy
from sklearn.metrics import accuracy_score
print("The accuracy is", accuracy_score(y_test, y_pred))

#classification report

from sklearn.metrics import classification_report
print("The classification report is",classification_report(y_test,y_pred))

#tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=cv.get_feature_names_out(), class_names=model.classes_, filled=True, rounded=True)
plt.show()



# Flask app
app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('indexHate.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_text = request.form['sentence']
        print("Input Text:", input_text)

        # Clean and vectorize the input text
        # input_text_cleaned = clean(input_text)

        input_text = cv.transform([input_text]).toarray()

        # print("Vectorized Text:", input_text_vectorized)


        # Map the prediction to a human-readable label
        pred_label = model.predict(input_text)

        print("Prediction Label:", pred_label)

        # return render_template('indexHate.html', prediction=pred_label)
        return render_template('indexHate.html', prediction=pred_label, input_text=input_text)

if __name__ == "__main__":
    app.run(debug=True)

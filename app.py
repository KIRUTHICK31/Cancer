from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
import warnings

app = Flask(__name__)


warnings.filterwarnings("ignore", category=UserWarning)


breast_cancer_dataset = load_breast_cancer()
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
data_frame['label'] = breast_cancer_dataset.target
X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_train_std = pd.DataFrame(X_train_std, columns=X.columns)
X_test_std = scaler.transform(X_test)


svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_std, Y_train)



@app.route('/')
def home():
    return render_template('test.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        input_features = [float(request.form[value]) for value in request.form]
        input_data = np.array(input_features).reshape(1, -1)
        input_data_std = scaler.transform(input_data)
        prediction = svm_classifier.predict(input_data_std)
        result = 'Malignant' if prediction[0] == 0 else 'Benign'
        return render_template('test.html', prediction_text=f'The tumor is {result}')
    except Exception as e:
        return render_template('test.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)

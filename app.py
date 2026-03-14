from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('model.joblib')
model_columns = joblib.load('features.joblib')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = ""
    
    if request.method == 'POST':
        try:
            input_data = []
            for col in model_columns:
                val = float(request.form[col])
                input_data.append(val)

            data = pd.DataFrame([input_data], columns=model_columns)
            prediction = model.predict(data)[0]
            prediction_text = f"Predicted AQI: {prediction:.2f}"
            
        except Exception as e:
            prediction_text = f"Error: {str(e)}"

    return render_template('index.html', result=prediction_text, columns=model_columns)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
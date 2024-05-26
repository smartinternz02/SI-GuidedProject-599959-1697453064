from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load your trained machine learning model
model_loaded = pickle.load(open("model.pkl", 'rb'))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/submit')
def submit():
    return render_template("predict.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the form\
    Client_Id = int(request.form['Client_Id'])
    Client_Income = float(request.form['Client_Income'])
    Car_Owned = int(request.form['Car_Owned'])
    Bike_Owned = int(request.form['Bike_Owned'])
    Active_Loan = int(request.form['Active_Loan'])
    Credit_Amount = float(request.form['Credit_Amount'])
    Loan_Annuity = float(request.form['Loan_Annuity'])
    client_income_type = int(request.form['client_income_type'])
    client_education = int(request.form['client_education'])
    client_marital_status = int(request.form['client_marital_status'])
    client_gender = int(request.form['client_gender'])
    loan_contract_type = int(request.form['loan_contract_type'])
    client_occupation = int(request.form['client_occupation'])
    type_organization = int(request.form['type_organization'])

    # Create an input array for prediction
    input_data = [
        Client_Id, Client_Income, Car_Owned, Bike_Owned, Active_Loan, Credit_Amount, Loan_Annuity,
        client_income_type, client_education, client_marital_status, client_gender,
        loan_contract_type, client_occupation, type_organization
    ]

    final_features = [np.array(input_data)]
    prediction = model_loaded.predict(final_features)
    output = "Likely to default on the loan payment" if prediction[0] == 1 else "Not likely to default on the loan payment"

    return render_template("result.html", prediction_text=output)

if __name__ == '__main__':
    app.run(debug=True)


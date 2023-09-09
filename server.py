from flask import Flask, render_template, request
import pickle
import csv

filenames = ["modelRF.pkl", "modelNB.pkl", "modelXGB.pkl", "modelLR.pkl", "modelDT.pkl", "modelMLP.pkl", "modelRF_NORATE.pkl"]
models = {}

for filename in filenames:
    with open(filename, 'rb') as file:
        models[filename] = pickle.load(file)

modelXGB = models["modelXGB.pkl"]
modelRF = models["modelRF.pkl"]
modelNB = models["modelNB.pkl"]
modelLR = models["modelLR.pkl"]
modelDT = models["modelDT.pkl"]
modelMLP = models["modelMLP.pkl"]
modelRF_NORATE = models["modelRF_NORATE.pkl"]

app = Flask(__name__)

def write_to_csv(data):
    try:
        with open('form_data.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
    except Exception as e:
        print(f"Error writing to CSV: {e}")

@app.route("/", methods=["GET"])
def root():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
        gender = int(request.form.get("gender"))
        own_property = int(request.form.get("own_property"))
        income_source = int(request.form.get("income_source"))
        education = int(request.form.get("education"))
        family_status = int(request.form.get("family_status"))
        occupation_type = int(request.form.get("occupation_type"))
        income = float(request.form.get("income"))
        mobile_no = str(request.form.get("mobile_no"))
        email_id = str(request.form.get("email_id"))
        age = int(request.form.get("age"))
        experience = int(request.form.get("experience"))
        family_size = float(request.form.get("family_size"))
        address = str(request.form.get("address"))
        pincode = str(request.form.get("pincode"))
        credit_history = float(request.form.get("credit_history"))
        
        if credit_history == 6:
            predictions = modelRF_NORATE.predict([[gender, own_property, income, income_source, education, family_status,
                                            age, experience, occupation_type, family_size]])
        else:
            algo = request.form.get("algo")

            if algo == "rf":
                predictions = modelRF.predict([[gender, own_property, income, income_source, education, family_status,
                                                age, experience, occupation_type, family_size,credit_history]])
            elif algo == "nb":
                predictions = modelNB.predict([[gender, own_property, income, income_source, education, family_status,
                                                age, experience, occupation_type, family_size,credit_history]])
            elif algo == "xgb":
                predictions = modelXGB.predict([[gender, own_property, income, income_source, education, family_status,
                                                age, experience, occupation_type, family_size,credit_history]])
            elif algo == "lr":
                predictions = modelLR.predict([[gender, own_property, income, income_source, education, family_status,
                                                age, experience, occupation_type, family_size,credit_history]])
            elif algo == "dt":
                predictions = modelDT.predict([[gender, own_property, income, income_source, education, family_status,
                                                age, experience, occupation_type, family_size,credit_history]])
            elif algo == "mlp":
                predictions = modelMLP.predict([[gender, own_property, income, income_source, education, family_status,
                                                age, experience, occupation_type, family_size,credit_history]])

        data_to_write = [gender, own_property, income_source, education, family_status,
                         occupation_type, algo, income, mobile_no, email_id,
                         age, experience, address, pincode, credit_history]
        
        write_to_csv(data_to_write)
        
        return render_template(
            "result.html",
            prediction=predictions[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4000, debug=True)

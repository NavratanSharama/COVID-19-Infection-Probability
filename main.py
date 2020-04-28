from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
# open a file, where you ant to store the data
file = open('model.pkl', 'rb')
clf = pickle.load(file)
file.close()

@app.route('/', methods=["GET", "POST"])
def hello_world():
    if request.method == "POST":
        myDict = request.form
        fever = int(myDict['fever'])
        age = int(myDict['age'])
        pain = int(myDict['pain'])
        runnynose = int(myDict['runnynose'])
        diffBreath = int(myDict['diffBreath'])
        # code for inference
        input_features = [fever, pain, age, runnynose, diffBreath]
        infProb = clf.predict_proba([input_features])[0][1]
        print(infProb)
        return render_template('show.html', inf=round(infProb*100))
    return render_template('index.html')
    #return 'Hello, World!' + str(infProb)

if __name__ == '__main__':
    app.run(debug=True)
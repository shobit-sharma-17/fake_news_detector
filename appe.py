from flask import Flask,render_template,request
import pickle
from sklearn.linear_model import LogisticRegression

model = pickle.load(open('pred.pkl','rb'))
vector = pickle.load(open('tfidf.pkl','rb'))

app = Flask(__name__)
@app.route('/')
def starting():
    return render_template("index_text.html")
@app.route("/predict", methods = ["POST", "GET"])
def predict():
    if request.method == "POST":
        message = str(request.form["message"])
        output = model.predict(vector.transform([message]))
    return render_template("index_text.html", pred = output)    
if __name__ == '__main__':
    app.run(debug=True)


from flask import Flask,render_template,request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer 


model = pickle.load(open("modelStore.pkl", "rb"))
fe = pickle.load(open("fetureExtraction.pkl",'rb'))

app = Flask(__name__)

@app.route('/')

def fun():
    return render_template("index.html")

@app.route("/handle",methods=['POST'])

def handle_post():
    email = request.form['email']
    efe = fe.transform([email])
    pred = model.predict(efe)
    return render_template("output.html",data = pred)

if __name__=="__main__":
    app.run(debug=True)



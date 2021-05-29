import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template,session, redirect, url_for
import pickle
import mysql.connector

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="forex"
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == "POST": 
       # getting input with name = fname in HTML form 
       base = request.form.get("base") 
       # getting input with name = lname in HTML form  
       target = request.form.get("target")  

   
    # AUD_USD = mydb.cursor()
    # AUD_USD.execute("SELECT * FROM AUD_USD WHERE Date = '%s'" % target)
    # myresult = AUD_USD.fetchall()
    # for x in myresult:
    #     print(x)    
    AUD_USD = pd.read_csv("AUD_USD_Final.csv")
    EUR_USD = pd.read_csv("EUR_USD_Final.csv")
    GBP_USD = pd.read_csv("GBP_USD_Final.csv")
    USD_CAD = pd.read_csv("USD_CAD_Final.csv")
    USD_JPY = pd.read_csv("USD_JPY_Final.csv")

    AUD_USD.Date = pd.to_datetime(AUD_USD.Date)
    EUR_USD.Date = pd.to_datetime(EUR_USD.Date)
    GBP_USD.Date = pd.to_datetime(GBP_USD.Date)
    USD_CAD.Date = pd.to_datetime(USD_CAD.Date)
    USD_JPY.Date = pd.to_datetime(USD_JPY.Date)
    
    temp = mydb.cursor()
    if(base == "AUD" and target == "USD"):
        temp.execute("SELECT Date,Average_Bid,Broker FROM aud_usd_final WHERE Date = '7/21/2018'")
        # temp = pd.DataFrame(data = AUD_USD[AUD_USD["Date"]=="7/21/2018"])
    elif(base == "EUR" and target == "USD"):
        temp.execute("SELECT Date,Average_Bid,Broker FROM eur_usd_final WHERE Date = '7/21/2018'")
        # temp = pd.DataFrame(data = EUR_USD[EUR_USD["Date"]=="7/21/2018"])
    elif(base == "GBP" and target == "USD"):
        temp.execute("SELECT Date,Average_Bid,Broker FROM gbp_usd_final WHERE Date = '7/21/2018'")
        # temp = pd.DataFrame(data = GBP_USD[GBP_USD["Date"]=="7/21/2018"])
    elif(base == "USD" and target == "CAD"):
        temp.execute("SELECT Date,Average_Bid,Broker FROM usd_cad_final WHERE Date = '7/21/2018'")
        # temp = pd.DataFrame(data = USD_CAD[USD_CAD["Date"]=="7/21/2018"])
    else:
        temp.execute("SELECT Date,Average_Bid,Broker FROM usd_jpy_final WHERE Date = '7/21/2018'")
        # temp = pd.DataFrame(data = USD_JPY[USD_JPY["Date"]=="7/21/2018"])
    myresult = temp.fetchall()
    # bid = temp[['Date','Average_Bid', 'Broker']]
    # bid = bid.sort_values(by=['Average_Bid'], ascending=False)
    # bid.to_string(index=False)
    # print(bid)

    # ask = temp[['Date','Average_Ask', 'Broker']]
    # ask=ask.sort_values(by=['Average_Ask'], ascending=True)
    # ask = ask.style.hide_index()
    # print(ask)
    # ask.index
    for x in myresult:
        print(x) 

    return render_template('index.html')
    # return render_template('predict.html',tables=[myresult.to_html(classes='myresult'), myresult.to_html(classes='myresult')],titles = ['na', 'Bidding', 'Asking'])

if __name__ == "__main__":
    app.run(debug=True)
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template,redirect, make_response
import pickle
import random
import json
from time import time
from random import random
from matplotlib import rcParams
import matplotlib.pyplot as plt
from datetime import date, timedelta
# model building libraries
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from forex_python.converter import CurrencyRates

app = Flask(__name__)
AUD_USD,EUR_USD,GBP_USD,USD_CAD,USD_JPY,Broker_Info= pickle.load(open('model.pkl', 'rb'))





@app.route('/')
def home():
    return render_template('index.html')

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')


@app.route('/futureforecast')
def future():
    return render_template('futureforecast.html')


@app.route('/forexpredict',methods=['POST'])
def Forex():
    Forex_Future = pd.read_csv('DATA/Forex.csv')
    Forex_Future.Date = pd.to_datetime(Forex_Future.Date)

    if request.method == "POST": 
       base = request.form.get("base") 
       target = request.form.get("target") 
       days = int(request.form.get("days") )
    print(days)
    # checking for input currency
    if(base == "USD" and target == "AUD"):
        Forex_Future["Open"]=Forex_Future["USD_AUD"]
    elif(base=="USD" and target=="EUR"):
        Forex_Future["Open"]=Forex_Future["USD_EUR"]
    elif(base=="USD" and target=="GBP"):
        Forex_Future["Open"]=Forex_Future["USD_GBP"]
    elif(base=="USD" and target=="INR"):
        Forex_Future["Open"]=Forex_Future["USD_INR"]
    elif(base=="USD" and target=="JPY"):
        Forex_Future["Open"]=Forex_Future["USD_JPY"]
    else:
        return render_template('predict.html')
    Forex_Future=Forex_Future.drop(["Date","USD_AUD","USD_EUR","USD_GBP","USD_INR","USD_JPY"], axis = 1)

    future_days = days

    Forex_Future["Prediction"] = Forex_Future[["Open"]].shift(-future_days)

    X = np.array(Forex_Future.drop(["Prediction"], 1))[:-future_days]

    y = np.array(Forex_Future["Prediction"])[:-future_days]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.05)

    tree = DecisionTreeRegressor().fit(x_train, y_train)

    x_future = Forex_Future.drop(["Prediction"], 1)[:-future_days]
    x_future = x_future.tail(future_days)
    x_future = np.array(x_future)

    tree_prediction = tree.predict(x_future)
    print(tree_prediction)

    prediction = tree_prediction
    valid = Forex_Future[X.shape[0]:]
    valid['Prediction'] = prediction
    plt.figure(figsize=(9, 4))
    plt.title('Forecasted Result')
    plt.xlabel('Days')
    plt.ylabel("Price")
    plt.plot(Forex_Future["Open"])
    plt.plot(valid[["Open", "Prediction"]])
    plt.legend(["Orig", "val", "Pred"])
    # plt.show()
    ax = plt.axes()
    ax.set_facecolor("white")
    # plt.show()
    plt.savefig('static/forecast_res.png', dpi=None, facecolor='silver', color='red')
    print(valid["Prediction"])
    curr = CurrencyRates()
    live=curr.get_rate(base, target)
    print(curr.get_rate(base, target))
    return render_template('futureforecast.html', forexprediction=valid.to_dict(orient='record'),live=live,b=base,t=target)




@app.route('/predict',methods=['POST'])
def predict():
    # getting user input 
    if request.method == "POST": 
       base = request.form.get("base") 
       target = request.form.get("target")  
       date = request.form.get("date")

    date = pd.to_datetime(date)
    dt = pd.to_datetime(date) - timedelta(365)
    # take only those value in Dataframe that matches user input
    if(base == "AUD" and target == "USD"):
        temp = pd.DataFrame(data = AUD_USD[AUD_USD["Date"]==date])
        temp1 = AUD_USD[AUD_USD["Date"].between(dt, date)]
    elif(base == "EUR" and target == "USD"):
        temp = pd.DataFrame(data = EUR_USD[EUR_USD["Date"]==date])
        temp1 = EUR_USD[EUR_USD["Date"].between(dt, date)]
    elif(base == "GBP" and target == "USD"):
        temp = pd.DataFrame(data = GBP_USD[GBP_USD["Date"]==date])
        temp1 = GBP_USD[GBP_USD["Date"].between(dt, date)]
    elif(base == "USD" and target == "CAD"):
        temp = pd.DataFrame(data = USD_CAD[USD_CAD["Date"]==date])
        temp1 = USD_CAD[USD_CAD["Date"].between(dt, date)]
    elif(base == "USD" and target == "JPY"):
        temp = pd.DataFrame(data = USD_JPY[USD_JPY["Date"]==date])
        temp1 = USD_JPY[USD_JPY["Date"].between(dt, date)]
    else:
        return render_template('predict.html')

# Final Bid 
    bid = temp[['Date','Average_Bid', 'Broker']]
    bid = bid.sort_values(by=['Average_Bid'], ascending=False)
    bid1= bid.merge(Broker_Info, how='inner', left_on=['Broker'], right_on=['Broker'])
# Final Ask 
    ask = temp[['Date','Average_Ask', 'Broker']]
    ask=ask.sort_values(by=['Average_Ask'], ascending=True)
    ask1= ask.merge(Broker_Info, how='inner', left_on=['Broker'], right_on=['Broker'])
# Separating data According to brokers for chart
    temp1 = temp1[["Date", "Average_Bid", "Average_Ask", "Broker"]]
    temp1 = temp1.set_index("Date")
    PLUS_500 = pd.DataFrame(data =temp1[temp1["Broker"]=="PLUS 500"])
    CMC_Market = pd.DataFrame(data =temp1[temp1["Broker"]=="CMC Market"])
    LCG = pd.DataFrame(data =temp1[temp1["Broker"]=="London Capital Group"])
    SAXO = pd.DataFrame(data =temp1[temp1["Broker"]=="SAXO"])
    FXTM = pd.DataFrame(data =temp1[temp1["Broker"]=="FXTM"])
    PapperStone = pd.DataFrame(data =temp1[temp1["Broker"]=="PapperStone"])
    XTB = pd.DataFrame(data =temp1[temp1["Broker"]=="XTB"])
    IC_Market = pd.DataFrame(data =temp1[temp1["Broker"]=="IC Market"])
    OCTA_FX = pd.DataFrame(data =temp1[temp1["Broker"]=="OCTA FX"])
    ROBO_Market = pd.DataFrame(data =temp1[temp1["Broker"]=="ROBO Market"])
    rcParams['figure.figsize'] = 13,5
    # bid.loc[bid['Broker']=="ROBO Market"]
    # plotting graph for Bid
    plot=bid.head(3)
    for i in plot.index:
        if(plot.loc[i,'Broker'] == "PLUS 500"):
            plt.plot(PLUS_500.Average_Bid, label = 'PLUS 500')
        if(plot.loc[i,'Broker'] == "CMC Market"):
            plt.plot(CMC_Market.Average_Bid, label = 'CMC Market')
        if(plot.loc[i,'Broker'] == "London Capital Group"):
            plt.plot(LCG.Average_Bid, label = 'LCG')
        if(plot.loc[i,'Broker'] == "SAXO"):
            plt.plot(SAXO.Average_Bid, label = 'SAXO')
        if(plot.loc[i,'Broker'] == "FXTM"):
            plt.plot(FXTM.Average_Bid, label = 'FXTM')
        if(plot.loc[i,'Broker'] == "PapperStone"):
            plt.plot(PapperStone.Average_Bid, label = 'PaperStone')
        if(plot.loc[i,'Broker'] == "XTB"):
            plt.plot(XTB.Average_Bid, label = 'XTB')
        if(plot.loc[i,'Broker'] == "IC Market"):
            plt.plot(IC_Market.Average_Bid, label = 'IC Market')
        if(plot.loc[i,'Broker'] == "OCTA FX"):
            plt.plot(OCTA_FX.Average_Bid, label = 'OCTA FX')
        if(plot.loc[i,'Broker'] == "ROBO Market"):
            plt.plot(ROBO_Market.Average_Bid, label = 'ROBO Market')
    ax = plt.axes()
    legend = ax.legend()
    legend.remove()
    plt.legend(loc=2)
    plt.xlabel("Days")
    plt.ylabel("Bid")
    plt.title("Top 3 Brokers")
    
    ax.set_facecolor("white")
    # plt.show()
    plt.savefig('static/bid_avg.png', dpi=None, facecolor='silver', color='red')

# Plotting graph for Ask
    plot=ask.head(3)
    for i in plot.index:
        if(plot.loc[i,'Broker'] == "PLUS 500"):
            plt.plot(PLUS_500.Average_Ask, label = 'PLUS 500')
        if(plot.loc[i,'Broker'] == "CMC Market"):
            plt.plot(CMC_Market.Average_Ask, label = 'CMC Market')
        if(plot.loc[i,'Broker'] == "London Capital Group"):
            plt.plot(LCG.Average_Ask, label = 'LCG')
        if(plot.loc[i,'Broker'] == "SAXO"):
            plt.plot(SAXO.Average_Ask, label = 'SAXO')
        if(plot.loc[i,'Broker'] == "FXTM"):
            plt.plot(FXTM.Average_Ask, label = 'FXTM')
        if(plot.loc[i,'Broker'] == "PapperStone"):
            plt.plot(PapperStone.Average_Ask, label = 'PaperStone')
        if(plot.loc[i,'Broker'] == "XTB"):
            plt.plot(XTB.Average_Ask, label = 'XTB')
        if(plot.loc[i,'Broker'] == "IC Market"):
            plt.plot(IC_Market.Average_Ask, label = 'IC Market')
        if(plot.loc[i,'Broker'] == "OCTA FX"):
            plt.plot(OCTA_FX.Average_Ask, label = 'OCTA FX')
        if(plot.loc[i,'Broker'] == "ROBO Market"):
            plt.plot(ROBO_Market.Average_Ask, label = 'ROBO Market')
    plt.legend(loc=2)
    plt.xlabel("Days")
    plt.ylabel("Ask")
    plt.title("Top 3 Brokers")
    ax = plt.axes()
    ax.set_facecolor("white")
    # plt.show()
    plt.savefig('static/ask_avg.png', dpi=None, facecolor='silver', color='red')
    curr = CurrencyRates()
    live=curr.get_rate(base, target)
    print(curr.get_rate(base, target))
    return render_template('predict.html', bid=bid1.to_dict(orient='record'),ask=ask1.to_dict(orient='record'),b=base,t=target,live=live)
    # return render_template('predict.html',tables=[bid.to_html(classes='bid',index=False), ask.to_html(classes='ask',index=False)],titles = ['na', 'Bidding', 'Asking')

if __name__ == "__main__":
    app.run(debug=True)
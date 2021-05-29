import pandas as pd
import numpy as np

ECB = pd.read_csv("C:/Users/Swapnil/Desktop/WU/ECB (1).csv")
Oanda = pd.read_csv("C:/Users/Swapnil/Desktop/WU/Oanda.csv")

ECB["Broker"] = "European Central Bank"
Oanda["Broker"] = "Oanda"
Oanda.to_string(index=False)
x = pd.concat([ECB, Oanda])

temp1 = pd.DataFrame(data =ECB[ECB["date"]=="11/14/20"] )
temp1 =pd.DataFrame(data =temp1[temp1["base"]=="USD"])
temp1 =pd.DataFrame(data =temp1[temp1["target"]=="EUR"])

temp2 = pd.DataFrame(data =Oanda[Oanda["date"]=="11/14/2020"] )
temp2 =pd.DataFrame(data =temp2[temp2["base"]=="USD"])
temp2 =pd.DataFrame(data =temp2[temp2["target"]=="EUR"])

temp=pd.concat([temp1,temp2])
# temp.style.hide_index()

bid=temp[['base','target','date','average_bid', 'Broker']]
print("BID PRICE")
bid=bid.sort_values(by=['average_bid'], ascending=False)
print(bid)

ask=temp[['base','target','date','average_ask', 'Broker']]
print("ASK PRICE")
ask=ask.sort_values(by=['average_ask'], ascending=True)
print(ask)
import numpy as np
import pandas as pd
from time import strptime
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from math import exp

BondData = pd.read_csv("CleanBondDatacsv")
BondData['m_year'] = pd.DatetimeIndex(BondData["Maturity"]).year
BondData['m_month'] = pd.DatetimeIndex(BondData["Maturity"]).month

bond_list = ["CAN 1.5 MAR 20", 
             "CAN 0.75 SEP 20", 
             "CAN 0.75 MAR 21",
             "CAN 0.75 SEP 21", 
             "CAN 0.5 MAR 22", 
             "CAN 2.75 JUN 22",
             "CAN 1.75 MAR 23", 
             "CAN 1.5 JUN 23", 
             "CAN 2.25 MAR 24",
             "CAN 1.5 SEP 24"]

bond_time = []
Panda_Data_Frame = []
for i in bond_list:
    holder = i.split(" ")
    month = strptime(holder[2], "%b").tm_mon
    bond_time.append(np.array([month, int(holder[3]), holder[1]]))
    
# Select the bond BondData we want, given the list that we have chosen to construct yield to maturity.
for bond in bond_time:
    cur_BondData = BondData[(BondData.m_month == int(bond[0])) & (BondData.m_year == int("20" + bond[1])) & (BondData.BondPayment == float(bond[2]))]
    cur_BondData['TimeToMaturity'] = (cur_BondData['m_year'] - 2020) + (cur_BondData['m_month'] - 2) / 12
    cur_BondData['par'] = 100

    Panda_Data_Frame.append(cur_BondData)
meta_BondData = pd.concat(Panda_Data_Frame)
holder = []
# Use Newton Optimization method with
def Optimization_YTM(price, par, T, coup, freq=2, guess=0.05):
    freq = float(freq)
    periods = T * freq
    BondPayment = coup / 100. * par / freq
    dt = [(i + 1) / freq for i in range(int(periods))]

    ytm_func = lambda y:  sum([BondPayment/(1+y/freq)**(freq*t) for t in dt]) + par/(1+y/freq)**(periods) - price
    return optimize.newton(ytm_func, guess)

for index, row in meta_BondData.iterrows():
    yield_rate = Optimization_YTM(row['Price'], row['par'], row['TimeToMaturity'], row['BondPayment'])
    holder.append(pd.Series([row["ISIN"], row["date"], row['TimeToMaturity'], yield_rate, row["BondPayment"], row["Price"], row['m_month'], row['m_year']]))
DataFrame_yield = pd.concat(holder, axis=1).T
DataFrame_yield.rename(columns={0:"ISIN", 1:"date", 2:"TimeToMaturity", 3:"Yield_rate", 4:"BondPayment", 5:"Price", 6:'m_month', 7:'m_year'}, inplace=True)
DataFrame_yield['date'] = pd.to_datetime(DataFrame_yield.date)

for bond in bond_time:
    cur_BondData = DataFrame_yield[(DataFrame_yield.m_month == int(bond[0])) & (DataFrame_yield.m_year == int("20" + bond[1])) & (DataFrame_yield.BondPayment == float(bond[2]))]
    if int(bond[0]) == 3:
        cur_BondData['Price'] = 136 / 365 * DataFrame_yield.BondPayment/2 + cur_BondData['Price']
    elif int(bond[0]) == 6:
        cur_BondData['Price'] = 45 / 365 * DataFrame_yield.BondPayment/2 + cur_BondData['Price']
    elif int(bond[0]) == 9:
        cur_BondData['Price'] = 136 / 365 * DataFrame_yield.BondPayment/2 + cur_BondData['Price']
date = DataFrame_yield["date"].unique()
spot_rate = np.full([100, 2], 0)
Panda_Data_Frame_2 = []
DataFrame_yield['spot'] = 0.0000
DataFrame_yield['forward'] = 0.0000
for item in date:
    yield_current = DataFrame_yield[DataFrame_yield["date"] == item].sort_values(by=["TimeToMaturity"])
    NumBonds = yield_current["ISIN"].shape[0]
    for i in range(NumBonds):
        Principal = 100
        payment = yield_current["BondPayment"].iloc[i]/2
        Notional = Principal + payment
        Price = yield_current["Price"].iloc[i]
        T = yield_current["TimeToMaturity"].iloc[i]
        BondPaymentPayment = 0
        if i>0:
            for j in range(i - 1):
                spotrate = yield_current["spot"].iloc[j]
                BondPayment_time = yield_current["TimeToMaturity"].iloc[j]
                BondPaymentPayment = BondPaymentPayment + payment*np.exp(-1 * spotrate * BondPayment_time)
        rate = - np.log((Price - BondPaymentPayment) / Notional) / T

        yield_current.at[int(yield_current.iloc[i].name), 'spot'] = rate

    Panda_Data_Frame_2.append(yield_current)
SpotRateBondDataFrame = pd.concat(Panda_Data_Frame_2)
forwardlist = []
for item in date:
    yield_current = SpotRateBondDataFrame[SpotRateBondDataFrame["date"] == item].sort_values(by=["TimeToMaturity"])
    NumBonds = yield_current["ISIN"].shape[0]
    for i in range(NumBonds):
        if i > 0:
            riti = yield_current['spot'].iloc[i] * yield_current['TimeToMaturity'].iloc[i]
            r0t0 = yield_current['spot'].iloc[0] * yield_current['TimeToMaturity'].iloc[0]
            forward_rates = (riti - r0t0) / (yield_current['TimeToMaturity'].iloc[i] - yield_current['TimeToMaturity'].iloc[0])
            yield_current.at[int(yield_current.iloc[i].name), 'forward'] = forward_rates
    forwardlist.append(yield_current)
forward_rate = pd.concat(forwardlist)
X_mat_1 = np.full([9, 5], np.nan)
X_mat_2 = np.full([9, 5], np.nan)
bond_names = DataFrame_yield.ISIN.unique()
i = 0
for i in range(int(len(bond_names)/2)):
    yield_current = DataFrame_yield[DataFrame_yield['ISIN']==bond_names[2*i+1]].sort_values(by=["date"])
    cur_forward = forward_rate[forward_rate['ISIN']==bond_names[2*i+1]].sort_values(by=["date"])
    X_mat_1[:, i] = np.log(yield_current['Yield_rate'].values.astype('float')[:-1]/yield_current['Yield_rate'].values.astype('float')[1:])
    X_mat_2[:, i] = np.log(cur_forward['forward'].values.astype('float')[:-1]/cur_forward['forward'].values.astype('float')[1:])
#Plotting For The Yield To Maturity Curve
fig = plt.figure()
for item in date:
    yield_current = DataFrame_yield[DataFrame_yield["date"] == item].sort_values(by=["TimeToMaturity"])["Yield_rate"]
    plt.plot(list(range(10)), yield_current, label=str(item)[:10])
plt.title("Yield")
plt.legend(fontsize='x-small')
plt.show()
#Plotting for Spot Curve
fig = plt.figure()
for item in date:
    yield_current = SpotRateBondDataFrame[SpotRateBondDataFrame["date"] == item].sort_values(by=["TimeToMaturity"])["spot"]
    plt.plot(list(range(8)), yield_current[2:], label=str(item)[:10])
plt.title("Spot")
plt.legend(fontsize='x-small')
plt.show()
#Forward Curve
fig = plt.figure()
for item in date:
    yield_current = forward_rate[forward_rate["date"] == item].sort_values(by=["TimeToMaturity"])["forward"]
    plt.plot(list(range(9)), yield_current[1:], label=str(item)[:10])
plt.title('Forward')
plt.legend(fontsize='x-small')
plt.show()
cov_1 = np.cov(X_mat_1.T*100)
cov_2 = np.cov(X_mat_2.T*100)

eig_1 = np.linalg.eig(cov_1)
eig_2 = np.linalg.eig(cov_2)

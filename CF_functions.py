#%%
# INSTALLING PACKAGES
import os
import pandas as pd
from copy import copy
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import itertools
from scipy.interpolate import splev, splrep
import statsmodels.api as sm
import random
from datetime import datetime
import time

os.chdir(os.path.dirname(__file__))
# READING SAVED DATA
datapath = os.getcwd()+'/data/data.pkl'
data = pd.read_pickle(datapath)

# REFORMAT DATA - WIDE DESIGN MATRIX
# mle_data = copy(data[['Binary','Group','Revenue year','Size','Price']])
mle_data = copy(data[['Binary', 'Group', 'Revenue year', 'Size', 'Price',
                      'Nview', 'Age', 'Cost', 'New', 'Amount']])
# LOGALIZE NVIEW,AGE
mle_data['Nview'] = np.log(mle_data['Nview']+1)
mle_data['Age'] = np.log(mle_data['Age'])
mle_data = mle_data.dropna()
# WIDEN
groups = pd.get_dummies(mle_data['Group'])
mle_data['Medium'] = groups.iloc[:,1] 
mle_data['Large'] = groups.iloc[:,2]
times = pd.get_dummies(mle_data['Revenue year'])
mle_data['Year'] = times.iloc[:,1]

#%%
# MLE estimates
# x = [ 2.22674415e+03, -5.49937040e+02, -6.66158998e+02,  1.80445730e+02, -4.32924199e+01,  8.05901134e+01,  2.57653653e-03]

def Value_generation(MLEstimates, data=mle_data):

    s = 1/MLEstimates[-1]
    print(MLEstimates[1],MLEstimates[2])

    X = np.array(data[['Medium','Large','Nview','Age','Year']])
    X = sm.add_constant(X)
    data['Value'] = X@MLEstimates[:-1]
    opt_data = copy(data.loc[:,['Group', 'Year', 'Size', 'Value', 'Nview', 'Age',
                                'New', 'Binary', 'Cost', 'Amount', 'Price']].reset_index(drop=True))

    # MEMBERSHIP COST FUNCTION ESTIMATION: NVIEWS
    inter_data = copy(data)
    inter_data = inter_data.sort_values(by='Nview')
    knots = [3,6]
    tck_nview = splrep(x=inter_data['Nview'], y=inter_data.Cost, t=knots, k=1)

    return(opt_data, tck_nview, s)

# CONSUMER WILLINGNESS-TO-PAY: value as a function of quantity
# v_it/qbar_it: value/size estimated for indiv consumer
# q: range of quantity
def Consumer_i(v_it, qbar_it, q):
    idx = q < qbar_it
    q[idx] *= v_it
    q[~idx] = v_it * qbar_it
    return(q)

def Smooth_Consumer_i(v_it, qbar_it, q, b, m):
    if qbar_it != 0 :
        A = qbar_it ** (-b)
        B = q ** (-b)
        smooth_q = (A+m*B)**(-1/b)
        smooth_q[0] = 0
    else:
        A = 0
        B = q * 0
        smooth_q = B
    return(v_it * smooth_q)

# CONSUMER WTP MATRIX WITH VARYING QUANTITY: building matrix from Consumer_i
# V_it/Qbar_it: value/size vector - each row = indiv consumer
# rows: consumers, columns: quantities 
def Consumer_matrix(V_it, Qbar_it, smooth=False, b=4, m=0.48525429746950965):
    Consumers = np.zeros((len(V_it), 201)) 
    # break tie toward larger quantity given same WTP
    noise = np.array(range(201)) * 1e-5
    if not smooth: 
        for i in range(len(V_it)):
            Consumers[i,:] = Consumer_i(v_it=V_it[i], qbar_it=Qbar_it[i], q=np.linspace(0, 200, 201)) + noise
    else:
        for i in range(len(V_it)):
            Consumers[i,:] = Smooth_Consumer_i(v_it=V_it[i], qbar_it=Qbar_it[i], q=np.linspace(0, 200, 201), b=b, m=m) + noise
    # Add Qbar_it vecter to the WTP matrix
    Consumers = np.hstack((np.reshape(np.array(Qbar_it), (len(Qbar_it), 1)), Consumers))
    return(Consumers)

# SCHEDULE VECTOR(q): total compensation as a function of quantity
# Y / Q / q: total compensation / quantity cutoffs / range of quantities
def Schedule(Y, Q, q, continuity, add_zero=True):
    ind = copy(q)
    # continuity imposed
    if continuity == True:
        for i in range(len(Q)-1):
            P = (Y[i+1]-Y[i])/(Q[i+1]-Q[i])
            idx = (ind > Q[i]) & (ind <= Q[i+1])
            q[idx] = P*(q[idx]-Q[i])+Y[i]
    # continuity not imposed
    else:
        diffQ = np.diff(Q)
        diffY = np.diff(Y)
        P = [diffY[i]/diffQ[i] for i in range(len(diffQ)) if i % 2 == 0]
        for i in range(int(len(Q)/2)):
            idx = (Q[2*i] <= ind) & (ind <= Q[2*i+1])
            q[idx] = P[i]*(q[idx]-Q[2*i])+Y[2*i]
    # add zero(origin) at the beginning
    if add_zero == False:
        pass
    else:
        q = np.insert(q,0,0)
    return(q)

# PROFIT FUNCTION: profit as a function of consumer WTP matrix, price schedule, cost
# cmat / Y,Q / cost: consumer WTP matrix / price schedule / cost
def Profit(Y, Q, cmat, cost, continuity, group=None):
    sched = Schedule(Y=Y, Q=Q, q=np.linspace(1, 200, 200), continuity=continuity) 
    # consumer surplus = consumer WTP - price 
    surplus = cmat[:, 1:] - sched
    # Get the qbar vector
    Qbar_it = cmat[:, 0]
    # optimal quantity demanded
    q_star = np.argmax(surplus, axis=1)
    c_welfare = np.max(surplus, axis=1)
    if np.any(c_welfare < 0):
        print('negative consumer welfare warning')
    else: pass
    # consumers buying positive quantity
    q_pos = copy(q_star)
    q_pos[q_pos.nonzero()] = 1
    # consumer optimal revenue
    sched_star = Schedule(Y, Q, q=copy(q_star), add_zero=False, continuity=continuity)
    # cost information
    SNC, new, fixed, SNC_vc = cost

    #Profit function with cost assumptions
    if cost[0].shape[0] == 201:
        print('wrong cost input')
    else:
        profits = sched_star - (601 + SNC_vc) * q_star - (SNC * fixed) * q_pos  - (new * 3194) * q_pos
        variable_costs = (601 + SNC_vc) * q_star 

    total_costs = sched_star - profits

    df = None
    if group is not None:
        groups = np.zeros(len(Qbar_it))
        for i, lb in enumerate(group):
            # Assign group based on Qbar_it and the group thresholds;
            # Greater size, larger group number
            groups[Qbar_it >= lb] = int(i)
        # Create df
        df = pd.DataFrame({
            "Group": groups,
            "Revenue": sched_star/1e7,
            "Cost": total_costs/1e7,
            "Profit": profits/1e7,
            "Value": "Fitted"
            })
        df["Acceptance"] = np.where(df.Revenue == 0, 0, 1)

    return(np.sum(profits)/len(profits), profits, q_star, q_pos, total_costs, variable_costs, c_welfare, df)


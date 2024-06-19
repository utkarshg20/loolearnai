import pandas as pd
import numpy as np
import datetime, requests, re, time
from functools import reduce
# from sqlalchemy import create_engine
from selenium import webdriver
from bs4 import BeautifulSoup
from autograd import value_and_grad

def get_fond_dataframes(start_date, end_date, fond_id):

    #-----Get data from Morningstar-Tool-----
    data_url = ("http://tools.morningstar.at/api/rest.svc/timeseries_price/"
                "5370efewxk?id={fond_id}%5D2%5D1%5D&currencyId=EUR&"
                "idtype=Morningstar&priceType=&frequency=monthly&startDate=  {start_date}&"
                "endDate={end_date}&outputType=COMPACTJSON").format(fond_id = fond_id, start_date = start_date, end_date = end_date)  

    result = requests.get(data_url).json()

    #-----Convert from Unix-Time to usable Time Format-----
    conv_time = lambda x: datetime.datetime.fromtimestamp(int(x/1000)).strftime('%Y-%m-%d %H:%M:%S')
    fond_prices = pd.DataFrame([ [ conv_time(i[0]), i[1]] for i in result ], columns = ["date", "price"])
    #rendite_ts = [ [ conv_time(i[0]), i[1]] for i in result ]

    return fond_prices


def get_Optimized_Portfolio( Covar_Mat, MeanVals, Exp_Value ):

    constraints1 = ({'type': 'eq', 'fun': lambda weights:  np.sum(weights) - 1})
    constraints2 = ({'type': 'ineq', 'fun': lambda weights:  weights @ Exp_Ret - Exp_Value})
    constraints = [constraints1, constraints2]

    bounds = [(0,1)] * len(MeanVals)

    #-----Define minimization Problem for Modern-Portfolio-Theory-----
    def min_markowitz(weights, cov_mat):
        return weights @ (cov_mat @ weights)  

    def target_func(weights):
        return min_markowitz(weights, Covar_Mat)

    target_with_grad = value_and_grad(target_func)


    #-----Do the Minimization with ScipY-----
#    result = minimize(target_with_grad, x0, jac = True, method='SLSQP', bounds = bounds, constraints = constraints, tol = 1e-30, options = {"maxiter": 1500})
    result = minimize(target_func, x0, jac = False, method='SLSQP', bounds = bounds, constraints = constraints, tol = 1e-30, options = {"maxiter": 3500})
    return result

if __name__ == "__main__":

    #-----Set Start & End Date
    start_date = "2012-01-01"
    end_date = str(datetime.datetime.today().date())


    #-----Set Fond ids----
    fond_ids = ["F0GBR06DWD","F00000T4KE","F000000255","F00000QLUP",
                "0P0000VHOL","0P0000JNCV","F000002J6W","F0GBR04LVP","F0GBR04FOH","F0GBR04D0X","0P0000M7TK",
                "F0GBR04D20","F0GBR04PMR","F000005KE0","F0GBR04CIW","F0GBR064OK","F0000020H2"]
    #"F0000007LD","F00000LNTR"
    all_prices = []

    #-----Loop over all fonds and collect data-----
#    for i in fond_ids:
#        print("i = ", i)
#        all_prices.append ( get_fond_dataframes(start_date, end_date, i) )    

    #-----calculate the covariance Matrix and Expected Returns-----
    R_frame = pd.DataFrame()


    #-----pct_change(1) gives back: list[i + 1] / list[i] to calculate the returns-----
    for i in fond_ids:
        print("i = ", i)
        R_frame[i] = get_fond_dataframes(start_date, end_date, i).price.pct_change(1).values + 1.0

    Covar_Mat = R_frame.cov()
    MeanVals  = R_frame.mean()        
    Exp_Ret = MeanVals.values

    #-----Optimize the Weights for different Fonds-----
    from scipy.optimize import minimize
    x0 = (lambda x: x/np.sum(x))(np.random.uniform(low = 0, high = 1, size = len(MeanVals)))

   # x0 = np.linspace(1 / len(MeanVals), 1 / len(MeanVals), num = len(MeanVals))

    #-----Set Constraints: Only Long Portfolios sum(w_i) == 1 && Expected Return > value-----

    constraints1 = ({'type': 'eq', 'fun': lambda weights:  np.sum(weights) - 1})
    constraints2 = ({'type': 'ineq', 'fun': lambda weights:  weights @ Exp_Ret - 1.001})
    constraints = [constraints1, constraints2]

    bounds = [(0,1)] * len(MeanVals)

    #-----Define minimization Problem for Modern-Portfolio-Theory-----
    def min_markowitz(weights, cov_mat):
        return weights @ (cov_mat @ weights)  

    def target_func(weights):
        return min_markowitz(weights, Covar_Mat)

    #-----Do the Minimization with ScipY-----
    res = minimize(target_func, x0, jac = False, method='SLSQP', bounds = bounds, constraints = constraints)

    #-----Set the expected return 'Exp_Value' that the portfolio should yield in 1 year-----
    #-----Choose wether the data was gathered daily, weekly or monthly-----
    weekly  = 1.0 / 52
    daily   = 1.0 / 251
    monthly = 1.0 / 12  

    Exp_Ret_List = np.linspace(1.02, 1.20, num = 20)

    fond_list = []

    for i in Exp_Ret_List:

        Exp_Value = i**(monthly)        
        fond_list.append( get_Optimized_Portfolio( Covar_Mat, MeanVals, Exp_Value ) )

    x_axis = []

    for i in fond_list:
        x_axis.append(i.fun * np.sqrt(12))


    #-----Plot the Efficiency Frontier-----    
    import matplotlib.pyplot as plt    

    plt.plot(x_axis, Exp_Ret_List)
    plt.ylabel('Expected Return')
    plt.xlabel('Standard Deviation')
    plt.title('MPT Efficiency Frontier')
    plt.grid(True)
    plt.show()

    #-----Print out Fonds between 1.08 and 1.16 Expected return
    np.set_printoptions(precision=4)  
    for i in range(5,20):
        print(i, np.round(fond_list[i].x, 5), Exp_Ret_List[i])
        #print("{:.1f} {:.1f}".format(i, fond_list[i].x))
        #print("{:.4f} {:.4f} {:.4f}".format(i, fond_list[i].x, Exp_Ret_List[i]))    

    np.round(fond_list[10].x, 5)
    Exp_Ret

    np.round(fond_list[10].x, 5) * Exp_Ret

    (np.sum(np.round(fond_list[10].x, 5) * Exp_Ret))**12
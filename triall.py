# login_app.py
import streamlit as st
import mysql.connector
import plotly.graph_objs as go
import pandas as pd
import yfinance as yf 
import pandas_datareader.data as web
import pandas as pd 
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import copy
import pandas_datareader as pdr
from io import BytesIO
from pypfopt import objective_functions
from pypfopt import plotting
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.hierarchical_portfolio import HRPOpt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import HRPOpt
from passlib.hash import pbkdf2_sha256
from datetime import timedelta, date
from yahoo_fin import stock_info as si 
from pypfopt import risk_models
from pypfopt import expected_returns

print(plt.style.available)
if "form" not in st.session_state:
    st.session_state.form = "signin"

if "user" not in st.session_state:
    st.session_state.user = ""
# Database connection
connection = mysql.connector.connect(
    host="sql5.freesqldatabase.com",
    user="sql5681425",
    password="qH5z2s6jQc",
    database="sql5681425",
    port=3306
)
cursor = connection.cursor()

today=date.today()
five_yr= today - timedelta(days=1825)
stocks_close = pd.DataFrame()

def login():
    global user
    if st.session_state.form == 'signin':
        st.markdown(
        """
        <style>
            .login-container {
                max-width: 400px;
                padding: 20px;
                margin: auto;
                background-color: #f4f4f4;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }

            .login-header {
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 20px;
            }

            .login-input {
                width: 100%;
                padding: 10px;
                margin-bottom: 15px;
                border: 1px solid #ccc;
                border-radius: 5px;
                box-sizing: border-box;
            }

            .login-button {
                width: 100%;
                padding: 10px;
                background-color: #4caf50;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }

            .login-button:hover {
                background-color: #45a049;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
        log_placeholder = st.empty()
        createacct = st.empty()

        with log_placeholder.form(key='login_form'):
            st.markdown('<div class="login-header">Login</div>', unsafe_allow_html=True)
            login_username = st.text_input("Username:", key='login-input')
            login_password = st.text_input("Password:", type="password", key='pass_l')
            login_button = st.form_submit_button("Login")

        if createacct.button('Create an account'):
            st.session_state.form = 'signup'
            createacct.empty()
            log_placeholder.empty()
            login()
            
        if login_button:
            if authenticate_user(login_username, login_password):
                #success = st.success("Login Successful!")
                log_placeholder.empty()
                createacct.empty()
                #success.empty()
                st.session_state.user = login_username
                st.session_state.form = 'loggedin'
                home()
            else:
                st.error("Invalid username or password.")

    elif st.session_state.form == 'signup':
        register_user()    

    else:
        home()
        
def authenticate_user(username, password):
    # Fetch user from the database
    query = f"SELECT * FROM login_info WHERE username = '{username}'"
    cursor.execute(query)
    result = cursor.fetchone()

    # Check if the user exists and the password matches
    if result and pbkdf2_sha256.verify(password, result[2]):
        return True
    else:
        return False

###################################################################################################

def register_user():
    reg_placeholder = st.empty()
    with reg_placeholder.form(key='register_form'):
        st.header("Register")

        # Text inputs for registration
        register_username = st.text_input("Username:", key='user_r')
        register_password = st.text_input("Password:", type="password", key='pass_r')
        register_name = st.text_input("Name:", key='name_r')
        register_email = st.text_input("Email:", key='email_r')

        # Register button
        register_button = st.form_submit_button("Register")

    # Validate registration upon form submission
    if register_button:
        if register_conf(register_username, register_password, register_name, register_email):
            #success = st.success("Registration Successful!")
            # Redirect to the login page or perform additional actions
            reg_placeholder.empty()
            #success.empty()
            st.session_state.form = 'signin'
            login()
        else:
            st.error("Username already exists. Choose a different username.")

def register_conf(username, password, name, email):
    # Check if the username already exists
    query_check_username = f"SELECT * FROM login_info WHERE username = '{username}'"
    cursor.execute(query_check_username)
    existing_user = cursor.fetchone()

    if existing_user:
        return False

    # If the username doesn't exist, insert the new user into the database
    hashed_password = pbkdf2_sha256.hash(password)
    query_insert_user = "INSERT INTO login_info (username, password, name, email) VALUES (%s, %s, %s, %s)"
    data = (username, hashed_password, name, email)

    cursor.execute(query_insert_user, data)
    connection.commit()

    return True

###############################################################################################################

def list_to_string(input_list, delimiter=','):
    """
    Convert a list to a string using the specified delimiter.

    Parameters:
    - input_list (list): The list to be converted to a string.
    - delimiter (str): The delimiter used to join the elements. Default is ', '.

    Returns:
    - str: The string representation of the list.
    """
    return delimiter.join(map(str, input_list))

tickers_strings = ''    
def plot_cum_returns(data, title):    
        daily_cum_returns = 1 + data.dropna().pct_change()
        daily_cum_returns = daily_cum_returns.cumprod()*100
        fig = px.line(daily_cum_returns, title=title)
        return fig
def plot_efficient_frontier_and_max_sharpe(mu, S): 
        # Optimize portfolio for max Sharpe ratio and plot it out with efficient frontier curve
        ef = EfficientFrontier(mu, S)
        fig, ax = plt.subplots(figsize=(6,4))
        ef_max_sharpe = copy.deepcopy(ef)
        plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)
        # Find the max sharpe portfolio
        ef_max_sharpe.max_sharpe(risk_free_rate=0.02)
        ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
        ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")
        # Generate random portfolios
        n_samples = 1000
        w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
        rets = w.dot(ef.expected_returns)
        stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
        sharpes = rets / stds
        ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")
        # Output
        ax.legend()
        return fig

def hrp_plot_efficient_frontier_and_max_sharpe(hrp: HRPOpt, n_points: int = 100):
    # Generate a range of expected returns
    min_return = hrp.expected_returns.min()
    max_return = hrp.expected_returns.max()
    target_returns = np.linspace(min_return, max_return, n_points)

    # Calculate minimum volatility portfolio for each target return
    target_volatilities = []
    for target_return in target_returns:
        hrp.efficient_return(target_return)
        weights = hrp.clean_weights()
        target_volatilities.append(hrp.portfolio_performance(risk_free_rate=0)[1])

    # Plot the efficient frontier
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(target_volatilities, target_returns, label="Efficient Frontier", marker="o", linestyle="-", color="b")
    ax.set_title("Efficient Frontier - HRP Algorithm")
    ax.set_xlabel("Volatility")
    ax.set_ylabel("Expected Return")
    ax.grid(True)
    ax.legend()
    return fig
###########################################################################################################

def mean_variance(stock_list):
            # Mean Variance Optimization
                # Get Stock Prices using pandas_datareader Library	
            stocks_df = yf.download(stock_list, start = five_yr, end = today)['Adj Close']
            sp500=yf.download('SPY', start = five_yr, end = today)['Adj Close']
                # Plot Individual Stock Prices
            fig_price = px.line(stocks_df, title='')
                # Plot Individual Cumulative Returns
            fig_cum_returns = plot_cum_returns(stocks_df, '')
                # Calculatge and Plot Correlation Matrix between Stocks
            corr_df = stocks_df.corr().round(2)
            fig_corr = px.imshow(corr_df, text_auto=True)
                # Calculate expected returns and sample covariance matrix for portfolio optimization later
            mu = expected_returns.mean_historical_return(stocks_df)
            S = risk_models.sample_cov(stocks_df)
                
                # Plot efficient frontier curve
            fig = plot_efficient_frontier_and_max_sharpe(mu, S)
            fig_efficient_frontier = BytesIO()
            fig.savefig(fig_efficient_frontier, format="png")
                
                # Get optimized weights
            ef = EfficientFrontier(mu, S)
            ef.add_objective(objective_functions.L2_reg, gamma=0.1)
            ef.max_sharpe(risk_free_rate=0.02)
            weights = ef.clean_weights()
            expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance()
            weights_df = pd.DataFrame.from_dict(weights, orient = 'index')
            weights_df.columns = ['weights']  
                # Calculate returns of portfolio with optimized weights
            stocks_df['Optimized Portfolio'] = 0
            for ticker, weight in weights.items():
                    stocks_df['Optimized Portfolio'] += stocks_df[ticker]*weight
                
                # Plot Cumulative Returns of Optimized Portfolio
            fig_cum_returns_optimized = plot_cum_returns(stocks_df['Optimized Portfolio'], 'Cumulative Returns of Optimized Portfolio Starting with $100')
            latest_prices = get_latest_prices(stocks_df)
            da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=100000)
            allocation, leftover = da.greedy_portfolio()
            print(allocation, leftover)
                    # Display everything on Streamlit
            st.subheader("Your Portfolio Consists of: {} Stocks".format(list_to_string(stock_list)))
            col1,col2=st.columns([1.3,1])
            with col1:
                st.plotly_chart(fig_cum_returns_optimized, use_container_width=True)
            with col2:
                st.write('')	
                st.write('')	
                st.subheader('\tStock Prices')
                st.write(stocks_df)
            st.write('___________________________')
            col1,col2, stats=st.columns([0.5,1.3, 0.7])   
            with col1: 
                st.write('')
                st.write('')
                st.write('')
                st.subheader("Max Sharpe Portfolio Weights")
                st.dataframe(weights_df)
            with col2:
                st.write('')
                st.write('')
                stock_tickers=[]
                weightage=[]
                for i in weights:
                    if weights[i]!=0:
                        stock_tickers.append(i)
                        weightage.append(weights[i])
                fig_pie = go.Figure(
                    go.Pie(
                    labels =stock_tickers,
                    values = weightage,
                    hoverinfo = "label+percent",
                    textinfo = "value"
                    ))
                holdings='''
                <style>
                .holding{
                    float: center;
                    font-weight: 600;
                    font-size: 35px;
                    font-family: arial;
                }
                </style>
                <body>
                <center><p1 class='holding'> Optimized Portfolio Holdings </p1></center>
                </body>
                '''
                st.markdown(holdings, unsafe_allow_html=True)
                st.plotly_chart(fig_pie)
            with stats:
                st.subheader('Expected annual return: {}%'.format((expected_annual_return*100).round(2)))
                st.write('___________')
                st.subheader('Annual volatility: {}%'.format((annual_volatility*100).round(2)))
                st.write('___________')
                st.subheader('Sharpe Ratio: {}'.format(sharpe_ratio.round(2)))
                st.write('___________')
                st.subheader('''Discrete allocation: 
                {}'''.format(allocation))
                st.write('___________')
                st.subheader("Funds remaining: ${:.2f}".format(leftover))
            st.write('___________________________')
            col1, col2=st.columns(2)
            with col1:
                st.subheader("Optimized Max Sharpe Portfolio Performance")
                st.image(fig_efficient_frontier)
            with col2:
                st.subheader("Correlation between stocks")
                st.plotly_chart(fig_corr) # fig_corr is not a plotly chart
            col1,col2=st.columns(2)
            with col1:
                st.subheader('Price of Individual Stocks')
                st.plotly_chart(fig_price)
            with col2:
                st.subheader('Cumulative Returns of Stocks Starting with $100')
                st.plotly_chart(fig_cum_returns)


def hrp(stock_list):
                # Hierarchical Risk Parity (HRP)
            stocks_df = yf.download(stock_list, start = five_yr, end = today)['Adj Close']
            returns = stocks_df.pct_change().dropna()
            cov_matrix = returns.cov()
            hrp = HRPOpt(cov_matrix)
            weights = hrp.optimize()

    #//////////////////////////////////////////////////////////
            sp500=yf.download('SPY', start = five_yr, end = today)['Adj Close']
                # Plot Individual Stock Prices
            fig_price = px.line(stocks_df, title='')
                # Plot Individual Cumulative Returns
            fig_cum_returns = plot_cum_returns(stocks_df, '')
                # Calculatge and Plot Correlation Matrix between Stocks
            corr_df = stocks_df.corr().round(2)
            fig_corr = px.imshow(corr_df, text_auto=True)
                # Calculate expected returns and sample covariance matrix for portfolio optimization later
                
                # Plot efficient frontier curve
            fig = hrp_plot_efficient_frontier_and_max_sharpe(hrp) 
            fig_efficient_frontier = BytesIO() 
            fig.savefig(fig_efficient_frontier, format="png")
                
                # Get optimized weights
            expected_annual_return, annual_volatility, sharpe_ratio = hrp.portfolio_performance(verbose=True)
            hrp.max_sharpe(risk_free_rate=0.02) #//
            expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance() 
            weights_df = pd.DataFrame.from_dict(weights, orient = 'index')
            weights_df.columns = ['weights']  
                # Calculate returns of portfolio with optimized weights
            stocks_df['Optimized Portfolio'] = 0
            for ticker, weight in weights.items():
                    stocks_df['Optimized Portfolio'] += stocks_df[ticker]*weight
                
                # Plot Cumulative Returns of Optimized Portfolio
            fig_cum_returns_optimized = plot_cum_returns(stocks_df['Optimized Portfolio'], 'Cumulative Returns of Optimized Portfolio Starting with $100')
            latest_prices = get_latest_prices(stocks_df)
            da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=100000)
            allocation, leftover = da.greedy_portfolio()
            print(allocation, leftover)
                    # Display everything on Streamlit
            st.subheader("Your Portfolio Consists of: {} Stocks".format(list_to_string(stock_list)))
            col1,col2=st.columns([1.3,1])
            with col1:
                st.plotly_chart(fig_cum_returns_optimized, use_container_width=True)
            with col2:
                st.write('')	
                st.write('')	
                st.subheader('\tStock Prices')
                st.write(stocks_df)
            st.write('___________________________')
            col1,col2, stats=st.columns([0.5,1.3, 0.7])   
            with col1: 
                st.write('')
                st.write('')
                st.write('')
                st.subheader("Max Sharpe Portfolio Weights")
                st.dataframe(weights_df)
            with col2:
                st.write('')
                st.write('')
                stock_tickers=[]
                weightage=[]
                for i in weights:
                    if weights[i]!=0:
                        stock_tickers.append(i)
                        weightage.append(weights[i])
                fig_pie = go.Figure(
                    go.Pie(
                    labels =stock_tickers,
                    values = weightage,
                    hoverinfo = "label+percent",
                    textinfo = "value"
                    ))
                holdings='''
                <style>
                .holding{
                    float: center;
                    font-weight: 600;
                    font-size: 35px;
                    font-family: arial;
                }
                </style>
                <body>
                <center><p1 class='holding'> Optimized Portfolio Holdings </p1></center>
                </body>
                '''
                st.markdown(holdings, unsafe_allow_html=True)
                st.plotly_chart(fig_pie)
            with stats:
                st.subheader('Expected annual return: {}%'.format((expected_annual_return*100).round(2)))
                st.write('___________')
                st.subheader('Annual volatility: {}%'.format((annual_volatility*100).round(2)))
                st.write('___________')
                st.subheader('Sharpe Ratio: {}'.format(sharpe_ratio.round(2)))
                st.write('___________')
                st.subheader('''Discrete allocation: 
                {}'''.format(allocation))
                st.write('___________')
                st.subheader("Funds remaining: ${:.2f}".format(leftover))
            st.write('___________________________')
            col1, col2=st.columns(2)
            with col1:
                st.subheader("Optimized Max Sharpe Portfolio Performance")
                st.image(fig_efficient_frontier)
            with col2:
                st.subheader("Correlation between stocks")
                st.plotly_chart(fig_corr) # fig_corr is not a plotly chart
            col1,col2=st.columns(2)
            with col1:
                st.subheader('Price of Individual Stocks')
                st.plotly_chart(fig_price)
            with col2:
                st.subheader('Cumulative Returns of Stocks Starting with $100')
                st.plotly_chart(fig_cum_returns)
    
##########################################################################################
def home():
    stocks = ""
    st.empty()
    st.write('Hello')
    sp500 = si.tickers_sp500()
    select_stocks = st.multiselect('Select your portfolio from SP500 stocks', sp500)
    save_pf = st.button('Submit Portfolio')
    if save_pf:
        if select_stocks != []:
            for i in select_stocks:
                stocks = stocks + '' + i
            str_stocks = list_to_string(select_stocks)
            print(type(st.session_state.user))
            query = "UPDATE login_info SET portfolio = '{}' WHERE username = '{}'".format(str_stocks, st.session_state.user)
            cursor.execute(query)
            connection.commit()
            #mean_variance(select_stocks)
            mean_variance(select_stocks)
            ##########################################################################
        else:
            st.warning('You have not selected any stocks')
login()




# Invested decently heavily in multiple stocks but gave unrreal returns
# Of course, this return is inflated and is not likely to hold up in the future. 
# Mean variance optimization doesn’t perform very well since it makes many simplifying assumptions, such as returns being normally distributed and the need for an invertible covariance matrix. Fortunately, methods like HRP and mCVAR address these limitations. 

# Hierarchical Risk Parity (HRP)
# The HRP method works by finding subclusters of similar assets based on returns and constructing a hierarchy from these clusters to generate weights for each asset. 

'''
returns = portfolio.pct_change().dropna()
hrp = HRPOpt(returns)
hrp_weights = hrp.optimize()

hrp.portfolio_performance(verbose=True)
print(dict(hrp_weights))

da_hrp = DiscreteAllocation(hrp_weights, latest_prices, total_portfolio_value=100000)

allocation, leftover = da_hrp.greedy_portfolio()
print("Discrete allocation (HRP):", allocation)
print("Funds remaining (HRP): ${:.2f}\n".format(leftover))
'''
# Shown so much more diversification
#  Further, while the performance decreased, we can be more confident that this model will perform just as well when we refresh our data. This is because HRP is more robust to the anomalous increase in Moderna stock prices. 

# Mean Conditional Value at Risk (mCVAR)
# The mCVAR is another popular alternative to mean variance optimization. It works by measuring the worst-case scenarios for each asset in the portfolio, which is represented here by losing the most money. The worst-case loss for each asset is then used to calculate weights to be used for allocation for each asset. 
'''
from pypfopt.efficient_frontier import EfficientCVaR

S = portfolio.cov()
ef_cvar = EfficientCVaR(mu, S)
cvar_weights = ef_cvar.min_cvar()

cleaned_weights = ef_cvar.clean_weights()
print(dict(cleaned_weights))

returns, risk = ef_cvar.portfolio_performance(verbose=True)
if risk < -1:
    risk = -1
elif risk > 1:
    risk = 1

print("Expected annual return: {}%".format(round(returns*100)))
print("Conditional Value at Risk: {}%".format(risk*100))

latest_prices = get_latest_prices(portfolio)
da_ef_cvar = DiscreteAllocation(cvar_weights, latest_prices, total_portfolio_value=100000)

allocation, leftover = da_ef_cvar.greedy_portfolio()
print("Discrete allocation:", allocation)
print("Funds remaining: ${:.2f}\n".format(leftover))
# We see that this algorithm suggests we invest heavily into JP Morgan Chase (JPM) and also buy a single share each of Moderna (MRNA) and Johnson & Johnson (JNJ). Also we see that the expected return is 15.5 percent. As with HRP, this result is much more reasonable than the inflated 225 percent returns given by mean variance optimization since it is not as sensitive to the anomalous behaviour of the Moderna stock price. 













fig_cum_returns_optimized = plot_cum_returns(stocks_df['Optimized Portfolio'], 'Cumulative Returns of Optimized Portfolio Starting with $100')
latest_prices = get_latest_prices(stocks_df)


##################################################################################################
        st.subheader("Your Portfolio Consists of: {} Stocks".format(tickers_string))
        col1,col2=st.columns([1.3,1])
        with col1:
            st.plotly_chart(fig_cum_returns_optimized, use_container_width=True)
        with col2:
            st.write('')	
            st.write('')	
            st.subheader('\tStock Prices')
            st.write(stocks_df)
        st.write('___________________________')
        col1,col2, stats=st.columns([0.5,1.3, 0.7])   
        with col1: 
            st.write('')
            st.write('')
            st.write('')
            st.subheader("Max Sharpe Portfolio Weights")
            st.dataframe(weights_df)
        with col2:
            st.write('')
            st.write('')
            stock_tickers=[]
            weightage=[]
            for i in weights:
                if weights[i]!=0:
                    stock_tickers.append(i)
                    weightage.append(weights[i])
            fig_pie = go.Figure(
                go.Pie(
                labels =stock_tickers,
                values = weightage,
                hoverinfo = "label+percent",
                textinfo = "value"
                ))
'''
            


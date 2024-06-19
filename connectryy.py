import streamlit as st
import streamlit_authenticator as stauth
from streamlit_authenticator import Authenticate
import yaml
import numpy as np
import random
from yaml.loader import SafeLoader
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://utkarshhh20:2005utkarsh@cluster0.lkssv5k.mongodb.net/?retryWrites=true&w=majority"
# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
# Send a ping to confirm a successful connection
username = 'utkarshhh20'
password = '2005utkarsh'

if "form" not in st.session_state:
    st.session_state.form = "signin"

if "name" not in st.session_state:
    st.session_state.name = ""

if "username" not in st.session_state:
    st.session_state.username = ""

if "email" not in st.session_state:
    st.session_state.email = ""

@st.cache_resource
def connect_db():
    uri = f"mongodb+srv://{username}:{password}@cluster0.lkssv5k.mongodb.net/?retryWrites=true&w=majority"
    try:
        client = MongoClient(uri, server_api=ServerApi('1'))
        client.admin.command('ping')
        db = client["portfolio_app"]
        print("Pinged your deployment. You successfully connected to MongoDB!")
        return db
    except Exception as e:
        print(e)

def login_db(db):
    login_info = db["user_login"]
    return login_info

def state_to_signup():
    st.session_state.form = 'signup'

def state_to_signin():
    st.session_state.form = 'signin'

db = connect_db()
login = login_db(db)

@st.cache_data
def config_file():
    config = {'credentials': 
          {'usernames': ''},
          'cookie': '',
          'preauthorized': ''}
    
    user_det={}
    for info in login.find(): 
        user_det[info['username']] = {'email': info['email'], 'name': info['name'], 'password': info['password']}

    cookie_key = random.random()

    config['credentials']['usernames'] = user_det
    config['cookie']= {'expiry_days': 30, 'key': str(cookie_key), 'name': str(cookie_key)}
    config['preauthorized'] = {'emails': ['utkarshhh20@gmail.com']}
    return config

config = config_file()
authenticator = Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
    )
    
if st.session_state.form == 'signin':
    name, authentication_status, username = authenticator.login('Login', 'main')
    st.button('Sign Up', on_click=state_to_signup)
#print(stauth.Hasher('abc').generate())

    if authentication_status:
        st.session_state.form = 'loggedin'
        st.session_state.name = name
    elif authentication_status == False:
        st.error('Username/password is incorrect')
    elif authentication_status == None:
        st.warning('Please enter your username and password')

if st.session_state.form == 'signup':
    try:
        authenticator._register_credentials
        if authenticator.register_user('Register user', preauthorization=False):
            username = st.session_state.username
            email = st.session_state.email
            name = st.session_state.name
            st.write(username, email, name)
            st.success('User registered successfully')
            st.session_state.form = 'signin'
    except Exception as e:
        st.error(e)
    st.button('Sign In', on_click=state_to_signin)

if st.session_state.form == 'loggedin':
    st.write(f'Welcome *{st.session_state.name}*')
    st.title('Some content')
    authenticator.logout('Logout', 'main')
    st.write('HI')

#forgot pass
try:
    username_forgot_pw, email_forgot_password, random_password = authenticator.forgot_password('Forgot password')
    if username_forgot_pw:
        st.success('New password sent securely')
        # Random password to be transferred to user securely
    elif username_forgot_pw == False:
        st.error('Username not found')
except Exception as e:
    st.error(e)

'''
#reset pass
if authentication_status:
    try:
        if authenticator.reset_password(username, 'Reset password'):
            st.success('Password modified successfully')
    except Exception as e:
        st.error(e)
        '''

'''if st.session_state["authentication_status"]:
    authenticator.logout('Logout', 'main')
    st.write(f'Welcome *{st.session_state["name"]}*')
    st.title('Some content')
elif st.session_state["authentication_status"] == False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] == None:
    st.warning('Please enter your username and password')'''
'''

hashed_passwords = stauth.Hasher('123456').generate()
print(config)

user_det={}
for info in login_db.find....: 
    user_det[username] = {'email': email, 'name': name, 'password': password}

cookie_key = random.random()

config['credentials']['usernames'] = user_det
config['cookie']= {'expiry_days': 30, 'key': cookie_key, 'name': username}
config['preauthorized'] = {'emails': ['utkarshhh20@gmail.com']}
    
'''
import streamlit as st
import streamlit_authenticator as stauth
from streamlit_authenticator import Authenticate
import yaml
import numpy as np
import random
from yaml.loader import SafeLoader
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import time

uri = "mongodb+srv://utkarshhh20:2005utkarsh@cluster0.lkssv5k.mongodb.net/?retryWrites=true&w=majority"
# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
# Send a ping to confirm a successful connection
username = 'utkarshhh20'
password = '2005utkarsh'

if "form" not in st.session_state:
    st.session_state.form = "signin"

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

def add_user(new_user, new_pass, new_email, new_name):
    user_data = {
        'username':new_user,
        'password': stauth.Hasher([new_pass]).generate()[0],
        'email': new_email,
        'name': new_name
    }
    login.insert_one(user_data)

def fetch_user_credentials():
    user_credentials = {}
    for user in login.find():  # Assuming 'login' is your MongoDB collection
        username = user['username']  # Ensure username is lowercase
        user_credentials[username] = {
            'password': user['password'],
            'name': user['name'],
            'email': user['email']
        }
    user_creds = {'usernames': user_credentials}
    return user_creds

def authenticate_det():
    authenticator = Authenticate(
            fetch_user_credentials(),
            'cookie_name',
            'cookie_key',
            0
            )
    return authenticator

class RegisterError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

def register_user(form_name: str, credentials: dict, location: str='main', preauthorization=True) -> bool:
        if location == 'main':
            register_user_form = st.form('Register user')
        register_user_form.subheader(form_name)
        new_email = register_user_form.text_input('Email')
        new_username = register_user_form.text_input('Username').lower()
        new_name = register_user_form.text_input('Name')
        new_password = register_user_form.text_input('Password', type='password')
        new_password_repeat = register_user_form.text_input('Repeat password', type='password')

        if register_user_form.form_submit_button('Register'):
            if len(new_email) and len(new_username) and len(new_name) and len(new_password) > 0:
                if new_username not in credentials['usernames']:
                    if new_password == new_password_repeat:
                        return new_username, new_name, new_password, new_email, True
                    else:
                        raise RegisterError('Passwords do not match')
                else:
                    raise RegisterError('Username already taken')
            else:
                raise RegisterError('Please enter an email, username, name, and password')
            
def auth_page():
    global login, db, authenticator
    if st.session_state.form == 'signin':
        authenticator.login('Login', 'main')
        if st.session_state["authentication_status"]:
            st.session_state.form = 'loggedin'
        elif st.session_state["authentication_status"] is False:
            st.error('Username/password is incorrect')
        elif st.session_state["authentication_status"] is None:
            st.warning('Please enter your username and password')
        st.button('Sign Up', on_click=state_to_signup)

    elif st.session_state.form == 'signup':
        try:
            credentials = fetch_user_credentials()
            register = register_user('Register user', credentials, preauthorization=False)
            st.button('Sign In', on_click=state_to_signin)
            if type(register) == type((1,2)):
                st.write(register)
                add_user(register[0], register[2], register[3], register[1])
                db = connect_db()
                login = login_db(db)
                authenticator = authenticate_det()
                st.success('Account created successfully')
        except Exception as e:
            st.error(e)
    
    elif st.session_state.form == 'loggedin':
        authenticator.logout('Logout', 'main', key='unique_key')
        if st.session_state['logout'] == True:
            st.session_state.form = 'signin'
        st.write(f'Welcome *{st.session_state["name"]}*')
        st.title('Some content')

#db = connect_db()
#login = login_db(db)
#authenticator = authenticate_det()

auth_page()
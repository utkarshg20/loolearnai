import streamlit as st
import pymongo
from pymongo.server_api import ServerApi
from pymongo.mongo_client import MongoClient

username = 'utkarshhh20'
password = '2005utkarsh'
# Connect to the DB.
@st.cache_resource
def connect_db():
    uri = f"mongodb+srv://{username}:{password}@cluster0.lkssv5k.mongodb.net/?retryWrites=true&w=majority"
    client = MongoClient(uri, server_api=ServerApi('1'))
    project = client["portfolio_app"]
    db = project['user_login']
    #db = client.get_database("portfolio_app")
    return db

user_db = connect_db()

# Initialize Session States.
if 'username' not in st.session_state:
       st.session_state.username = ''
if 'form' not in st.session_state:
       st.session_state.form = 'signin_form'

def select_signup():
    st.session_state.form = 'signup_form'

def select_signin():
    st.session_state.form = 'signin_form'

def user_update(name):
    st.session_state.username = name

def back_home():
    st.session_state.username = ''
    st.session_state.form = 'signin_form'

# Initialize Sing In or Sign Up forms
if st.session_state.form == 'signup_form' and st.session_state.username == '':
  
    signup_form = st.form(key='signup_form', clear_on_submit=True)
    new_username = signup_form.text_input(label='Enter Username*')
    new_user_email = signup_form.text_input(label='Enter Email Address*')
    new_user_pas = signup_form.text_input(label='Enter Password*', type='password')
    user_pas_conf = signup_form.text_input(label='Confirm Password*', type='password')
    note = signup_form.markdown('**required fields*')
    signup = signup_form.form_submit_button(label='Sign Up')
    
    if signup:
        if '' in [new_username, new_user_email, new_user_pas]:
            st.error('Some fields are missing')
        else:
            if user_db.find_one({'_id' : new_username}):
                st.error('Username already exists')
            if user_db.find_one({'email' : new_user_email}):
                st.error('Email is already registered')
            else:
                if new_user_pas != user_pas_conf:
                    st.error('Passwords do not match')
                else:
                    user_update(new_username)
                    user_db.insert_one({'_id' : new_username, 'email' : new_user_email, 'pass' : new_user_pas})
                    st.success('You have successfully registered!')
                    st.session_state.form = ''
                    st.success(f"You are logged in as {new_username.upper()}")
                    del new_user_pas, user_pas_conf
                    
elif st.session_state.username == '' and st.session_state.form == 'signin_form':
    login_form = st.form(key='signin_form', clear_on_submit=True)
    username = login_form.text_input(label='Enter Username')
    user_pas = login_form.text_input(label='Enter Password', type='password')
    
    if user_db.find_one({'_id' : username, 'pass' : user_pas}):
        login = login_form.form_submit_button(label='Sign In', on_click=user_update(username))
        if login:
            st.session_state.form = ''
            st.success(f"You are logged in as {username.upper()}")
            del user_pas
    else:
        login = login_form.form_submit_button(label='Sign In')
        if login:
            st.error("Username or Password is incorrect. Please try again or create an account.")

else:
    logout_form = st.form(key='logout_form', clear_on_submit=True)
    logout = logout_form.form_submit_button(label='Log Out')
    if logout:
        back_home()
        
# 'Create Account' button
if st.session_state.username == "" and st.session_state.form == 'signin_form':
    signup_request = st.button('Create Account', on_click=select_signup)

elif st.session_state.username == "" and st.session_state.form == 'signup_form':
    signup_request = st.button('Sign in', on_click=select_signin)

'''elif st.session_state.username != "" and st.session_state.form == '':
    st.write(username)
    logout = st.button('Log Out', on_click=back_home)
'''
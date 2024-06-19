
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import streamlit as st
uri = "mongodb+srv://utkarshhh20:2005utkarsh@cluster0.lkssv5k.mongodb.net/?retryWrites=true&w=majority"
# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
# Send a ping to confirm a successful connection
username = 'utkarshhh20'
password = '2005utkarsh'

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

def valid_username(login_db, username): 
    existing_users = login_db.find() 
    for user in existing_users:
        if username == user['_id']:
            return False
    return True

def create_account(user, password, login_db):
    login_db.insert_one({
    "_id" : user,
    "password" : password,})

def login_success(login,username):
    return valid_username(login, username)

def sign_in(username, password, login_db):
    for i in login_db.find() :
        if username == i['_id']:
            if password == i['password']:
                st.write('LOGIN SUCCESSFUL')
            else:
                st.write("Username/password don't match, kindly try again")
        else: 
            pass

def forgot_password():
    print('TO BE COMPLETED')

def login_page():
    with st.form("my_form"):
        st.write("Inside the form")
        username = st.text_input("Username")
        password = st.text_input("Password")

        # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")
        if submitted:
            if login_success(login,username) == True:
                #create_account(username, password, login)
                home_page()
            else:
                st.write('ERROR')

def home_page():
    st.write('WELCOME!')

db = connect_db()
login = login_db(db)

user_det={}
for info in login.find(): 
    user_det[info['username']] = {'email': info['email'], 'name': info['name'], 'password': info['password']}
print(user_det)
login_page()

import streamlit_authenticator as stauth
#print(stauth.Hasher(['abc', 'def']).generate())
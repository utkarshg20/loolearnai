import streamlit as st
def load_view():
    # Set the page configuration to wide mode with a specified page title and favicon
    #st.set_page_config(page_title='Superpowered AI', layout='wide', page_icon=':zap:')

    blank1, main, blank2 = st.columns([1,1,1])
    loolearn = '''
    <style>
        .centered {
            text-align: center;
            margin: 0;
            color: black;
            font-size: 60px;
            font-weight: bold;
            font-family: 'Eloquia Text', sans-serif;
        }
    </style>
    <div class="centered">
        LooLearn AI
    </div>
    '''
    #4vw
    loolearn_desc = '''
    <style>
        .subheader {
            text-align: center;
            margin: auto;
            color: black;
            font-size: 25px;
            font-family: 'Eloquia Text', sans-serif;
        }
    </style>
    <div class="subheader">
        Seamlessly connect LLMs to your data and get accurate responses with citations.
    </div>


    '''
    create_account = '''
        <style>
            .custom-home-btn {
                border: 2px solid #1B0840;
                color: white;
                padding: 10px 35px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                transition-duration: 0.4s;
                cursor: pointer;
                background-color: white; 
                color: black; 
                border-radius: 8px;
            }
            .custom-home-btn:hover {
                background-color: #1B0840;
                color: white;
            }
        </style>
        
        <a class="custom-home-btn" href="about">Get started</a>
'''
    contact = '''
        <style>
            .custom-home-btn {
                border: 2px solid #1B0840;
                color: white;
                padding: 10px 35px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                transition-duration: 0.4s;
                cursor: pointer;
                background-color: white; 
                color: black; 
                border-radius: 8px;
            }
            .custom-home-btn:hover {
                background-color: #1B0840;
                color: white;
            }
        </style>
        
        <a class="custom-home-btn" href="about">Contact us</a>
    
'''
    #with main:
    #st.markdown(page, unsafe_allow_html=True)
    st.markdown(loolearn, unsafe_allow_html=True)
    st.write("")
    st.markdown(loolearn_desc, unsafe_allow_html=True)

    buttons = '''
    <div class="button-container">
        <a class="custom-home-btn" href="/?nav=about">Get Started</a>
        <a class="custom-home-btn" href="contact">Contact Us</a>
    </div>
'''
    st.write("")
    st.write("")
    st.markdown(buttons, unsafe_allow_html=True)
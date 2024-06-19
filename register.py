import streamlit as st

credentials = {'usernames': ''}
class RegisterError(Exception):
    """
    Exceptions raised for the register user widget.

    Attributes
    ----------
    message: str
        The custom error message to display.
    """
    def __init__(self, message: str):
        st.warning(message)
        self.message = message
        super().__init__(self.message)

def register_user(form_name: str, location: str='main', preauthorization=True) -> bool:
        """
        Creates a password reset widget.

        Parameters
        ----------
        form_name: str
            The rendered name of the password reset form.
        location: str
            The location of the password reset form i.e. main or sidebar.
        preauthorization: bool
            The preauthorization requirement, True: user must be preauthorized to register, 
            False: any user can register.
        Returns
        -------
        bool
            The status of registering the new user, True: user registered successfully.
        """
        if location not in ['main', 'sidebar']:
            raise ValueError("Location must be one of 'main' or 'sidebar'")
        if location == 'main':
            register_user_form = st.form('Register user')
        elif location == 'sidebar':
            register_user_form = st.sidebar.form('Register user')

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

try:
    x = register_user('Register user', preauthorization=False)
    if type(x) == type(('h', 'h')):
        st.write(x)
except:
    pass
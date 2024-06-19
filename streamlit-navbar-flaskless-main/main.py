import streamlit as st
import utils as utl
import streamlit.components.v1 as components
from views import Loolearnai,about,analysis,options,configuration

st.set_page_config(layout="wide", page_title='Navbar sample')
st.set_option('deprecation.showPyplotGlobalUse', False)
utl.inject_custom_css()
utl.navbar_component()

jscode = '''
<script>
document.addEventListener('DOMContentLoaded', function() {
    // Example: Selecting by text content "Get Started"
    // Note: Adjust the selector as needed to accurately target the "Get Started" button
    document.querySelectorAll('a').forEach(function(link) {
        if (link.textContent === 'Get Started') {
            link.addEventListener('click', function(event) {
                event.preventDefault(); // Prevent opening in a new tab
                window.location.href = this.getAttribute('href'); // Navigate in the same tab
            });
        }
    });
});
</script>
'''

def navigation():
    route = utl.get_current_route()
    # Injecting the JavaScript code into your app
    components.html(jscode, height=0)
    if route == "Loolearnai":
        Loolearnai.load_view()
    elif route == "about":
        about.load_view()
    elif route == "analysis":
        analysis.load_view()
    elif route == "options":
        options.load_view()
    elif route == "configuration":
        configuration.load_view()
    elif route == None:
        Loolearnai.load_view()
        
navigation()
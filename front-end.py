import streamlit as st

# Set the page configuration to wide mode with a specified page title and favicon
st.set_page_config(page_title='Superpowered AI', layout='wide', page_icon=':zap:')

blank1, main, blank2 = st.columns([1,1,1])
page = '''
<style>
body {
  background-color: lightblue;
}
</style>
'''
loolearn = '''
<style>
    .centered {
        text-align: center;
        margin: 0;
        color: black;
        font-size: 70px;
        font-weight: bold;
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
        font-size: 30px;
        font-family: 'Trivia Sans Medium';
    }
</style>
<div class="subheader">
     Seamlessly connect LLMs to your data and get accurate responses with citations.
</div>


'''

#with main:
st.markdown(page, unsafe_allow_html=True)
st.markdown(loolearn, unsafe_allow_html=True)
st.markdown(loolearn_desc, unsafe_allow_html=True)

course_subsection = {"MATH 136": ['1.1 Introduction', '1.2 Algebraic and Geometric Representation of Vectors', '1.3 Operations on Vectors', '1.4 Vectors in Cn', '1.5 Dot Product in Rn', '1.6 Projection, Components and Perpendicular', '1.7 Standard Inner Product in Cn', '1.8 Fields', '1.9 The Cross Product in R3', '2.1 Linear Combinations and Span', '2.2 Lines in R2', '2.3 Lines in Rn', '2.4 The Vector Equation of a Plane in Rn', '2.5 Scalar Equation of a Plane in R3', '3.1 Introduction', '3.2 Systems of Linear Equations', '3.3 An Approach to Solving Systems of Linear Equations', '3.4 Solving Systems of Linear Equations Using Matrices', '3.5 The Gauss–Jordan Algorithm for Solving Systems of Linear Equations', '3.6 Rank and Nullity', '3.7 Homogeneous and Non-Homogeneous Systems, Nullspace', '3.8 Solving Systems of Linear Equations over C', '3.9 Matrix–Vector Multiplication', '3.10 Using a Matrix–Vector Product to Express a System of Linear Equations', '3.11 Solution Sets to Systems of Linear Equations', '4.1 The Column and Row Spaces of a Matrix', '4.2 Matrix Equality and Multiplication', '4.3 Arithmetic Operations on Matrices', '4.4 Properties of Square Matrices', '4.5 Elementary Matrices', '4.6 Matrix Inverse', '5.1 The Function Determined by a Matrix', '5.2 Linear Transformations', '5.3 The Range of a Linear Transformation and ``Onto" Linear Transformations', "5.4 The Kernel of a Linear Transformation and ``One-to-One'' Linear Transformations", '5.5 Every Linear Transformation is Determined by a Matrix', '5.6 Special Linear Transformations: Projection, Perpendicular, Rotation and Reflection', '5.7 Composition of Linear Transformations', '6.1 The Definition of the Determinant', '6.2 Computing the Determinant in Practice: EROs', '6.3 The Determinant and Invertibility', '6.4 An Expression for the inverse of A', "6.5 Cramer's Rule", '6.6 The Determinant and Geometry', '7.1 What is an Eigenpair?', '7.2 The Characteristic Polynomial and Finding Eigenvalues', '7.3 Properties of the Characteristic Polynomial', '7.4 Finding Eigenvectors', '7.5 Eigenspaces', '7.6 Diagonalization', '8.1 Subspaces', '8.2 Linear Dependence and the Notion of a Basis of a Subspace', '8.3 Detecting Linear Dependence and Independence', '8.4 Spanning Sets', '8.5 Basis', '8.6 Bases for col(A) and null(A)', '8.7 Dimension', '8.8 Coordinates', '9.1 Matrix Representation of a Linear Operator', '9.2 Diagonalizability of Linear Operators', '9.3 Diagonalizability of Matrices Revisited', '9.4 The Diagonalizability Test', '10.1 Definition of a Vector Space', '10.2 Span, Linear Independence and Basis', '10.3 Linear Operators'], 
                     "MATH 138": ['1.1 Areas Under Curves', '1.2 Estimating Areas', '1.3 Approximating Areas Under Curves', '1.4 The Relationship Between Displacement and Velocity', '1.5 Riemann Sums and the Definite Integral', '1.6 Properties of the Definite Integral', '1.7 Additional Properties of the Integral', '1.8 Geometric Interpretation of the Integral', '1.9 The Average Value of a Function', '1.10 An Alternate Approach to the Average Value of a Function', '1.11 The Fundamental Theorem of Calculus (Part 1)', '1.12 The Fundamental Theorem of Calculus (Part 2)', '1.13 Antiderivatives', '1.14 Evaluating Definite Integrals', '1.15 Change of Variables', '1.16 Change of Variables for the Indefinite Integral', '1.17 Change of Variables for the Definite Integral', '2.1 Inverse Trigonometric Substitutions', '2.2 Integration by Parts', '2.3 Partial Fractions', '2.4 Introduction to Improper Integrals', '2.5 Properties of Type I Improper Integrals', '2.6 Comparison Test for Type I Improper Integrals', '2.7 The Gamma Function', '2.8 Type II Improper Integrals', '3.1 Areas Between Curves', '3.2 Volumes of Revolution: Disk Method', '3.3 Volumes of Revolution: Shell Method', '3.4 Arc Length', '4.1 Introduction to Differential Equations', '4.2 Separable Differential Equations', '4.3 First-Order Linear Differential Equations', '4.4 Initial Value Problems', '4.5 Graphical and Numerical Solutions to Differential Equations', '4.6 Direction Fields', "4.7 Euler's Method", '4.8 Exponential Growth and Decay', "4.9 Newton's Law of Cooling", '4.10 Logistic Growth', '5.1 Introduction to Series', '5.2 Geometric Series', '5.3 Divergence Test', '5.4 Arithmetic of Series', '5.5 Positive Series', '5.6 Comparison Test', '5.7 Limit Comparison Test', '5.8 Integral Test for Convergence of Series', '5.9 Integral Test and Estimation of Sums and Errors', '5.10 Alternating Series', '5.11 Absolute versus Conditional Convergence', '5.12 Ratio Test', '5.13 Root Test', '6.1 Introduction to Power Series', '6.2 Finding the Radius of Convergence', '6.3 Functions Represented by Power Series', '6.4 Building Power Series Representations', '6.5 Differentiation of Power Series', '6.6 Integration of Power Series', '6.7 Review of Taylor Polynomials', "6.8 Taylor's Theorem and Errors in Approximations", '6.9 Introduction to Taylor Series', '6.10 Convergence of Taylor Series', '6.11 Binomial Series', '6.12 Additional Examples and Applications of Taylor Series']}
course, start_subsection, stop_subsection = st.columns(3)

with course:
     course_selection = st.selectbox("Select your course", course_subsection.keys())
with start_subsection:
     start_topic = st.selectbox("Select the starting subsection", options = course_subsection[course_selection])
with stop_subsection:
     stop_topic = st.selectbox("Select the last subsection", options = course_subsection[course_selection])

question = st.text_input(label = "Enter your query", label_visibility='hidden', placeholder="Please enter your question")
if question != "":
     if (start_subsection,stop_subsection) != ('',''):
          if float(start_topic[:3]) > float(stop_topic[:3]):
               st.error("Please enter the correct range for the subsections")
          else:
               print(1)

'''rse_col, autoquery_col, length_col = st.columns(3)
with rse_col:
     rse = st.checkbox("RSE", value=True)
with autoquery_col:
     auto_query = st.checkbox("Auto Query", value=True)
with length_col:
     length = st.checkbox("Length", value=True)'''

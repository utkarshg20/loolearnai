import time
import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
start_time = time.time()
subsections = {'Vectors in R𝑛': 5, 'Algebraic and Geometric Representation of Vectors': 6, 'Operations on Vectors': 7, 'Vectors in C𝑛': 12, 'Dot Product in R𝑛': 13, 'Projection, Components and Perpendicular': 19, 'Standard Inner Product in C𝑛': 23, 'Fields': 27, 'The Cross Product in R3': 28, 'Linear Combinations and Span': 32, 'Lines in R2': 34, 'Lines in R𝑛': 36, 'The Vector Equation of a Plane in R𝑛': 40, 'Scalar Equation of a Plane in R3': 44, 'Introduction': 48, 'Systems of Linear Equations': 49, 'An Approach to Solving Systems of Linear Equations': 55, 'Solving Systems of Linear Equations Using Matrices': 61, 'The Gauss–Jordan Algorithm for Solving Systems of Linear Equations': 66, 'Rank and Nullity': 75, 'Homogeneous and Non-Homogeneous Systems, Nullspace': 78, 'Solving Systems of Linear Equations over C': 84, 'Matrix–Vector Multiplication': 84, 'Using a Matrix\u2013Vector Product to Express a System of\nLinear Equations': 88, 'Solution Sets to Systems of Linear Equations': 89, 'The Column and Row Spaces of a Matrix': 97, 'Matrix Equality and Multiplication': 99, 'Arithmetic Operations on Matrices': 103, 'Properties of Square Matrices': 107, 'Elementary Matrices': 109, 'Matrix Inverse': 112, 'The Function Determined by a Matrix': 121, 'Linear Transformations': 122, 'The Range of a Linear Transformation and \u201cOnto\u201d Linear Transformations': 127, "The Kernel of a Linear Transformation and \u201cOne-to-\nOne\u201d Linear Transformations": 130, 'Every Linear Transformation is Determined by a Matrix': 133, 'Special Linear Transformations: Projection, Perpendicular, Rotation and Reflection': 136, 'Composition of Linear Transformations': 143, 'The Definition of the Determinant': 147, 'Computing the Determinant in Practice: EROs': 153, 'The Determinant and Invertibility': 156, 'An Expression for 𝐴−1': 158, "Cramer’s Rule": 161, 'The Determinant and Geometry': 163, 'What is an Eigenpair?': 167, 'The Characteristic Polynomial and Finding Eigenvalues': 169, 'Properties of the Characteristic Polynomial': 172, 'Finding Eigenvectors': 178, 'Eigenspaces': 181, 'Diagonalization': 183, 'Subspaces': 190, 'Linear Dependence and the Notion of a Basis of a Subspace': 193, 'Detecting Linear Dependence and Independence': 197, 'Spanning Sets': 204, 'Basis': 208, 'Bases for Col(𝐴) and Null(𝐴)': 212, 'Dimension': 216, 'Coordinates': 219, 'Matrix Representation of a Linear Operator': 230, 'Diagonalizability of Linear Operators': 235, 'Diagonalizability of Matrices Revisited': 238, 'The Diagonalizability Test': 244, 'Definition of a Vector Space': 254, 'Span, Linear Independence and Basis': 258, 'Linear Operators': 264, 'Index': 270}
selected_subsections = {'Operations on Vectors': 7, 'Vectors in R𝑛': 5, 'Algebraic and Geometric Representation of Vectors': 6}
time.time
selected_pages = []
match_found = 0
for i in selected_subsections:
    for j in subsections:
        if i == j:
            selected_pages.append(subsections[i])
            match_found = 1
        elif match_found == 1:
            selected_pages.append(subsections[j])
            match_found = 0
        else:
            pass

print(selected_pages)
end_time = time.time()  
elapsed_time = end_time - start_time
sorted_subsections = sorted(subsections.items(), key=lambda item: item[1])

# Find the page ranges for selected subsections
selected_pages = []
for title, page in selected_subsections.items():
    start_page = page
    end_page = None
    start_index = sorted_subsections.index((title, page))
    
    # Look for the next subsection's start page to determine the end page of the current subsection
    if start_index + 1 < len(sorted_subsections):
        end_page = sorted_subsections[start_index + 1][1]
    
    selected_pages.append((start_page, end_page))

expr = ""
for i in selected_pages:
    if expr=="":
        expr += "(page_numbers >= {} && page_numbers <= {})".format(i[0], i[1])
    else: 
        expr += " || (page_numbers >= {} && page_numbers <= {})".format(i[0], i[1])

print(expr)
# -*- coding: utf-8 -*-
import fitz  # PyMuPDF

def find_next_subsection_start(page_text, current_subsection_index, subsection_titles):
    """
    Attempts to find the start index of the next subsection within the page_text.
    Returns the index of the next subsection title in the page_text, or None if not found.
    """
    # Ensure the subsection titles are in the order they appear in the document
    if current_subsection_index + 1 < len(subsection_titles):
        next_subsection_title = subsection_titles[current_subsection_index + 1]
        # Try to find the next subsection title in the page_text
        start_index = page_text.find(next_subsection_title)
        if start_index != -1:
            return start_index
    return None


def extracting_subsections(pdf_path, subsections):
    doc = fitz.open(pdf_path)
    # Convert subsections dictionary to a list of titles to maintain order
    subsection_titles = list(subsections.keys())
    subsection_texts = {title: "" for title in subsection_titles}
    current_subsection_index = 0

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = page.get_text()
        
        while current_subsection_index < len(subsection_titles):
            # Check if this page contains the start of the current subsection
            if page_num + 1 >= subsections[subsection_titles[current_subsection_index]]:
                # Find the start of the next subsection to determine where to stop appending text
                next_start_index = find_next_subsection_start(page_text, current_subsection_index, subsection_titles)
                if next_start_index is not None:
                    # Append text up to the start of the next subsection
                    subsection_texts[subsection_titles[current_subsection_index]] += page_text[:next_start_index].strip()
                    page_text = page_text[next_start_index:]  # Prepare remaining text for the next loop iteration
                    current_subsection_index += 1  # Move to the next subsection
                else:
                    # If the next subsection isn't on this page, append all text and break to process the next page
                    subsection_texts[subsection_titles[current_subsection_index]] += page_text.strip()
                    break
            else:
                break
    return subsection_texts
        #print(subsection_texts)
def extract_subsections(pdf_path, subsections):
    doc = fitz.open(pdf_path)
    # Convert subsections dictionary to a list of titles to maintain order
    subsection_titles = list(subsections.keys())
    subsection_texts = {title: "" for title in subsection_titles}
    current_subsection_index = 0

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = page.get_text()

        while current_subsection_index < len(subsection_titles):
            # Check if this page contains the start of the current subsection
            if page_num + 1 >= subsections[subsection_titles[current_subsection_index]]:
                # Find the start of the next subsection to determine where to stop appending text
                next_start_index = find_next_subsection_start(page_text, current_subsection_index, subsection_titles)
                
                if next_start_index is not None:
                    # Append text up to the start of the next subsection
                    subsection_texts[subsection_titles[current_subsection_index]] += page_text[:next_start_index].strip()
                    page_text = page_text[next_start_index:]  # Prepare remaining text for the next loop iteration
                    current_subsection_index += 1  # Move to the next subsection
                else:
                    # If the next subsection isn't on this page, append all text and break to process the next page
                    subsection_texts[subsection_titles[current_subsection_index]] += page_text.strip()
                    break
            else:
                break

    return subsection_texts


# Example usage
pdf_path = r'C:\\Users\\Utki\\Desktop\\code\\project\\apug data\\MATH136-notes.pdf'
subsections = {'Vectors in R𝑛': 5, 'Algebraic and Geometric Representation of Vectors': 6, 'Operations on Vectors': 7, 'Vectors in C𝑛': 12, 'Dot Product in R𝑛': 13, 'Projection, Components and Perpendicular': 19, 'Standard Inner Product in C𝑛': 23, 'Fields': 27, 'The Cross Product in R3': 28, 'Linear Combinations and Span': 32, 'Lines in R2': 34, 'Lines in R𝑛': 36, 'The Vector Equation of a Plane in R𝑛': 40, 'Scalar Equation of a Plane in R3': 44, 'Introduction': 48, 'Systems of Linear Equations': 49, 'An Approach to Solving Systems of Linear Equations': 55, 'Solving Systems of Linear Equations Using Matrices': 61, 'The Gauss–Jordan Algorithm for Solving Systems of Linear Equations': 66, 'Rank and Nullity': 75, 'Homogeneous and Non-Homogeneous Systems, Nullspace': 78, 'Solving Systems of Linear Equations over C': 84, 'Matrix–Vector Multiplication': 84, 'Using a Matrix\u2013Vector Product to Express a System of\nLinear Equations': 88, 'Solution Sets to Systems of Linear Equations': 89, 'The Column and Row Spaces of a Matrix': 97, 'Matrix Equality and Multiplication': 99, 'Arithmetic Operations on Matrices': 103, 'Properties of Square Matrices': 107, 'Elementary Matrices': 109, 'Matrix Inverse': 112, 'The Function Determined by a Matrix': 121, 'Linear Transformations': 122, 'The Range of a Linear Transformation and \u201cOnto\u201d Linear Transformations': 127, "The Kernel of a Linear Transformation and \u201cOne-to-\nOne\u201d Linear Transformations": 130, 'Every Linear Transformation is Determined by a Matrix': 133, 'Special Linear Transformations: Projection, Perpendicular, Rotation and Reflection': 136, 'Composition of Linear Transformations': 143, 'The Definition of the Determinant': 147, 'Computing the Determinant in Practice: EROs': 153, 'The Determinant and Invertibility': 156, 'An Expression for 𝐴−1': 158, "Cramer’s Rule": 161, 'The Determinant and Geometry': 163, 'What is an Eigenpair?': 167, 'The Characteristic Polynomial and Finding Eigenvalues': 169, 'Properties of the Characteristic Polynomial': 172, 'Finding Eigenvectors': 178, 'Eigenspaces': 181, 'Diagonalization': 183, 'Subspaces': 190, 'Linear Dependence and the Notion of a Basis of a Subspace': 193, 'Detecting Linear Dependence and Independence': 197, 'Spanning Sets': 204, 'Basis': 208, 'Bases for Col(𝐴) and Null(𝐴)': 212, 'Dimension': 216, 'Coordinates': 219, 'Matrix Representation of a Linear Operator': 230, 'Diagonalizability of Linear Operators': 235, 'Diagonalizability of Matrices Revisited': 238, 'The Diagonalizability Test': 244, 'Definition of a Vector Space': 254, 'Span, Linear Independence and Basis': 258, 'Linear Operators': 264, 'Index': 270}
subsection_content_dict = extracting_subsections(pdf_path, subsections)
del subsection_content_dict['Index']

import json
with open("math136-subsection",'w') as fout:
    json.dump(list(subsection_content_dict.values()), fout)

print(subsection_content_dict['Linear Operators']) #['Operations on Vectors']
print(subsection_content_dict) #['Operations on Vectors']

'''metadata = [[1, 'Vectors in Rn', 5], [2, 'Introduction', 5], [2, 'Algebraic and Geometric Representation of Vectors', 6], [2, 'Operations on Vectors', 7], [2, 'Vectors in Cn', 12], [2, 'Dot Product in Rn', 13], [2, 'Projection, Components and Perpendicular', 19], [2, 'Standard Inner Product in Cn', 23], [2, 'Fields', 27], [2, 'The Cross Product in R3', 28], [1, 'Span, Lines and Planes', 32], [2, 'Linear Combinations and Span', 32], [2, 'Lines in R2', 34], [2, 'Lines in Rn', 36], [2, 'The Vector Equation of a Plane in Rn', 40], [2, 'Scalar Equation of a Plane in R3', 44], [1, 'Systems of Linear Equations', 48], [2, 'Introduction', 48], [2, 'Systems of Linear Equations', 49], [2, 'An Approach to Solving Systems of Linear Equations', 55], [2, 'Solving Systems of Linear Equations Using Matrices', 61], [2, 'The Gauss–Jordan Algorithm for Solving Systems of Linear Equations', 66], [2, 'Rank and Nullity', 75], [2, 'Homogeneous and Non-Homogeneous Systems, Nullspace', 78], [2, 'Solving Systems of Linear Equations over C', 84], [2, 'Matrix–Vector Multiplication', 84], [2, 'Using a Matrix–Vector Product to Express a System of Linear Equations', 88], [2, 'Solution Sets to Systems of Linear Equations', 89], [1, 'Matrices', 97], [2, 'The Column and Row Spaces of a Matrix', 97], [2, 'Matrix Equality and Multiplication', 99], [2, 'Arithmetic Operations on Matrices', 103], [2, 'Properties of Square Matrices', 107], [2, 'Elementary Matrices', 109], [2, 'Matrix Inverse', 112], [1, 'Linear Transformations', 121], [2, 'The Function Determined by a Matrix', 121], [2, 'Linear Transformations', 122], [2, 'The Range of a Linear Transformation and ``Onto" Linear Transformations', 127], [2, 
"The Kernel of a Linear Transformation and ``One-to-One'' Linear Transformations", 130], [2, 'Every Linear Transformation is Determined by a Matrix', 133], [2, 'Special Linear Transformations: Projection, Perpendicular, Rotation and Reflection', 136], [2, 'Composition of Linear Transformations', 143], [1, 'The Determinant', 147], [2, 'The Definition of the Determinant', 147], [2, 'Computing the Determinant in Practice: EROs', 153], [2, 'The Determinant and Invertibility', 156], [2, 'An Expression for the inverse of A', 158], [2, "Cramer's Rule", 161], [2, 'The Determinant and Geometry', 163], [1, 'Eigenvalues and Diagonalization', 167], [2, 'What is an Eigenpair?', 167], [2, 'The Characteristic Polynomial and Finding Eigenvalues', 169], [2, 'Properties of the Characteristic Polynomial', 172], [2, 'Finding Eigenvectors', 178], [2, 'Eigenspaces', 181], [2, 'Diagonalization', 183], [1, 'Subspaces and Bases', 190], [2, 'Subspaces', 190], [2, 'Linear Dependence and the Notion of a Basis of a Subspace', 193], [2, 'Detecting Linear Dependence and Independence', 
197], [2, 'Spanning Sets', 204], [2, 'Basis', 208], [2, 'Bases for col(A) and null(A)', 212], [2, 'Dimension', 216], [2, 'Coordinates', 219], [1, 'Diagonalization', 230], [2, 'Matrix Representation of a Linear Operator', 230], [2, 'Diagonalizability of Linear Operators', 235], [2, 'Diagonalizability of Matrices Revisited', 238], [2, 'The Diagonalizability Test', 244], [1, 'Vector Spaces', 254], [2, 'Definition of a Vector Space', 254], [2, 'Span, Linear Independence and Basis', 258], [2, 'Linear Operators', 264]]

subsection = {}
for i in metadata:
    if i[0] == 2:
        if i[1] == 'Introduction':
            print(i[2])
        subsection[i[1]] = i[2]
    if i[0] == 3:
        print(i)
print(subsection)
print(len(metadata), len(subsection))
'''
from transformers import RagTokenizer, RagTokenForGeneration
from openai import OpenAI
import fitz

client = OpenAI(api_key="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")

pdf_path = r'C:\\Users\\Utki\\Desktop\\code\\project\\apug data\\MATH136-notes.pdf'
doc = fitz.open(pdf_path)
toc = doc.get_toc()
doc.close()

#EX CONTENT UPTO CHAPTER 3
new_toc = []
chapter = 0
chapter_threshold = 3 
for i in toc:
    if i[0] == 1:
        chapter+=1
        if chapter <= chapter_threshold:
            pass
        #    new_toc.append(i)
        else:
            break
    else:
        new_toc.append(i[1])

'''#EX SUBSECTION 3.1 TO 3.5
new_toc = []
start_appending = 0
subsection_start = ""
subsection_threshold = ""
for i in toc:
    if i[0] == 2 or i[0]==3:
        if start_appending > 0:
            new_toc.append(i[1])
        if i[1]==subsection_start:
            start_appending+=1
            new_toc.append(i[1])
        if i[1]==subsection_threshold:
            new_toc.append(i[1])
            break'''

subsection = {}
for i in toc:
    if i[0]==2:
        subsection[i[1]]=i[2]
query = "Proveordisprovethefollowingstatements. (a)Asystemof3equationsin5variableshasinfinitelymanysolutions. (b)Asystemof5equationsin3variablescannothaveinfinitelymanysolutions. (c) If thesolutionsettoasystemofequations isaline, thenthecoefficientmatrixof the systemhasrankequal to1. NOTE: For 'prove or disprove' or 'true or false' questions please consider all trivial and edge cases."
#query = "Prove that if two planes in R3 intersect, then they either intersect in a line or a plane using the topics identified above"
sub_query = '''The question is as follows: {}.\n Please identify which of the following specific subsections are necessary for this proof to be solved using core concepts of linear algebra. It's important to use the exact names as listed:{}'''.format(query, str(new_toc))
#sub_query = '''Which of the following topics are required from {} to answer the question {}'''.format(toc, query)
extracted_content = ''''''

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  #model="gpt-4",
  messages=[
    {"role": "system", "content": "You are a mathematical professor at university of waterloo, skilled in explaining and solving linear algebra questions completely algebraically"},
    {"role": "user", "content": sub_query},
    #{"role": "assistant", "content": completion.choices[0].message["content"]},  # AI's response to the initial query
#    {"role": "user", "content": "Why did you not pick Rank and Nullity? How do i refine my query for you to look at all possible aspects and avoid such errors."},  # The follow-up query
#    {"role": "user", "content":"Here I have solved the question without rank and nullity, why did you overlook it: SincetheplanesareinR3,wecanrewritethemintheirnormal forms(that is,usingscalar equations): ax+by+cz=d a′x+b′y+c′z=d′. Theintersectpointsofthetwoplanesarethesimultaneoussolutionsoftheabovetwolinear equations. Thecoefficientmatrixof thissystemofequations isthusa2×3matrix. Then, byRankBounds,weseethattherankofthematrixisatmost2—inotherwords,therank of thematrixiseither1or2(*),whichcorrespondstoanullityof2(thesolutionset isa plane)or1(thesolutionset isaline)respectivelybytheSystemRankTheorem."}
  ],
  max_tokens=400, # Adjust based on how long you expect the answer to be
  temperature=0.1, # A higher temperature encourages creativity. Adjust based on your needs
  top_p=0.5,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None # You can specify a stop sequence if there's a clear endpoint. Otherwise, leave it as None
  #stream = True
)

chatgpt_subheaders = str(completion.choices[0].message)
replace = str("\\\( \\\mathbb{")
chatgpt_subheaders = chatgpt_subheaders.replace(replace, '')
chatgpt_subheaders = chatgpt_subheaders.replace('}^', '')

print(chatgpt_subheaders)
intro = 0
selected_subheaders = {}
for i in subsection:
    if i in chatgpt_subheaders:
        selected_subheaders[i] = subsection[i]

subsections = {'Vectors in R𝑛': 5, 'Algebraic and Geometric Representation of Vectors': 6, 'Operations on Vectors': 7, 'Vectors in C𝑛': 12, 'Dot Product in R𝑛': 13, 'Projection, Components and Perpendicular': 19, 'Standard Inner Product in C𝑛': 23, 'Fields': 27, 'The Cross Product in R3': 28, 'Linear Combinations and Span': 32, 'Lines in R2': 34, 'Lines in R𝑛': 36, 'The Vector Equation of a Plane in R𝑛': 40, 'Scalar Equation of a Plane in R3': 44, 'Introduction': 48, 'Systems of Linear Equations': 49, 'An Approach to Solving Systems of Linear Equations': 55, 'Solving Systems of Linear Equations Using Matrices': 61, 'The Gauss–Jordan Algorithm for Solving Systems of Linear Equations': 66, 'Rank and Nullity': 75, 'Homogeneous and Non-Homogeneous Systems, Nullspace': 78, 'Solving Systems of Linear Equations over C': 84, 'Matrix–Vector Multiplication': 84, 'Using a Matrix\u2013Vector Product to Express a System of\nLinear Equations': 88, 'Solution Sets to Systems of Linear Equations': 89, 'The Column and Row Spaces of a Matrix': 97, 'Matrix Equality and Multiplication': 99, 'Arithmetic Operations on Matrices': 103, 'Properties of Square Matrices': 107, 'Elementary Matrices': 109, 'Matrix Inverse': 112, 'The Function Determined by a Matrix': 121, 'Linear Transformations': 122, 'The Range of a Linear Transformation and \u201cOnto\u201d Linear Transformations': 127, "The Kernel of a Linear Transformation and \u201cOne-to-\nOne\u201d Linear Transformations": 130, 'Every Linear Transformation is Determined by a Matrix': 133, 'Special Linear Transformations: Projection, Perpendicular, Rotation and Reflection': 136, 'Composition of Linear Transformations': 143, 'The Definition of the Determinant': 147, 'Computing the Determinant in Practice: EROs': 153, 'The Determinant and Invertibility': 156, 'An Expression for 𝐴−1': 158, "Cramer’s Rule": 161, 'The Determinant and Geometry': 163, 'What is an Eigenpair?': 167, 'The Characteristic Polynomial and Finding Eigenvalues': 169, 'Properties of the Characteristic Polynomial': 172, 'Finding Eigenvectors': 178, 'Eigenspaces': 181, 'Diagonalization': 183, 'Subspaces': 190, 'Linear Dependence and the Notion of a Basis of a Subspace': 193, 'Detecting Linear Dependence and Independence': 197, 'Spanning Sets': 204, 'Basis': 208, 'Bases for Col(𝐴) and Null(𝐴)': 212, 'Dimension': 216, 'Coordinates': 219, 'Matrix Representation of a Linear Operator': 230, 'Diagonalizability of Linear Operators': 235, 'Diagonalizability of Matrices Revisited': 238, 'The Diagonalizability Test': 244, 'Definition of a Vector Space': 254, 'Span, Linear Independence and Basis': 258, 'Linear Operators': 264, 'Index': 270}
selected_subsections = []
for i in selected_subheaders:
    for j in subsections:
        if selected_subheaders[i] == subsections[j]:
            selected_subsections.append(j)

print(selected_subsections)
AI_response = (str(completion.choices[0].message)[30:])
final_response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  #model="gpt-4",
  messages=[
    {"role": "system", "content": "You are a mathematical professor at university of waterloo, skilled in explaining and solving linear algebra questions completely algebraically"},
    {"role": "user", "content": sub_query},
    {"role": "assistant", "content": AI_response},  # AI's response to the initial query
    {"role": "user", "content": query},
#    {"role": "user", "content": "Why did you not pick Rank and Nullity? How do i refine my query for you to look at all possible aspects and avoid such errors."},  # The follow-up query
#    {"role": "user", "content":"Here I have solved the question without rank and nullity, why did you overlook it: SincetheplanesareinR3,wecanrewritethemintheirnormal forms(that is,usingscalar equations): ax+by+cz=d a′x+b′y+c′z=d′. Theintersectpointsofthetwoplanesarethesimultaneoussolutionsoftheabovetwolinear equations. Thecoefficientmatrixof thissystemofequations isthusa2×3matrix. Then, byRankBounds,weseethattherankofthematrixisatmost2—inotherwords,therank of thematrixiseither1or2(*),whichcorrespondstoanullityof2(thesolutionset isa plane)or1(thesolutionset isaline)respectivelybytheSystemRankTheorem."}
  ],
  max_tokens=600, # Adjust based on how long you expect the answer to be
  temperature=0.1, # A higher temperature encourages creativity. Adjust based on your needs
  top_p=0.5,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None # You can specify a stop sequence if there's a clear endpoint. Otherwise, leave it as None
  #stream = True
)

print(final_response)
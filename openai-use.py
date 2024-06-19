import latex2mathml
import fitz
import latex2mathml.converter
from openai import OpenAI

pdf_path = r'C:\\Users\\Utki\\Desktop\\code\\project\\apug data\\MATH136-notes.pdf'

def extract_text_by_chapter(pdf_path):
    doc = fitz.open(pdf_path)
    chapters = {"Pre-Chapter": ''}  # Dictionary to store chapter texts
    chapter_title = "Pre-Chapter"  # Default chapter title for content before the first chapter

    for page_num in range(len(doc)):
        page_text = doc.load_page(page_num).get_text()
        
        # Attempt to identify chapter starts and titles
        if "Chapter" in page_text.split('\n')[0]:  # Assuming chapters start at the beginning of pages
            chapter_title = page_text.split('\n')[0]  # Simplified assumption
            chapters[chapter_title] = ""  # Initialize chapter text
        
        chapters[chapter_title] += page_text  # Append text to current chapter

    return chapters

chapters = extract_text_by_chapter(pdf_path)
del chapters['Pre-Chapter']
content = chapters['Chapter 1']
print(content)

section = 'Chapter 1'
course = 'MATH136'
client = OpenAI(api_key="")
sub_query = '''You are provided with the {} from the {} textbook for university of waterloo. 
I want you to provide a concise summary for it that includes the following details:Content for Summaries
1. Introduction to Key Concepts
Brief Overview: Start with a concise introduction that captures the main themes and objectives of the chapter or subsection.
Example: "This chapter introduces the fundamental concepts of linear algebra, focusing on vector spaces, linear transformations, and matrices."
2. Important Definitions and Formulas
Key Definitions: Summarize crucial definitions that are essential to understanding the chapter’s content.
Example: "Key definitions include vector addition, scalar multiplication, and the concept of a matrix inverse."
3. Core Principles or Theorems
Main Theorems or Concepts: Highlight the main theorems, proofs, or conceptual frameworks discussed.
Example: "Central to this chapter is the theorem on matrix invertibility, which states conditions under which a matrix can be inverted."
4. Chapter Summary and Conclusion
Recap: Conclude with a summary that encapsulates the key takeaways and their implications for future or previous chapters in the textbook.

Follow these tips while making the summaries:
For complex chapters involving multiple concepts and formulas, provide a paragraph-long summary.
For more straightforward subsections, limit the summary to a few impactful sentences.
Incorporate key terms and concepts within the summaries to enhance the ability to connect queries with relevant sections.'''.format(section, course)


topics_steps = client.chat.completions.create(
  model="gpt-3.5-turbo-1106",
  messages=[
    {"role": "system", "content": sub_query},
    {"role": "user", "content": content},
  ],
  max_tokens=15000, # Adjust based on how long you expect the answer to be
  temperature=0, # A higher temperature encourages creativity. Adjust based on your needs
  top_p=0,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None # You can specify a stop sequence if there's a clear endpoint. Otherwise, leave it as None
)

output = str(topics_steps.choices[0].message)
print(output, "\n\n")
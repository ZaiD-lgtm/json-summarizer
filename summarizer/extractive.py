from sentence_transformers import SentenceTransformer
import nltk
def extractive(text):
    model = SentenceTransformer('all-mpnet-base-v2')
    nltk.download('punkt')

    sentences = nltk.sent_tokenize(text)
    op_str = " "
    for i in sentences:
        # print(i)
        op_str += i

    return op_str


text= """Successfully navigated to the doctor directory page. Now need to extract information about all cardiologists available on healthbuddy.softnerve.com.

CARDIOLOGISTS LISTED ON THE PAGE

1. Dr. Sanjeev Bhatt
   • Qualification: MBBS
   • Experience: 15 Years
   • Consultations Done: 98
   • Available Time: 10:00 AM – 7:00 PM
   • Location: Raidhara, Pithoragarh, Uttarakhand

2. Dr. Sanjeev Bhatt
   • Qualification: MBBS
   • Experience: 1 Year
   • Consultations Done: 98
   • Available Time: 10:00 AM – 7:00 PM
   • Location: Raidhara, Pithoragarh, Uttarakhand

3. Dr. Aniket
   • Qualification: MBBS
   • Experience: 20 Years
   • Consultations Done: 98
   • Available Time: 10:00 AM – 7:00 PM
   • Location: Ausa Road, Latur, Maharashtra"
"""

output = extractive(text)
print(output)

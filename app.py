import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
import nltk.stem.porter import porterstemmer
ps = porterstemmer()

def transform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)

  y = []
  for i in text :
    if i.isalnum():
      y.append(i)
  text = y[:]
  y.clear()
  for i in text:
    if i not in stopwords.words('english')and i not in string.punctuation:
      y.append(i)
  text = y[:]
  y.clear()
  for i in text:
    y.append(ps.stem(i))
  return " ".join(y)


tkidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title('email/sms spam Classifier')

input_sms = st.text_input("enter the message")



# 1. preprocess
sms = transform_text(input_sms)
# 2. vectorize
vector_input = tkidf.transform([sms])
# 3. predict
result = model.predict(vector_input)[0]
# 4. Display
if result == 1:
  st.header('spam')
else:
  st.header('not spam')

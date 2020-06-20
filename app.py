import streamlit as st
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import sklearn
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob


import re
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")

stop_words = stopwords.words("english")

#pickle_in = open("NLP.pkl","rb")
#classifier = pickle.load(pickle_in)

def remove_punc(text):
    remove = re.sub(r"[^\w\s]","",text)
    return remove

def tokenize(text):
    nltk.download("punkt")
    nltk.download("wordnet")
    nltk.download("stopwords")
    r = remove_punc(text)
    token = nltk.word_tokenize(r)
    return token

def lemmatize(text):
    nltk.download("punkt")
    nltk.download("wordnet")
    nltk.download("stopwords")
    r = remove_punc(text)
    lemmatizer = WordNetLemmatizer()
    token = tokenize(r)
    lema = []
    for t in token:
        k = lemmatizer.lemmatize(t)
        lema.append(k)
    return lema

def stemmin(text):
    port_stem = PorterStemmer()
    r = remove_punc(text)
    tokens = tokenize(r)
    stem_sentence = []
    for word in tokens:
      stem_sentence.append(port_stem.stem(word))
    return stem_sentence
#def preprocess_sms(text):
 #   lemmatizer = WordNetLemmatizer()
#
 #   sentence = remove_punc(text)
#
 #   words = nltk.word_tokenize(sentence)

  #  words = [w for w in words if not w in stop_words]

   # filter_Sentence = ""
    #for word in words:
     #   filter_Sentence = filter_Sentence + ' ' + str(lemmatizer.lemmatize(word)).lower()
    #return filter_Sentence

#def word_vectorizer(text):
 #   data = [text]
  #  count_vec = CountVectorizer().fit(data)
   # freq_term_matrix = count_vec.transform(data)
    #tfidf = TfidfTransformer(norm = 'l2').fit(freq_term_matrix)
    #tf_idf_matrix = tfidf.fit_transform(freq_term_matrix)
    #return tf_idf_matrix

#def prediction(text):
 #   t = preprocess_sms(text)
  #  x_vec = word_vectorizer(t)
   # pred = classifier.predict(x_vec)
    #return pred



def main():
    st.title("Model to detect the sentiment of the text!")
    st.subheader("NLP TASK")

    SMS = st.text_input("ENTER THE SMS", "")

    if st.checkbox("Remove Punctuations!"):
        st.subheader("Removing Punctuations")
        if st.button("Remove"):
            rem = remove_punc(SMS)
            st.success(rem)
    if st.checkbox("Show Tokens"):
        st.subheader("Tokenized Text")
        if st.button("TOKENIZE"):
            tok_word = tokenize(SMS)
            st.success(tok_word)
    if st.checkbox("Show lemmatized words"):
        st.subheader("Lemmatized words")
        if st.button("LEMMATIZE"):
            lemma_word = lemmatize(SMS)
            st.success(lemma_word)
    if st.checkbox("Show Stemmed Words"):
        st.subheader("Stemmed words")
        if st.button("STEM"):
            stem_word = stemmin(SMS)
            st.success(stem_word)
    if st.button("PREDICT"):
        blob = TextBlob(SMS)
        result = blob.sentiment
        st.success(result)
        #if result == 1:
        #    st.success("IT'S SPAM")
        #if result == 0:
         #   st.success("NOT SPAM")
    if st.button("WHAT IS THIS"):
        st.text("MY SECOND DEPLOYMENT")
        st.text("HAPPY LEARNING!!!!")







if __name__ == '__main__':
    main()
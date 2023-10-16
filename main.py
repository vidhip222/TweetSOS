import pandas as pd
import numpy as np
import re
import urllib.request
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

# function to load data from a URL
def fetch_data():
    url = 'https://raw.githubusercontent.com/glrn/nlp-disaster-analysis/master/dataset/socialmedia-disaster-tweets-DFE.csv'
    csv = urllib.request.urlopen(url).read()
    with open('./disasters_social_media.csv', 'wb') as fx:
        fx.write(csv)
    df_raw = pd.read_csv('./disasters_social_media.csv', encoding='latin-1')
    df = df_raw[df_raw.choose_one != "Can't Decide"]
    df = df[['text', 'choose_one']].rename(columns={'choose_one': 'relevance'})
    df['relevance'] = df['relevance'].map({'Relevant': 1, 'Not Relevant': 0})
    return df

# function for text preprocessing
custom_ignore_words = ['this', 'that', 'with', 'some', 'these', 'those', 'there', 'a', 'the', 'if', 'br', 'and', 'of', 'to', 'is', 'are', 'he', 'she', 'my', 'you', 'it', 'how']

def preprocess_text(sentence):
    words = re.sub("[^\w]", " ", sentence).split()
    words_cleaned = [word.lower() for word in words if word.lower() not in custom_ignore_words]
    return ' '.join(words_cleaned)



# function for predicting disaster tweets
def disaster_tweet_predictor(tweet, logreg, vectorizer):
    try:
        cleaned_tweet = preprocess_text(tweet)
        word_tfidf = vectorizer.transform([cleaned_tweet])
        prediction = logreg.predict(word_tfidf)[0]
        results = {1: 'Relevant', 0: 'Not Relevant'}
        return results[prediction]
    except Exception as e:
        return f"Error: {e}"

# Main function
def main():
    df = fetch_data()
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english', lowercase=True)
    tfidf = vectorizer.fit_transform(df['text'])
    
    X_train, X_test, y_train, y_test = train_test_split(tfidf, df['relevance'], shuffle=True)
    
    parameters = {'C': [0.001, 0.01, 0.1, 1, 10], 'tol': [0.0001, 0.001, 0.01], 'max_iter': [300, 1000]}
    logreg = LogisticRegression(solver='lbfgs', max_iter=200)
    clf = GridSearchCV(logreg, parameters, cv=3, return_train_score=True)
    clf.fit(X_train, y_train)

    st.markdown("# TweetSOS 🆘")
    st.markdown("## Disaster Tweet Predictor 🌀")
    
    st.markdown("### **Empowering Disaster Response for Sustainable Development**")
    st.markdown(
        "TweetSOS is a powerful web application designed to address Goal #3 of the United Nations' 17 Sustainable Development Goals (SDGs). "
        "By harnessing the real-time power of social media, TweetSOS aims to provide invaluable support to emergency responders and aid organizations during natural disasters, "
        "enabling them to take prompt action and safeguard lives in times of crisis."
    )

    st.markdown("### **Ensuring Healthy Lives and Well-Being for All**")
    st.markdown(
        "Through state-of-the-art machine learning algorithms, TweetSOS analyzes tweets from around the world to identify those directly related to natural disasters. "
        "By accurately predicting the relevance of each tweet, this web application aids in rapid disaster response, "
        "ensuring that the right resources and assistance are channeled to affected areas swiftly."
    )

    st.markdown("### **Key Features:**")
    st.markdown("- Real-time Disaster Tweet Analysis")
    st.markdown("- Predicts Relevance to Disaster Events")
    st.markdown("- Assists Emergency Responders and Aid Organizations")
    st.markdown("- Enhances Decision-Making for Disaster Relief Efforts")

    st.markdown("### **Be Part of the Solution:**")
    st.markdown(
        "You can actively contribute to disaster response and sustainable development efforts by using TweetSOS. "
        "Simply enter a tweet, and our advanced algorithms will determine its relevance to natural disasters, "
        "empowering you with information to support emergency response and relief initiatives."
    )

    st.markdown("**Join us in building a safer, resilient, and sustainable world for all!**")

    st.header("Enter the Tweet 👇")
    tweet = st.text_input('')
    st.write('Tweet Entered:', tweet)
    
    if tweet:
        result = disaster_tweet_predictor(tweet, clf, vectorizer)
        if result == "Relevant": 
            st.subheader("Prediction: there is a disaster :heavy_check_mark:")
            st.write("In case of a disaster or emergency, it is essential to contact the relevant authorities or organizations for prompt and appropriate disaster response. The contact information may vary based on the location and the type of disaster. Here are some general guidelines on who to contact in different situations:")
            st.write("1. Local Emergency Services: In most cases, the first line of contact for immediate assistance during a disaster is local emergency services, such as police, fire departments, and medical services. In the event of an emergency, dial the emergency number for your country (e.g., 911 in the United States, 999 in the United Kingdom, 000 in Australia).")
            st.write("2. National Emergency Response Agencies: Many countries have specific national agencies or organizations responsible for disaster response and emergency management. These agencies often have dedicated hotlines and contact information for reporting disasters and seeking assistance.")
            st.write("3. International Humanitarian Organizations: For large-scale disasters or crises, international humanitarian organizations like the United Nations Office for the Coordination of Humanitarian Affairs (OCHA), Red Cross, and Médecins Sans Frontières (Doctors Without Borders) may be involved in disaster response efforts. You can find their contact information on their official websites.")
            st.write("4. Local Government and Civil Authorities: Local government and civil authorities may have disaster response departments or offices that you can contact for information and assistance during disasters.")
            st.write("5. Online Disaster Reporting Platforms: Some countries and organizations have online platforms or apps where people can report disasters and emergencies. Check if such platforms exist in your region and use them to provide information about the situation.")
        else:
            st.subheader("Prediction: there are no disasters :x:")
        
    st.caption('Made by Vidhi')


if __name__ == '__main__':
    import streamlit as st
    main()
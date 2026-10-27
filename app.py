import streamlit as st
import joblib

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("SMS Spam Detection")
st.write("Enter a message to check if it is spam.")

message = st.text_area("Write your message here:")

if st.button("Check Message"):
    if message.strip():
        message_vector = vectorizer.transform([message])
        prediction = model.predict(message_vector)[0]

        if prediction == "spam":
            st.error("This message is SPAM!")
        else:
            st.success("This message is NOT SPAM!")
    else:
        st.warning("Please enter a message first.")


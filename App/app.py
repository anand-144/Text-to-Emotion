import streamlit as st
import pandas as pd
from datetime import datetime
import sqlite3
import threading

# Core Packages
import altair as alt
import joblib

# Exp Packages
import pandas as pd
import numpy as np

# Utils
import joblib
pipe_lr = joblib.load(
    open("Model/emotion_classifier_pipe_lr_01_june_2023.pkl", "rb"))

# Create a thread-local database connection
local_db = threading.local()
local_db.conn = sqlite3.connect('data.db')


# Function to create the pageTrackTable
def create_page_visited_table():
    c = local_db.conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS pageTrackTable(pagename TEXT, timeOfvisit TIMESTAMP)')


# Function to add page visited details to pageTrackTable
def add_page_visited_details(pagename, timeOfvisit):
    c = local_db.conn.cursor()
    c.execute('INSERT INTO pageTrackTable(pagename, timeOfvisit) VALUES (?, ?)', (pagename, timeOfvisit))
    local_db.conn.commit()


# Function to retrieve all page visited details
def view_all_page_visited_details():
    c = local_db.conn.cursor()
    c.execute('SELECT * FROM pageTrackTable')
    data = c.fetchall()
    return data


# Fxn
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]


def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results


emotions_emoji_dict = {
    "anger": "😠",
    "disgust": "🤮",
    "fear": "😨😱",
    "happy": "🤗",
    "joy": "😂",
    "neutral": "😐",
    "sad": "😔",
    "sadness": "😔",
    "shame": "😳",
    "surprise": "😮",
}


# Main Application
def main():
    st.title("Emotion Classifier App")
    menu = ["Home", "Monitor", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    create_page_visited_table()
    if choice == "Home":
        add_page_visited_details("Home", datetime.now().timestamp())
        st.subheader("Home-Emotion In Text")

        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1, col2 = st.columns(2)

            # Apply Fxn Here
            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

            add_page_visited_details(raw_text, datetime.now().timestamp())

            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write(f"{prediction}: {emoji_icon}")
                st.write("Confidence: {0:.2%}".format(np.max(probability)))

            with col2:
                st.success("Prediction Probability")
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions", "probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(
                    x='emotions', y='probability', color='emotions'
                )
                st.altair_chart(fig, use_container_width=True)

    elif choice == "Monitor":
        st.subheader("Monitor App")

        with st.expander("Page Metrics"):
            page_visited_details = pd.DataFrame(view_all_page_visited_details(), columns=['Pagename', 'Time_of_Visit'])
            st.dataframe(page_visited_details)

            pg_count = page_visited_details['Pagename'].value_counts().rename_axis('Pagename').reset_index(
                name='Counts'
            )
            c = alt.Chart(pg_count).mark_bar().encode(
                x='Pagename', y='Counts', color='Pagename'
            )
            st.altair_chart(c, use_container_width=True)

    else:
        st.subheader("About")


if __name__ == '__main__':
    main()

import streamlit as st
import Youtube_assistant_helper as Yah
import textwrap

st.title("Youtube Assistant")

with st.sidebar:
    with st.form(key='my_form'):
        youtube_url=st.sidebar.text_area(
            label="What is the Youtube video URL?",
            max_chars=50
        )

        query=st.sidebar.text_area(
            label="Ask me about the video",
            max_chars=50,
            key="query"
        )
        
        submit_button=st.form_submit_button(label='Submit')

if query and youtube_url:
    db= Yah.create_vector_db_from_youtube_url(youtube_url)
    response= Yah.get_response_from_query(db, query,k=25)
    st.subheader("Answer:")
    st.text(textwrap.fill(response,width=80))
import streamlit as st
import Questions_and_Tests_generator_helper as QTGH
import textwrap

st.title("Test Generator")

with st.sidebar:
    with st.form(key='my_form'):
        level=st.sidebar.selectbox("How difficult?",("Beginner","Intermediate","Expert"))
        domain=st.text_input("Field of knowledge")
        number_of_questions= st.text_input("How many questions?")       
        submit_button=st.form_submit_button(label='Submit')

if level and domain and number_of_questions:
    dico= QTGH.generate_dico_QA(number_of_questions, domain, level)
    st.subheader(f"Test of {domain}")

    def Running_Test(dico_QA=QTGH.default_QA):
        nbr_of_Question= len(dico_QA)
        list_of_questions, list_of_answers= QTGH.obtain_listes(dico_QA)
        score=0
        for i in range (0, nbr_of_Question):
            answer= st.text_input(list_of_questions[i])
            if answer.lower() == list_of_answers[i].lower():
                st.write("Correct!")
                score += 1
            else:
                st.write("Incorrect!")
        percent= (score/nbr_of_Question)*100
        st.write(f"You have {score}/{nbr_of_Question} ({percent}%) of correct answers!!!")

    if __name__ == "__main__":
        Running_Test(dico)
import Pet_name_helper as pnh
import streamlit as st

st.title("Générateur de Noms pour animaux")
user_animal_type= st.sidebar.selectbox("Pour quel type d'animal?",("Chat","Chien","Hamster","Serpent","Vache"))

if user_animal_type:
    user_couleur= st.sidebar.text_area(label=f"Quelle est la(les) couleur(s) de votre {user_animal_type.lower()}?",max_chars=35)

if user_couleur:
    response= pnh.generate_pet_name(user_animal_type,user_couleur)
    st.text(response["pet_name"])
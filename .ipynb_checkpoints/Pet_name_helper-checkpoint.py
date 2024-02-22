from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

def generate_pet_name(animal_type, couleur):
    llm= HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", model_kwargs={"temperature":0.5})
    
    prompt_template_name= PromptTemplate(
        input_variables=['animal_type', 'couleur'],template=" J'ai {animal_type} qui est de couleur {couleur} et je veux lui donner un nom cool. Donne moi une liste de 5 noms cools pour mon animal.")
    name_chain= LLMChain(llm=llm, prompt=prompt_template_name, output_key="pet_name")
    response = name_chain.invoke({'animal_type': animal_type, 'couleur': couleur})
    return response

if __name__ == "__main__":
    print(generate_pet_name("un chien", "marron"))
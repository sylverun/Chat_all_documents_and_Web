from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import ast

load_dotenv()


dico_QA={}

default_QA={
    "What is the capital of France?": "Paris",
    "Who wrote 'Romeo and Juliet'?": "William Shakespeare",
    "What is the tallest mountain in the world?": "Mount Everest",
    "What is the chemical symbol for water?": "H2O",
    "Who painted the Mona Lisa?": "Leonardo da Vinci",
    "What is the currency of Japan?": "Yen",
    "What is the powerhouse of the cell?": "Mitochondria",
    "What is the boiling point of water in Celsius?": "100",
    "Who is the first president of the United States?": "George Washington",
    "What is the largest ocean on Earth?": "Pacific Ocean"
}

def obtain_listes(dictionnaire):
    cles = []
    valeurs = []
    for cle, valeur in dictionnaire.items():
        cles.append(cle)
        valeurs.append(valeur)    
    return cles, valeurs


def generate_dico_QA(number_of_questions, domain, level):
    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    llm= HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0.3})
    
    prompt_template_name= PromptTemplate(
        input_variables=['number_of_questions','domain','level'],
        template=" You are a helpful assistant who can develop questions to test knowledge. Generate {number_of_questions} questions relating to the domain of {domain} and for a {level} level.The answers don't exceed three word.You will format the questions as the keys and the answers as the values of a python dictionary.")
    name_chain= LLMChain(llm=llm, prompt=prompt_template_name)
    response = name_chain.run({'number_of_questions': number_of_questions, 'domain':domain, 'level': level})
    index_debut = response.find("{")
    index_fin = response.rfind("}") + 1
    dictionnaire_str = response[index_debut:index_fin]
    try:
        dictionnaire = ast.literal_eval(dictionnaire_str)
    except (SyntaxError, ValueError) as e:
        print("Erreur lors de l'extraction du dictionnaire :", e)
    return dictionnaire
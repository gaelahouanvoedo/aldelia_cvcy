import streamlit as st
from PIL import Image
import PyPDF2
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

st.set_page_config(
    page_title="cvCy - ♥ IA of Gael Ahouanvoedo",
    page_icon="🤥",
    initial_sidebar_state="expanded",
)


def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def search_candidates(competences, df):
    df_select = pd.DataFrame(columns=df.columns)

    for index, row in df.iterrows():
        cv = row['contenu_textuel']
        for competence in competences:
            if competence.lower() in cv.lower():
                df_select = pd.concat([df_select, row.to_frame().transpose()], ignore_index=True)
                break

    df_select['skills'] = df_select['contenu_textuel'].apply(lambda x: [comp for comp in competences if comp.lower() in x.lower()])
    df_select.drop(['contenu_textuel'], axis=1, inplace=True)

    vectorizer = CountVectorizer()
    skills_matrix = vectorizer.fit_transform(df_select['skills'].apply(lambda x: ', '.join(x)))
    similarity_scores = cosine_similarity(skills_matrix, vectorizer.transform([', '.join(competences)]))
    df_select['similarite'] = similarity_scores.flatten()
    df_select = df_select.sort_values('similarite', ascending=False)

    return df_select


with st.sidebar:
    image = Image.open('log.png')
    st.image(image, width=180)
    st.success("Lancez l'application ici 👇")
    menu = st.sidebar.selectbox("Menu", ('Introduction', "Charger", "Rechercher"))
    st.subheader("Informations")
    st.write("Cette application permet de rechercher des mots clés dans une base de CVs", unsafe_allow_html=True)
    '***'
    '**Build with ♥ by Gael Ahouanvoedo**'


if menu == "Introduction":
    st.write("""
    # Sélection de CV.
    
    Cette application permet de sélectionner le CV qui répond le mieux à une liste de mots clés. 
                   
    """)

    st.write("""
    **👈 Pour démarrer, sélectionnez "Charger" dans la barre latérale.**             
    """)

    st.write("""
    ### Credit
    Gael Ahouanvoedo, gael.ahouanvoedo@aldelia.com
    """)

    st.write("""
    ### Website
    https://www.aldelia.com/en/        
    """)

    st.write("""
    ### Avertissement
    Il s'agit d'une micro application web créé pour un besoin spécifique. Il peut ne pas répondre à vos attentes dans tous vos contexte. Veuilez donc ne pas entièrement vous fier aux résultas issues de son exploitation.
    """)

if menu == "Charger":
    st.title("Chargez un CV.")

    cv = st.file_uploader("Chargez un ou plusieurs CV au format PDF", type=["pdf"], accept_multiple_files=True)

    if st.button("Soumettre"):
        if cv:
            dfs = []
            for file in cv:
                if file.type == "application/pdf":
                    cv_text = extract_text_from_pdf(file)
                    dfs.append(pd.DataFrame({'nom_fichier': [file.name], 'skills': [cv_text]}))
            if dfs:
                df = pd.concat(dfs, ignore_index=True)
                df.to_csv('cv_data.csv', index=False)
                st.success(f"{len(dfs)} CVs soumis avec succès !")
            else:
                st.warning("Aucun fichier PDF valide trouvé. Veuillez télécharger des fichiers PDF.")
        else:
            st.warning("Veuillez charger au moins un CV.")

        
    
if menu == "Rechercher":

    st.title("Trouvez le meilleur candidat.")

    user_input = st.text_input("Saisir des compétences séparées par des virgules : ")
    competences = user_input.split(',')

    if st.button("Rechercher"):
        if len(competences) > 0:
            # Charger le fichier CSV contenant les CV
            df = pd.read_csv('cv_data.csv')
            df_select = search_candidates(competences, df)
            st.write(df_select)
        else:
            st.warning("Veuillez saisir au moins une compétence.")

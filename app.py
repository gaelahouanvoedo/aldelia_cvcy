import streamlit as st
from PIL import Image
import PyPDF2
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="cvCy - ♥ IA de Gaël Ahouanvoedo",
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
        cv = row['skills']
        for competence in competences:
            if competence.lower() in cv.lower():
                df_select = pd.concat([df_select, row.to_frame().transpose()], ignore_index=True)
                break

    df_select['skills'] = df_select['skills'].apply(lambda x: [comp for comp in competences if comp.lower() in x.lower()])
    df_select = df_select[df_select['skills'].apply(lambda x: len(x) > 0)]  # Filter out rows with empty skills

    if len(competences) > 0 and len(df_select) > 0:  # Check if competences and df_select are not empty
        vectorizer = CountVectorizer()
        skills_matrix = vectorizer.fit_transform(df_select['skills'].apply(lambda x: ', '.join(x)))
        similarity_scores = cosine_similarity(skills_matrix, vectorizer.transform([', '.join(competences)]))
        df_select['similarite'] = similarity_scores.flatten()
        df_select = df_select.sort_values('similarite', ascending=False)

    return df_select

df = pd.DataFrame(columns=['nom_fichier', 'skills'])

with st.sidebar:
    image = Image.open('log.png')
    st.image(image, width=180)
    st.success("Lancez l'application ici 👇")
    menu = st.sidebar.selectbox("Menu", ('Introduction', "Lancer l'app"))
    st.subheader("Informations")
    st.write("Cette application permet de rechercher des mots-clés dans une base de CVs.", unsafe_allow_html=True)
    '***'
    '**Conçu avec ♥ par Gaël Ahouanvoedo**'

if menu == "Introduction":
    st.write("""
    # Sélection de CV.
    
    Cette application permet de sélectionner le CV qui répond le mieux à une liste de mots-clés. 
                   
    """)

    st.write("""
    **👈 Pour démarrer, sélectionnez "Lancer l'app" dans la barre latérale.**             
    """)

    st.write("""
    ### Crédits
    Gaël Ahouanvoedo, gael.ahouanvoedo@aldelia.com
    """)

    st.write("""
    ### Site Web
    https://www.aldelia.com/en/        
    """)

    st.write("""
    ### Avertissement
    Il s'agit d'une micro-application web créée pour un besoin spécifique. Elle peut ne pas répondre à vos attentes dans tous vos contextes. Veuillez donc ne pas vous fier entièrement aux résultats issus de son exploitation.
    """)

if menu == "Lancer l'app":
    st.title("1 - Chargez les CVs.")

    cv = st.file_uploader("Chargez un ou plusieurs CV au format PDF", type=["pdf"], accept_multiple_files=True)

    st.title("2 - Recherchez les mots-clés.")

    user_input = st.text_input("2 - Saisissez les mots-clés recherchés séparés par des virgules (ex: data, business, banque) : ")
    competences = user_input.split(',')

    if st.button("Soumettre"):
        if len(competences) > 0 and not df.empty:  # Check if competences and df are not empty
            df_select = search_candidates(competences, df)
            st.write(df_select)

            # Filtrer les CVs avec une similarité supérieure à 0.5 et 0.7
            df_min = df_select[df_select['similarite'] > 0]
            df_top = df_select[df_select['similarite'] > 0.5]

            # Afficher une alerte avec le nombre de CVs correspondant à chaque similarité
            if len(df_min) > 0:
                st.info(f"Il y a {len(df_min)} CVs qui correspondent à au moins un mot clé.")
            if len(df_top) > 0:
                st.success(f"Il y a {len(df_top)} CVs qui correspondent à plus de la moitié des mots clés.")
                st.markdown("**Les CVs qui correspondent le mieux :**")
                rank = 1
                for idx, row in df_top.iterrows():
                    expander = st.expander(f"{rank} - {row['nom_fichier']} - Cliquez pour voir le CV")
                    with expander:
                        cv_row = df[df['nom_fichier'] == row['nom_fichier']].iloc[0]
                        st.write(cv_row['skills'])
                    rank += 1
        else:
            st.warning("Veuillez charger des CVs et saisir des mots-clés pour effectuer une recherche.")

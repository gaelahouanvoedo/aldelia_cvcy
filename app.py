import streamlit as st
from PIL import Image
import PyPDF2
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="cvCy - ♥ AI by Gaël Ahouanvoedo",
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

df = pd.DataFrame(columns=['file_name', 'skills'])

with st.sidebar:
    image = Image.open('C:/Users\GaelAHOUANVOEDO\DATAWEB\cvCy\cvCy\log.png')
    st.image(image, width=180)
    st.success("Launch the application here 👇")
    menu = st.sidebar.selectbox("Menu", ('Introduction', "Launch the app"))
    st.subheader("Information")
    st.write("This application allows you to search for keywords in a database of CVs.", unsafe_allow_html=True)
    '***'
    '**Designed with ♥ by Gaël Ahouanvoedo**'

if menu == "Introduction":
    st.write("""
    # CV Selection.
    
    This application allows you to select the CV that best matches a list of keywords. 
                   
    """)

    st.write("""
    **👈 To get started, select "Launch the app" from the sidebar.**             
    """)

    st.write("""
    ### Credits
    Gaël Ahouanvoedo, gael.ahouanvoedo@aldelia.com
    """)

    st.write("""
    ### Website
    https://www.aldelia.com/en/        
    """)

    st.write("""
    ### Disclaimer
    This is a web micro-application created for a specific need. It may not meet your expectations in all your contexts. Therefore, please do not rely entirely on the results obtained from its use.
    """)

if menu == "Launch the app":
    st.title("1 - Upload CVs.")

    cv = st.file_uploader("Upload one or more CVs in PDF format", type=["pdf"], accept_multiple_files=True)

    st.title("2 - Search for keywords.")

    user_input = st.text_input("Enter the keywords you are looking for, separated by commas (e.g., data, business, banking): ")
    competences = user_input.split(',')

    if st.button("Submit"):
        if cv:
            dfs = []
            for file in cv:
                if file.type == "application/pdf":
                    cv_text = extract_text_from_pdf(file)
                    dfs.append(pd.DataFrame({'file_name': [file.name], 'skills': [cv_text]}))
            if dfs:
                df = pd.concat(dfs, ignore_index=True)
                st.success(f"{len(dfs)} CVs processed successfully!")
            else:
                st.warning("No valid CVs found. Please upload PDF files.")
        else:
            st.warning("Please upload at least one CV.")

        if len(competences) > 0 and not df.empty:  # Check if competences and df are not empty
            df_select = search_candidates(competences, df)
            
            # Filter out CVs with a similarity score above 0.5 and 0.7
            df_top = df_select[df_select['similarite'] > 0]

            # Display an alert with the number of CVs corresponding to each similarity score
            if len(df_top) > 0:
                st.success(f"There are {len(df_top)} CVs that match at least one keyword.")
                st.write(df_select)
                st.markdown("**Top matching CVs:**")
                rank = 1
                for idx, row in df_top.iterrows():
                    expander = st.expander(f"{rank} - {row['file_name']} - Click to view CV")
                    with expander:
                        cv_row = df[df['file_name'] == row['file_name']].iloc[0]
                        st.write(cv_row['skills'])
                    rank += 1

import streamlit as st
import pandas as pd

st.title("Product recommendation")

# Create a file uploader widget
# Create a file uploader widget
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read and display the uploaded file
    file_type = uploaded_file.type
    st.write(f"File type: {file_type}")

    if file_type == 'application/vnd.ms-excel':
        data = pd.read_excel(uploaded_file)
    else:
        data = pd.read_csv(uploaded_file)

    st.write('Displaying the uploaded data:')
    st.dataframe(data)
    import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the shopping trends dataset
#df = pd.read_csv('shopping_trends.csv')

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the 'Category' column
tfidf_matrix = vectorizer.fit_transform(data['Category'].values.astype('U'))

def recommend_product(input_text):
    if not input_text:
        return "Please enter a valid input."

    # Transform the input text using the vectorizer
    input_vector = vectorizer.transform([input_text])

    # Calculate the cosine similarities between input and all dataset entries
    similarities = cosine_similarity(input_vector, tfidf_matrix)

    # Get the index of the most similar product
    most_similar_index = similarities.argmax()

    # Return the details of the most similar product
    most_similar_product = data.iloc[most_similar_index]
    return most_similar_product

# Test the function
user_input = st.text_input("Enter your text: ")
if user_input:
    recommended_product = recommend_product(user_input)
    st.write("Recommended Product:")
    st.write("Category:", recommended_product['Category'])
    st.write("User:", recommended_product['Customer'])
    st.write("Rating:", recommended_product['Rating'])

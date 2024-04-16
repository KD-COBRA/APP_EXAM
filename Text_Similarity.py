# import pandas as pd
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer, WordNetLemmatizer
# from matplotlib import pyplot as plt
# from wordcloud import WordCloud

# df = pd.read_csv("WomensClothingE-CommerceReviews.csv")
# df.head()

# # Tokenization
# print("Tokenization:")
# tokens = nltk.word_tokenize(df['Review Text'].str.cat(sep=' '))
# print(tokens[:100])  # Print the first 10 tokens
# print(len(tokens))

# #Remove stopwords
# print("\nRemoving stopwords:")
# stop_words = set(stopwords.words('english'))
# filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
# print(filtered_tokens[:200])
# print(len(filtered_tokens))

# # Stemming
# print("\nStemming:")
# stemmer = PorterStemmer()
# stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
# print(stemmed_tokens[:100])
# print(len(stemmed_tokens))

# # Lemmatization
# print("\nLemmatization:")
# lemmatizer = WordNetLemmatizer()
# lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
# print(lemmatized_tokens[:200])
# print(len(lemmatized_tokens))

# # Clean text (remove special characters, punctuation, and numbers)
# print("\nCleaned text:")
# cleaned_tokens = [word for word in lemmatized_tokens if word.isalpha()]
# print(cleaned_tokens[:200])
# print(len(cleaned_tokens))

# freq_dist = nltk.FreqDist(cleaned_tokens)

# most_common_words = freq_dist.most_common(50)

# Create a WordCloud object
# wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(most_common_words))

# # Plot the WordCloud
# plt.figure(figsize=(10, 6))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.title('Most Common 50 Words')
# plt.axis('off')
# plt.show()











import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Function to process text
def process_text(text):
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    cleaned_tokens = [word for word in lemmatized_tokens if word.isalpha()]
    return cleaned_tokens

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("WomensClothingE-CommerceReviews.csv")
    return df

# Main function
def main():
    st.title("Text Processing App")
    
    # Load data
    df = load_data()
    
    # Display data
    st.subheader("Raw Data")
    st.write(df.head())
    
    # Process text
    processed_text = process_text(' '.join(df['Review Text'].dropna()))
    
    # Display processed text
    st.subheader("Processed Text")
    st.write(processed_text[:200])  # Display first 200 tokens
    
    # # Generate WordCloud
    # freq_dist = nltk.FreqDist(processed_text)
    # most_common_words = freq_dist.most_common(50)
    # wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(most_common_words))
    
    # # Display WordCloud
    # st.subheader("WordCloud")
    # plt.figure(figsize=(10, 6))
    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.title('Most Common 50 Words')
    # plt.axis('off')
    # st.pyplot()

if __name__ == "__main__":
    main()





















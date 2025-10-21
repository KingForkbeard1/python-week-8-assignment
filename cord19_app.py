import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from wordcloud import WordCloud

st.title("CORD-19 Data Explorer")
st.write("An interactive tool to explore COVID-19 research data (metadata.csv)")

try:
    df = pd.read_csv("metadata.csv")
    st.success("Dataset loaded successfully.")
except FileNotFoundError:
    st.error("metadata.csv not found. Please upload the file.")
    st.stop()

st.subheader("Dataset Overview")
st.write(df.head())
st.write(f"Shape of dataset: {df.shape}")
st.write(df.info())
st.write("Missing values per column:")
st.write(df.isnull().sum())

df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
df['year'] = df['publish_time'].dt.year
df['abstract_word_count'] = df['abstract'].astype(str).apply(lambda x: len(x.split()) if x != "nan" else 0)
cleaned_df = df.dropna(subset=['title', 'publish_time', 'journal'])

st.subheader("Cleaned Dataset Sample")
st.write(cleaned_df.head())

st.subheader("Basic Statistics")
st.write(cleaned_df.describe())

year_counts = cleaned_df['year'].value_counts().sort_index()
st.subheader("Publications Over Time")
fig1, ax1 = plt.subplots()
ax1.bar(year_counts.index, year_counts.values)
ax1.set_title("Publications by Year")
ax1.set_xlabel("Year")
ax1.set_ylabel("Number of Publications")
st.pyplot(fig1)

top_journals = cleaned_df['journal'].value_counts().head(10)
st.subheader("Top 10 Journals Publishing COVID-19 Research")
fig2, ax2 = plt.subplots()
sns.barplot(x=top_journals.values, y=top_journals.index, ax=ax2)
ax2.set_xlabel("Number of Papers")
ax2.set_ylabel("Journal")
ax2.set_title("Top 10 Journals")
st.pyplot(fig2)

st.subheader("Most Frequent Words in Paper Titles")
titles = " ".join(cleaned_df['title'].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(titles)
fig3, ax3 = plt.subplots(figsize=(10, 5))
ax3.imshow(wordcloud, interpolation='bilinear')
ax3.axis("off")
st.pyplot(fig3)

source_counts = cleaned_df['source_x'].value_counts().head(10)
st.subheader("Publications by Source")
fig4, ax4 = plt.subplots()
sns.barplot(x=source_counts.values, y=source_counts.index, ax=ax4)
ax4.set_xlabel("Number of Papers")
ax4.set_ylabel("Source")
ax4.set_title("Top Sources of Research Papers")
st.pyplot(fig4)

st.subheader("Filter by Year Range")
min_year, max_year = int(cleaned_df['year'].min()), int(cleaned_df['year'].max())
year_range = st.slider("Select year range", min_year, max_year, (2020, 2021))
filtered = cleaned_df[(cleaned_df['year'] >= year_range[0]) & (cleaned_df['year'] <= year_range[1])]
st.write(filtered[['title', 'journal', 'publish_time']].head(20))

st.markdown("---")
st.markdown("✅ **Insights:** Most COVID-19 papers were published between 2020 and 2021. Top journals include *medRxiv* and *Nature*. Common keywords are 'COVID-19', 'SARS-CoV-2', and 'pandemic'.")
st.markdown("**Reflection:** This assignment demonstrates how Python frameworks like Pandas and Streamlit simplify data analysis and visualization workflows, enabling efficient insight sharing through web applications.")

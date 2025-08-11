from app import indexing,gitabot
import streamlit as st

st.set_page_config(page_title="Gita Agent Bot", page_icon=":guardsman:", layout="wide")

st.title("Gita Agent Bot")
st.write("first create an index of the Bhagavad Gita by clicking the button below.")
if st.button("Index Bhagavad Gita"):
   indexing()
   st.success("Indexing complete!")
st.write("You can now ask questions about the Bhagavad Gita or search the web.")

question = st.text_input("Enter your question about the Bhagavad Gita or general knowledge:")
if st.button("Ask"):
    response = gitabot(question)
    st.write("🔮 Final Answer:", response)
import os
import streamlit as st

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import OpenAI
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts.prompt import PromptTemplate
from decouple import config

os.environ['OPENAI_API_KEY'] = config('KEY')

# llm stuff
llm = OpenAI(temperature=0.5)
embeddings = OpenAIEmbeddings(model="ada")

st.title("PDF Summarizer")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

question = st.text_input("What do you want your summary to focus on?")
custom_template= "\nWrite an executive summary on\n" + question + "\n using the following:\n {text}"
prompt = PromptTemplate(template=custom_template, input_variables=["text"])

if uploaded_file is not None:  
    with st.spinner('Reading pdf....'):
        loader = UnstructuredPDFLoader(uploaded_file.name)
        load_and_split = loader.load_and_split()
        st.success('PDF successfully read!')
    
    if st.button('Summarize'):
        with st.spinner('Summarizing...'):
            chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=prompt)
            final_summary = chain.run(load_and_split)
            st.success('Summary successfully generated!')
            st.write(final_summary)
else:
    st.write("Please upload a PDF file.")
from gpt_index import GPTSimpleVectorIndex, SimpleDirectoryReader, PromptHelper
from langchain import OpenAI
from gpt_index import  LLMPredictor
import streamlit as st
import os
from streamlit_option_menu import option_menu
from io import BytesIO
from PyPDF2 import PdfReader
import textwrap, ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from styling import local_css

def doc_search (files,llm_predictor):
    documents = ()
    new_content = False
    # get the value of each file into a string named "text"
    for file in files: 
        #title = file.name
        file_ext = os.path.splitext(file.name)[1]
        file_data = file.getvalue()
        # Create a BytesIO object from the binary data
        byte_file = BytesIO(file_data)
        text=convert_to_txt(byte_file,file_ext)
        if(file_ext == ".txt"):
            text = str(text)
        
        # check to see if text is in "corpus.txt" already, if not, add to it.

        if not os.path.exists(f'inputs/corpus.txt'):
            with open(f"inputs/corpus.txt","a",encoding="utf=8") as f:
                f.write(text+" ")
                new_content = True
        else: 
            with open(f'inputs/corpus.txt', "r", encoding="utf-8") as f:
                existing_content = f.read()
            if text in existing_content:   
                continue
            else:
                with open(f"inputs/corpus.txt","a",encoding="utf=8") as f:
                    f.write(text+" ")
                    new_content = True
    if new_content:
        documents = SimpleDirectoryReader('inputs').load_data()
        print ("starting index...")
        index = GPTSimpleVectorIndex(documents,llm_predictor=llm_predictor)
        index.save_to_disk(f'index/index.json')
    else: 
        print ("Index exists")
        index = GPTSimpleVectorIndex.load_from_disk(f'index/index.json',llm_predictor=llm_predictor)

    return index
    

@st.experimental_memo()
def chunk_text(text,char):
    sentences = text.split(". ")
    wrapped_sentences = []
    for sentence in sentences:
        wrapped_sentences.extend(textwrap.wrap(sentence, char))
    return ". ".join(wrapped_sentences)

@st.experimental_memo()
def convert_to_txt (file,file_ext):
    full_text=""
    if file_ext == ".epub":
        try:
            book = epub.read_epub(file)
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    html_content = item.get_content().decode('utf-8')
                    soup = BeautifulSoup(html_content, 'html.parser')  
                    text = soup.get_text()
                    full_text += text
        except Exception as oops:
            st.error("Something wrong with the file - could be corrupt, encrypted, or wrong format. Error: "+str(oops))
    elif file_ext == ".pdf":
        try:
            pdf = PdfReader(file)
            for page in pdf.pages:
                full_text+=page.extract_text()+" "
        except Exception as oops:
            st.error("Something wrong with the file - could be corrupt, encrypted, or wrong format. Error: "+str(oops))
    else:
        full_text=file.read()
    return full_text

def sidebar():
    with st.sidebar:
        option_menu(
            menu_title=None,
            options = ["AFAIK"],
            icons=["search"],
            default_index = 0,
        )
        st.markdown("---")

        api_key_input = st.text_input(
            label="Enter your OpenAI API key:",
            type="password",
            placeholder="sk-xxxxxxxxxxxxxxxxxxxxxxxx",
            help="You can get your API key from https://platform.openai.com/account/api-keys.",
            value=st.session_state.get("OPENAI_API_KEY", ""),
        )
        st.session_state["OPENAI_API_KEY"] = api_key_input
        st.markdown("---")
        st.markdown("Made by some random Asian guy 😅")

if __name__=="__main__":

    if not os.path.exists('inputs'):
        os.mkdir('inputs')
    if not os.path.exists('index'):
        os.mkdir('index')
    st.set_page_config(page_title="Documents Q&A", page_icon=":guardsman:", layout="wide")
    local_css('style/style.css')
    sidebar()

    st.header("Ask Your Docs")
    if not st.session_state["OPENAI_API_KEY"]:
        st.info("Add your OpenAI key on the left to get started.",icon="ℹ")
    files = st.file_uploader(label="Upload the document(s) you want to query:",type=["pdf","epub","txt"],accept_multiple_files=True)
    placeholder = st.empty()
    if os.path.exists(f'index/index.json'):
        reset = placeholder.button (label="Clear Existing Memories")
        if reset: 
            files = os.listdir('inputs')
            for file in files: 
                os.remove(f'inputs/{file}')
            files = os.listdir('index')
            for file in files: 
                os.remove(f'index/{file}')
            placeholder.empty()
    # ask a question
    question = st.text_input("Now ask a good question:", key="1", placeholder="Type a question and press Ask")
    st.write("")
    ask = st.button(label="Ask",type='primary')
    
    if question and ask:
        if not st.session_state["OPENAI_API_KEY"]:
            st.error("Please configure your OpenAI API key in the left sidebar")
        else:
            llm_predictor = LLMPredictor(llm=OpenAI(openai_api_key=st.session_state["OPENAI_API_KEY"], temperature=0,model_name="text-davinci-003"))
            # prompt_helper = PromptHelper(max_input_size=4096, num_output=3000, max_chunk_overlap=0)
            index = doc_search(files,llm_predictor)
            try:
                with st.spinner("Working hard to get you a good answer..."):
                    st.markdown(index.query(question, llm_predictor=llm_predictor)) 
            except Exception as oops:
                st.error("GPT Server Error: " + str(oops)+" Please try again.")
    
    
    
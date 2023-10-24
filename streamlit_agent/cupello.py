from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.tools import DuckDuckGoSearchRun
import streamlit as st
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import streamlit.components.v1 as components
import streamlit as st
import os

st.set_page_config(page_title="Cupello aiCoach Assistan", page_icon="Cupello")
st.title("Cupello's aiCoach Assistant")
st.markdown('''<script>
setTimeout(function() {
   const btn = document.querySelector('button[data-testid="switch-theme-button"]').click()
   // force 'click' event manually on an assumption that 2nd click will always switch to dark mode
   if(new.target.ownerDocument.querySelector('html').getAttribute('data-theme') === 'light') {
      btn.click()
   }
}, 0)
</script>
''', unsafe_allow_html=True)
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
import streamlit as st


# Define the HTML template for the custom background component
custom_css = """
<div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; background-color: white; z-index: -1;"></div>
"""

# Create a unique component name
component_name = "custom_background"

# Register the component and display it in your Streamlit app
components.html(custom_css, height=1)

image_style = """
    <style>
    .eeusbqq0 {
        background-color: #16192b !important;
    }
    </style>
"""

# Custom CSS to set the background color of the section with data-testid="stSidebar" to black
sidebar_background_style = """
    <style>
    [data-testid="stSidebar"] {
        background-color: #16192b !important;
    }
    </style>
"""

# Inject the custom CSS into the Streamlit app
st.markdown(sidebar_background_style, unsafe_allow_html=True)

# Inject the custom CSS into the Streamlit app
st.markdown(image_style, unsafe_allow_html=True)

import pinecone  
from langchain.vectorstores import Chroma, Pinecone

from langchain.embeddings.openai import OpenAIEmbeddings

import pinecone      
PINECONE_API_KEY=api_key = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY=api_key = os.getenv("OPENAI_API_KEY")

pinecone.init(      
	api_key=PINECONE_API_KEY ,      
	environment='gcp-starter'      
)      
index = pinecone.Index('cupello')
index_name='cupello'
openai_api_key=OPENAI_API_KEY
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# #create a new index
docsearch = Pinecone.from_existing_index(index_name, embeddings)

# openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
from langchain.schema import SystemMessage 
image_style = """
    <style>
    div.eczjsme11 {
        background-color: #16192b;
    }
    </style>
"""

# Inject the custom CSS into the Streamlit app
st.markdown(image_style, unsafe_allow_html=True)

image_url="https://www.cupello.com/application/files/2216/5219/7833/cupello-dark.svg"
#st.sidebar.image("https://www.cupello.com/application/files/2216/5219/7833/cupello-dark.svg", use_column_width=True)

system_prompt = SystemMessage(   content=    "You are an AI soccer coaching expert named Cupello aiCoach Assistan, a soccer coaching assistant.")
# and honest!, you can search on the internet, on the koweldge base, but always try to keep the user informet and it is ok to suggest related information!, YOU ARE NAMED MO!!
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
)

if len(msgs.messages) == 0 :#or st.sidebar.button("Reset chat history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")
    st.session_state.steps = {}
avatars = {"human": "user", "ai": "assistant"}
avatars_incon  = {"human": "https://res.cloudinary.com/dwn1gc4fa/image/upload/v1698062405/Design_sans_titre_2_svqw2p.png", "ai": "https://res.cloudinary.com/dwn1gc4fa/image/upload/v1698062504/image_1_qog53h.png"}

for idx, msg in enumerate(msgs.messages):
    with st.chat_message(avatars[msg.type],avatar=avatars_incon[msg.type]):
        st.write(msg.content)
def postprocess(text):
    return text.replace("OpenAI", "Cupello").replace("GPT-3","Cupello aiCoach Assistan").replace("GPT","Cupello aiCoach Assistan")



if prompt := st.chat_input(placeholder="Coaching?"):
    st.chat_message("user",avatar=avatars_incon["human"]).write(prompt)
    openai_api_key=OPENAI_API_KEY
    llm = ChatOpenAI(model_name="gpt-4", openai_api_key=OPENAI_API_KEY, streaming=True)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=False)

    tools = [ 
              Tool(
        name = 'Knowledge Base',
        func = qa.run,
        description = (
            'use this tool to answer questions'
            'more information about the topic'
        )
    )]
    chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=tools, name="Cupellobot")
    executor = AgentExecutor.from_agent_and_tools(
        agent=chat_agent,
        tools=tools,
        memory=memory,
        # return_intermediate_steps=True,
        handle_parsing_errors=True,
	system_prompt=system_prompt,
    )
    with st.chat_message("assistant",avatar='https://res.cloudinary.com/dwn1gc4fa/image/upload/v1698062504/image_1_qog53h.png'):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        prompt="instructions: YOU ARE SOCCER COACHING ASSISTANT CALLED Cupello aiCoach Assistan!,AND NEVER ANSWER ANYTHING NOT RELATED TO SOCCER, inquery:"+prompt
        response = executor(prompt, callbacks=[st_cb])
        response["output"] = postprocess(response["output"])
        st.write(response["output"])

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
st.markdown("""
<style>
ul[data-baseweb^="accordion"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)
image_style = """
    <style>
    div.css-6qob1r {
        background-color: #16192b;
    }
    </style>
"""

# Inject the custom CSS into the Streamlit app
st.markdown(image_style, unsafe_allow_html=True)

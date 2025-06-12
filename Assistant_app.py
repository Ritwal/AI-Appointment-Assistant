import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from workflow import app, State ,tools 



st.set_page_config(
    page_title="AI Appointment Assistant",
    page_icon="📅",
    layout="centered"
)

st.title("📅 AI Appointment Assistant")
st.caption("I can answer questions and book appointments for you.")



# Streamlit Chat Interface Logic ---


if "messages" not in st.session_state:
    st.session_state.messages = []
if "graph_state" not in st.session_state:
    st.session_state.graph_state = {"messages": []}
if "tools" not in st.session_state:
    st.session_state.tools = {}

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What can I help you with?"):
    
    if not st.session_state.tools:
        st.session_state.tools = tools

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    current_graph_state = st.session_state.graph_state
    current_graph_state["messages"].append(HumanMessage(content=prompt))
    
    
    
    try:
        with st.spinner("Thinking..."):
            final_state = app.invoke(current_graph_state)
            ai_response = final_state['messages'][-1].content
            
            st.session_state.graph_state = final_state
            
            with st.chat_message("assistant"):
                st.markdown(ai_response)
            
            st.session_state.messages.append({"role": "assistant", "content": ai_response})

    
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")




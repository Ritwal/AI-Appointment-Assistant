import streamlit as st
import os
from typing import TypedDict, Annotated, List
from langchain_openai import ChatOpenAI
from openai import AuthenticationError, APIConnectionError, RateLimitError
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
import logging

api_key = 'your-api-key'  # replace with your real key


class AppointmentDetails(BaseModel):
    date: str = Field(..., description="The date of the appointment, e.g., '2024-10-29'.")
    time: str = Field(..., description="The time of the appointment, e.g., '3:00 PM'.")

class IntentDetection(BaseModel):
    intent: str = Field(description="Must be one of: 'book_appointment', 'general_query', 'end_conversation'.")

class ModeDetection(BaseModel):
    mode: str = Field(..., description="Must be one of: 'Virtual', 'In person'.")

class State(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    intent: str
    date: str
    time: str
    mode: str


def get_llm_with_tools(api_key):
    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key)
    return {
        "llm": llm,
        "intent_tool": llm.with_structured_output(IntentDetection),
        "details_tool": llm.with_structured_output(AppointmentDetails),
        "mode_tool": llm.with_structured_output(ModeDetection)
    }

tools = get_llm_with_tools(api_key)


def handle_user_input(state: State, tools=tools):
    user_input = state['messages'][-1].content.lower()
    if any(phrase in user_input for phrase in ["no", "that's all", "nothing else", "thank you"]):
        return {"intent": "end_conversation"}
    response = tools["intent_tool"].invoke(f"Analyze the user's message and determine their intent. User message: '{user_input}'")
    return {"intent": response.intent}

def handle_general_query(state: State, tools=tools):
    response = tools["llm"].invoke(state['messages'] + [SystemMessage(content="Address the user's general query. Greet user back.")])
    return {"messages": [response]}

def extract_appointment_details(state: State, tools=tools):
    history = "\n".join([f"{msg.type}: {msg.content}" for msg in state['messages']])
    prompt = f"From the conversation history, extract date and time format from the last user message. Today is Wednesday, June 11, 2025.\n\nHistory:\n{history}\n\nIf date or time is missing from the last user message, return an empty string for it."
    response = tools["details_tool"].invoke(prompt)
    return {"date": response.date or "", "time": response.time or ""}

def clarify_missing_details(state: State):
    last_time = state.get('time')
    last_date = state.get('date')

    if not last_date and not last_time:
        message = AIMessage(content="Sure, what date and time would you like to book your appointment?")
    elif not last_date:
        message = AIMessage(content=f"Got it for {last_time}. What date would you prefer?")
    elif not last_time:
        message = AIMessage(content=f"Noted for {last_date}. What time works best for you?")
    else:
        # Clarify if time is ambiguous (e.g., "4", "10", etc.)
        if last_time.strip().isdigit() or not any(tag in last_time.lower() for tag in ['am', 'pm']):
            message = AIMessage(content=f"You mentioned {last_time}. Do you mean {last_time} AM or {last_time} PM?")
        else:
            message = AIMessage(content=f"Just to confirm — {last_date} at {last_time}, right?")
    return {"messages": [message]}

def prompt_for_mode(state: State):
    message = AIMessage(content=f"Great, I have you down for {state.get('date')} at {state.get('time')}. Is this a 'Virtual' or 'In person' appointment?")
    return {"messages": [message]}

def register_mode(state: State, tools=tools):
    history = "\n".join([f"{msg.type}: {msg.content}" for msg in state['messages']])
    prompt = f"From the conversation history, extract user's mode of appointment.\n\nHistory:\n{history}\n\nIf mode is missing from the last user message, return an empty string for it."
    response = tools["mode_tool"].invoke(prompt)
    return {"mode": response.mode or ""}

def confirm_appointment(state: State):
    message = AIMessage(content=f"✅ Your {state.get('mode')} appointment is confirmed for {state.get('date')} at {state.get('time')}. You'll receive a calendar invite shortly.\n\nIs there anything else I can help you with?")
    return {"messages": [message]}

def handle_fallback(state: State):
    message = AIMessage(content="Thank you for chatting with us. Have a great day!")
    return {"messages": [message]}


# === ROUTING LOGIC ===
def route_by_intent(state: State):
    return state["intent"]

def check_details_extracted(state: State):
    if state.get('date') and state.get('time'):
        return "prompt_for_mode"
    return "clarify_missing_details"


# === BUILD WORKFLOW GRAPH ===
workflow = StateGraph(State)

workflow.add_node("handle_user_input", handle_user_input)
workflow.add_node("handle_general_query", handle_general_query)
workflow.add_node("extract_appointment_details", extract_appointment_details)
workflow.add_node("clarify_missing_details", clarify_missing_details)
workflow.add_node("prompt_for_mode", prompt_for_mode)
workflow.add_node("register_mode", register_mode)
workflow.add_node("confirm_appointment", confirm_appointment)
workflow.add_node("handle_fallback", handle_fallback)

workflow.set_entry_point("handle_user_input")

workflow.add_conditional_edges("handle_user_input", route_by_intent, {
    "book_appointment": "extract_appointment_details",
    "general_query": "handle_general_query",
    "end_conversation": "handle_fallback"
})

workflow.add_conditional_edges("extract_appointment_details", check_details_extracted, {
    "prompt_for_mode": "prompt_for_mode",
    "clarify_missing_details": "clarify_missing_details"
})

workflow.add_edge("prompt_for_mode", 'register_mode')
workflow.add_edge("register_mode", 'confirm_appointment')

workflow.add_edge("clarify_missing_details", END)
workflow.add_edge("handle_general_query", END)
workflow.add_edge("confirm_appointment", END)
workflow.add_edge("handle_fallback", END)

app = workflow.compile()

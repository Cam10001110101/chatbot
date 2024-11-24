import os
import yaml
import streamlit as st
import json
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from Tools.TavilySearchResults import get_tavily_search_tool
from Tools.DateTime import get_datetime_tool

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Define state with proper typing
class State(TypedDict):
    messages: Annotated[list, add_messages]

def load_config():
    """Load configuration from yaml file"""
    try:
        with open('config.yaml', 'r') as config_file:
            return yaml.safe_load(config_file)
    except Exception as e:
        st.error(f"Error loading config: {str(e)}")
        return {
            'models': {
                'openai': {
                    'temperature': 0,
                    'name': 'gpt-4o-mini'
                }
            }
        }

def setup_graph():
    """Setup and return the LangGraph workflow"""
    config = load_config()
    
    # Initialize tools
    tavily_tool = get_tavily_search_tool()
    datetime_tool = get_datetime_tool()
    tools = [tavily_tool, datetime_tool]
    
    # Create the base LLM
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY
    ).bind_tools(tools)
    
    # Create the graph
    workflow = StateGraph(State)
    
    # Define the agent node function
    def agent_node(state: State):
        messages = state["messages"]
        try:
            # Add system message for tool usage
            if len(messages) == 1 and isinstance(messages[0], HumanMessage):
                system_message = """You are a helpful AI assistant with access to the following tools:
                1. Tavily Search - For finding accurate and up-to-date information from the internet
                2. DateTime - For getting the current date, time, and timezone information
                
                When users ask for information like latest news or current events, always check the current 
                date and time first using the DateTime tool to provide timely and contextual responses. Then 
                use the Tavily search tool to find relevant information, and summarize the results in a clear 
                and concise way, mentioning how recent the information is relative to the current date."""
                messages = [HumanMessage(content=system_message)] + messages
            
            response = llm.invoke(messages)
            return {"messages": [response]}
            
        except Exception as e:
            st.error(f"Error in agent_node: {str(e)}")
            return {"messages": [AIMessage(content=f"Error: {str(e)}")]}
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools=tools))
    
    # Add edges
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END: END
        }
    )
    workflow.add_edge("tools", "agent")
    workflow.add_edge(START, "agent")
    
    # Create checkpointer for state persistence
    memory = MemorySaver()
    
    return workflow.compile(checkpointer=memory)

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "graph" not in st.session_state:
        st.session_state.graph = setup_graph()

def display_message(message):
    """Display a message in the Streamlit chat interface"""
    if isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)
            # Display additional message attributes if they exist
            if hasattr(message, 'additional_kwargs'):
                with st.expander("Show raw message data"):
                    st.json(message.additional_kwargs)
    elif hasattr(message, 'tool_call'):
        with st.chat_message("tool", avatar="🔧"):
            st.write("Tool Call:")
            # Display the tool name and input
            if hasattr(message.tool_call, 'name'):
                st.write(f"**Tool:** {message.tool_call.name}")
            if hasattr(message.tool_call, 'args'):
                with st.expander("Show tool input"):
                    st.json(message.tool_call.args)
            # Display the tool output
            st.write("**Output:**")
            st.write(message.content)
            # If there's raw output data, show it in an expander
            if hasattr(message, 'raw_output'):
                with st.expander("Show raw output"):
                    st.json(message.raw_output)
    else:
        with st.chat_message("user"):
            st.write(message.content)
            # Display additional message attributes if they exist
            if hasattr(message, 'additional_kwargs'):
                with st.expander("Show raw message data"):
                    st.json(message.additional_kwargs)

def main():
    st.set_page_config(page_title="LangChain - OpenAI - gpt-4o-mini", page_icon="🤖")
    st.title("LangChain - OpenAI - gpt-4o-mini")
    
    # Initialize session state
    initialize_session_state()
    
    # Display chat history
    for message in st.session_state.messages:
        display_message(message)
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history
        user_message = HumanMessage(content=prompt)
        st.session_state.messages.append(user_message)
        display_message(user_message)
        
        try:
            # Process through graph
            config = {"configurable": {"thread_id": "1"}}
            events = st.session_state.graph.stream(
                {"messages": [("user", prompt)]},
                config,
                stream_mode="values"
            )
            
            # Display responses
            for event in events:
                if "messages" in event:
                    for message in event["messages"]:
                        st.session_state.messages.append(message)
                        display_message(message)
                        
        except Exception as e:
            st.error(f"Error processing input: {str(e)}")

if __name__ == "__main__":
    main()

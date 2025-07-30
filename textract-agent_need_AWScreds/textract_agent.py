# agent.py
from langchain_ollama import ChatOllama
from langchain.agents import initialize_agent, AgentType
from textract_tool import TextractTool

llm = ChatOllama(model="llama3:8B")          # streamed=False by default
textract_tool = TextractTool()

extract_agent = initialize_agent(
    tools=[textract_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
)

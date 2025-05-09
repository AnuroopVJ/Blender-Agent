from pydantic_ai import Agent, DocumentUrl, RunContext, Tool
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
import os
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from pydantic import BaseModel

from db_stuff import search_RAG

load_dotenv()

key_groq = os.getenv('GROQ_API_KEY')


class TechQuery(BaseModel):
    query: str

class ResponseModel(BaseModel):
    response: str
    docs_search_tool_used: bool
    internet_search_tool_used: bool

model = GroqModel(
    'llama-3.3-70b-versatile', provider=GroqProvider(api_key=key_groq)
)
agent = Agent(model=model,system_prompt=f"""
-Firstly thinkif the reponse would require internet search, if yes use 'internet_search_tool' to search the internet,or if it requires docs search use 'common_FAQs' tool, 
-responses could be made using the 'common_FAQs' tool that allows you to search the documentation of Blender (Useful for blender's core questions),
-pls use that tool in every response and make responses based on results from it
-Use 'internet_search_tool' -->Useful for finding anything else"""
, result_type=ResponseModel,deps_type=TechQuery
)


@agent.system_prompt
async def system_prompt(ctx: RunContext[TechQuery]):
    return f"CustomerRequest: {str(ctx)} \n"

@agent.tool_plain
async def internet_search_tool(inp: str):
    try:
        results_ddgs = list(DDGS().text(inp, max_results=5))
        if not results_ddgs:
            return "No relevant results found on the internet."
        return results_ddgs
    except Exception as e:
        print("ERROR: ", e)
        return f"[ERROR] An error occurred!"

@agent.tool_plain
async def common_FAQs(inp: str):
    try:
        print("Query: ", inp)
        res = search_RAG(inp)
        if res == "No relevant documents found in the FAQs.":
            print("No results in FAQs. Searching the internet...")
            internet_results = await internet_search_tool(inp)
            return f"FAQs Search: {res}\nInternet Search: {internet_results}"
        print("Search result: ", res, "\n\n\n\n")
        return res
    except Exception as e:
        print("ERROR: ", e)
        return f"[ERROR] An error occurred!"


user_input = input("Enter Your Question: ")  
result = agent.run_sync(
    [
        f"{user_input}",
    
    ]
)
print(result.output)
from dotenv import load_dotenv
load_dotenv()
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
import os
from langchain_google_genai import ChatGoogleGenerativeAI
# search tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities import StackExchangeAPIWrapper
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langchain_core.tools import Tool
import re
from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
from langchain_community.utilities.github import GitHubAPIWrapper
import getpass
github = GitHubAPIWrapper()
toolkit = GitHubToolkit.from_github_api_wrapper(github)



os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
for env_var in [
    "GITHUB_APP_ID",
    "GITHUB_APP_PRIVATE_KEY",
    "GITHUB_REPOSITORY",
]:
    if not os.getenv(env_var):
        os.environ[env_var] = getpass.getpass()

# ----------------- CUSTOM URL-AWARE SEARCH FUNCTION -----------------

def google_search_with_urls(query: str) -> str:
    """Google search that always includes URLs in results."""
    # Use news type for news queries, web for general queries
    search_type = "news" if any(term in query.lower() for term in ["news", "latest", "recent"]) else "search"
    search = GoogleSerperAPIWrapper(type=search_type)
    results = search.results(query)
    
    formatted_results = []
    
    # Handle web search results
    if "organic" in results and results["organic"]:
        for i, result in enumerate(results["organic"][:5]):
            title = result.get('title', 'No title')
            link = result.get('link', 'No URL available')
            snippet = result.get('snippet', '')
            
            formatted_result = (
                f"{i+1}. {title}\n"
                f"URL: {link}\n"
                f"{snippet[:100]}...\n"
            )
            formatted_results.append(formatted_result)
    
    # Handle news results
    elif "news" in results and results["news"]:
        for i, result in enumerate(results["news"][:5]):
            title = result.get('title', 'No title')
            source = result.get('source', 'Unknown source')
            link = result.get('link', 'No URL available')
            snippet = result.get('snippet', '')
            
            formatted_result = (
                f"{i+1}. {title}\n"
                f"Source: {source}\n"
                f"URL: {link}\n"
                f"{snippet[:100]}...\n"
            )
            formatted_results.append(formatted_result)
    
    # If we found results
    if formatted_results:
        return "SEARCH RESULTS WITH URLS (INCLUDE THESE URLS IN YOUR RESPONSE):\n\n" + "\n".join(formatted_results)
    else:
        return f"No relevant results found for '{query}'."


def tavily_search_with_urls(query: str) -> str:
    """Tavily search that always includes URLs in results."""
    tavily = TavilySearchResults()
    results = tavily.run(query)
    
    # Parse the results to ensure URLs are included
    # Tavily typically returns a JSON structure or formatted text
    # First, check if results is already a list of dictionaries
    formatted_results = []
    
    try:
        # If it's a string (which can happen), try to extract URLs
        if isinstance(results, str):
            # Try to find URL patterns in the results
            urls = re.findall(r'https?://[^\s]+', results)
            
            # If we found URLs, format them nicely
            if urls:
                lines = results.split('\n')
                current_result = []
                result_index = 1
                
                for line in lines:
                    if line.strip():
                        current_result.append(line)
                        
                        # If this line contains a URL, format the current result
                        if any(url in line for url in urls):
                            formatted_result = f"{result_index}. " + "\n".join(current_result)
                            formatted_results.append(formatted_result)
                            current_result = []
                            result_index += 1
                
                # Handle any remaining results
                if current_result:
                    formatted_result = f"{result_index}. " + "\n".join(current_result)
                    formatted_results.append(formatted_result)
            else:
                # If no URLs found, ensure they are prominently mentioned in results
                # Split by number patterns that might indicate results
                pattern = re.compile(r'^\d+\.', re.MULTILINE)
                result_sections = pattern.split(results)
                
                for i, section in enumerate(result_sections):
                    if section.strip():
                        # Append "URL not available" to make it clear
                        formatted_result = f"{i+1}. {section.strip()}\nURL: Source did not provide URL"
                        formatted_results.append(formatted_result)
        
        # If it's a list (common for Tavily)
        elif isinstance(results, list):
            for i, result in enumerate(results[:5]):
                if isinstance(result, dict):
                    title = result.get('title', 'No title')
                    url = result.get('url', result.get('link', 'No URL available'))
                    content = result.get('content', result.get('snippet', ''))
                    
                    formatted_result = (
                        f"{i+1}. {title}\n"
                        f"URL: {url}\n"
                        f"{content[:100]}...\n"
                    )
                    formatted_results.append(formatted_result)
    
    except Exception as e:
        # Fallback for any parsing errors
        return f"Search completed but had trouble formatting results. Original results: {results}\nError: {str(e)}"
    
    # If we successfully formatted results
    if formatted_results:
        return "SEARCH RESULTS WITH URLS (INCLUDE THESE URLS IN YOUR RESPONSE):\n\n" + "\n".join(formatted_results)
    else:
        # If we couldn't format them, return the original with a note
        return f"Search results (please include any URLs in your response):\n\n{results}"



def stack_exchange_with_urls(query: str) -> str:
    """Stack Exchange search that always includes URLs in results."""
    stack_exchange = StackExchangeAPIWrapper()
    results = stack_exchange.run(query)
    
    # Try to extract and highlight URLs from Stack Exchange results
    formatted_results = []
    
    # First, see if we can extract structured information with URLs
    # Stack Exchange often returns with titles and links
    try:
        # If the result is a string, try to extract question titles and URLs
        if isinstance(results, str):
            # Find question blocks using common patterns in Stack Exchange results
            question_blocks = re.split(r'\n\s*\n', results)
            
            for i, block in enumerate(question_blocks):
                if block.strip():
                    # Try to extract URLs
                    urls = re.findall(r'https?://[^\s]+', block)
                    url_text = f"URL: {urls[0]}" if urls else "URL: Not available in results"
                    
                    # Format with title and URL emphasized
                    title_match = re.search(r'^(.+?)(?:\n|$)', block)
                    title = title_match.group(1) if title_match else "Question"
                    
                    formatted_result = (
                        f"{i+1}. {title}\n"
                        f"{url_text}\n"
                        f"{block}\n"
                    )
                    formatted_results.append(formatted_result)
        
    except Exception as e:
        # Fall back to returning the original with a note about URLs
        return f"Stack Exchange results (please include any URLs in your response):\n\n{results}\n\nNote: {str(e)}"
    
    # If we successfully formatted results
    if formatted_results:
        return "STACK EXCHANGE RESULTS WITH URLS (INCLUDE THESE URLS IN YOUR RESPONSE):\n\n" + "\n".join(formatted_results)
    else:
        # If we couldn't format them, return the original with a note
        return f"Stack Exchange results (please include any URLs if present):\n\n{results}"





# tavily = TavilySearchResults()
# stackexchange = StackExchangeAPIWrapper()
# serper = GoogleSerperAPIWrapper()





google_search_tool = Tool(
    name="google_search",
    description="Search Google for recent or factual information. Results will include URLs.",
    func=google_search_with_urls
)

tavily_search_tool = Tool(
    name="tavily_search",
    description="Search Tavily for comprehensive information. Results will include URLs.",
    func=tavily_search_with_urls
)

stack_exchange_tool = Tool(
    name="stack_exchange_search",
    description="Search Stack Exchange for coding and technical information. Results will include URLs.",
    func=stack_exchange_with_urls
)


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", 
    temperature=0
)

# tools
tools = [tavily_search_tool, stack_exchange_tool, google_search_tool]

llm_with_tools = llm.bind_tools(tools=tools)



class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical", "analysis", "tool_calling_llm"] = Field(
        ...,
        description="Classify if the message requires an emotional (therapist) or logical or analysis or tool calling response."

    ) 

class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None


def classify_message(state:State):
    last_message = state["messages"][-1]
    classifier_llm = llm.with_structured_output(MessageClassifier)

    result = classifier_llm.invoke(
        [
            {
                "role":"system",
                "content":"""
                Classify the user message as either:
                - 'emotional': If it asks for emotional support, therapy, deals with feelings, or personal problems.
                - 'logical': If it asks for facts, information, logical analysis, or practical solutions
                - 'analysis': Unified route handling SQL, Python data analysis, and ML/statistical analysis requests.
    Automatically determines the type of analysis needed based on input context.
                - 'tool_calling_llm': If it asks for any recent topic or recent code generation
"""
            },
            {
                "role":"user",
                "content":last_message.content
            }
        ]
    )
    return {"message_type":result.message_type}

def router(state:State):
    message_type = state.get("message_type","analysis")
    if message_type == "emotional":
        return {
            "next": "therapist"
        }
    elif message_type == "logical":
        return {
            "next":"logical"
        }
    elif message_type == "tool_calling_llm":
        return {
            "next":"tools_llm"
        }
    else:
        return {
            "next":"analysis"
        }

# Node definition
def tool_calling_llm(state:State):
    return {
        "messages":[llm_with_tools.invoke(state["messages"])]
    }




def therapist_agent(state:State):
    last_message = state["messages"][-1]
    messages = [
        {
            "role":"system",
            "content": """
           You are a compassionate therapist.
           Focus on the emotional aspects of the user's message.
           Show empathy, validate their feelings, and help them process their emotions.
           Ask thoughtful questions to help them explore their feelings more deeply.
           Avoid giving logical solutions unless explicitly asked.
"""
        },
        {
            "role":"user",
            "content":last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages":[
        {"role":"assistant", "content":reply.content}
    ]}



def analysis_agent(state: State):
    last_message = state["messages"][-1]
    text_input = last_message["content"] if isinstance(last_message, dict) else getattr(last_message, "content", "")

    # Define keywords for query type detection
    sql_keywords = {'sql', 'query', 'database', 'db', 'table', 'select', 'join', 
                   'where', 'group by', 'aggregate', 'from', 'having', 'mysql', 'postgresql', 'sqlite', 'oracle', 'sql server', 'mongodb', 'redis', 'cassandra', 'dynamodb', 'vector database','redis', 'bigquery', 'snowflake', 'databricks', 'hive', 'spark'}
    ml_keywords = {'machine learning', 'ml', 'predict', 'classification', 'regression',
                  'clustering', 'train', 'model', 'sklearn', 'tensorflow', 'keras',
                  'neural', 'deep learning', 'statistics', 'statistical','analysis', 'data', 'analyze'
                  'numpy', 'pandas', 'torch', 'neural network', 'training', 'testing', 'validation', 'dataset', 'feature', 'label', 'training set', 'test set', 'cross-validation', 'hyperparameter', 'tuning', 'pipeline', 'feature selection', 'feature engineering', 'data preprocessing', 'data cleaning', 'data augmentation', 'model evaluation', 'accuracy', 'precision', 'recall', 'f1 score', 'roc curve', 'confusion matrix', 'overfitting', 'underfitting', 'bias-variance tradeoff', 'ensemble', 'random forest', 'gradient boosting', 'xgboost', 'lightgbm', 'catboost', 'svm', 'support vector machine', 'decision tree', 'naive bayes', 'k-nearest neighbors', 'knn', 'pca', 'principal component analysis', 't-sne', 'tsne', 'umap', 'analyze data'
                  }

    is_sql_query = any(keyword in text_input.lower() for keyword in sql_keywords)
    is_ml_query = any(keyword in text_input.lower() for keyword in ml_keywords)

    # Select appropriate prompt based on query type
    if is_sql_query:
        prompt_template = (
            "You are an advanced SQL query generation assistant. Generate a precise, efficient SQL query "
            "based on the following user prompt:\n\n"
            "User Prompt: ```{input}```\n\n"
            "Guidelines for SQL Query Generation:\n"
            "- Focus on generating a practical, optimized SQL query\n"
            "- Consider different database systems (MySQL, PostgreSQL, SQLite)\n"
            "- Ensure the query is syntactically correct and follows best practices\n"
            "- Include appropriate JOIN, WHERE, GROUP BY, or aggregate functions as needed\n"
            "- If the query is too vague or unrelated to data analysis, politely decline\n\n"
            "Required Output:\n"
            "- Clean, executable SQL query\n"
            "- Relevant to data analysis or database querying\n"
            "### CODE (NO PREAMBLE)"
        )
    elif is_ml_query:
        prompt_template = (
            "You are an advanced data science assistant. Generate a precise Machine Learning or Statistical Analysis "
            "code snippet based on the following user prompt:\n\n"
            "User Prompt: ```{input}```\n\n"
            "Guidelines for Code Generation:\n"
            "- Focus on generating a practical, concise code snippet\n"
            "- Use Python with libraries like scikit-learn, pandas, or numpy\n"
            "- Ensure the code is relevant to the input query\n"
            "- Include necessary imports and data preprocessing steps\n"
            "- If the query is too vague, politely decline\n\n"
            "Required Output:\n"
            "- Clean, executable Python code snippet\n"
            "- Relevant to machine learning, statistical analysis, or data science\n"
            "### CODE (NO PREAMBLE)"
        )
    else:
        prompt_template = (
            "Data Analysis Code Generation Instruction Set:\n"
            "Context:\n"
            "- You are an expert Python data analysis assistant specializing in Pandas, NumPy, Matplotlib, "
            "Plotly and Seaborn Library.\n"
            "- Your goal is to generate precise, accurate, and efficient code snippets.\n"
            "- Only generate code if the input is a valid data analysis or data manipulation query.\n\n"
            "Input Analysis Guidelines:\n"
            "1. Carefully evaluate the user's text input for data analysis relevance\n"
            "2. Identify specific data manipulation, transformation, or analysis requirements\n"
            "3. Ensure the code snippet is:\n"
            "   - Syntactically correct\n"
            "   - Follows best practices\n"
            "   - Uses appropriate Pandas or NumPy methods\n"
            "   - Handles potential edge cases\n\n"
            "User Input:\n"
            "```\n{input}\n```\n\n"
            "Output Format:\n"
            "- Provide complete, runnable code\n"
            "- No placeholders or pseudo-code\n"
            "- Include necessary imports\n"
            "- Demonstrate clear data manipulation logic\n"
            "### CODE (NO PREAMBLE)"
        )

    formatted_prompt = prompt_template.format(input=text_input)
    # Get code snippet
    code_response = llm.invoke([{"role": "user", "content": formatted_prompt}])
    code = code_response.content.strip().lstrip("```sql").lstrip("```python").rstrip("```")

    # Validate code (simple check)
    is_valid_code = bool(code) and (
        (is_sql_query and any(kw in text_input.lower() for kw in sql_keywords)) or
        (is_ml_query and any(kw in code.lower() for kw in ['sklearn', 'numpy', 'fit', 'predict'])) or
        (not is_sql_query and not is_ml_query and any(kw in code.lower() for kw in ['import', 'pandas', 'numpy', 'def', 'dataframe', 'seaborn', 'matplotlib', 'test']))
    )

    if is_valid_code:
        # Generate expected output
        output_prompt_template = (
            "Given the following Python visualization code, generate a description of the expected "
            "visual output that can be parsed into a visualization:\n\n"
            "```python\n{code}\n```\n\n"
            "Requirements:\n"
            "- For plots, specify the type (scatter, line, bar)\n"
            "- Include all data points in the format: (x1,y1), (x2,y2), ...\n"
            "- Specify axes labels and title if present\n"
            "- No explanations, just the structured output\n"
            "### EXPECTED_OUTPUT (NO PREAMBLE)"
        )

        explanation_prompt_template = (
            "Provide a {style} explanation of this {type}:\n\n"
            "```{lang}\n{code}\n```\n\n"
            "Explanation Guidelines:\n{guidelines}\n"
            "### EXPLANATION (NO PREAMBLE)"
        )

        code_type = "SQL query" if is_sql_query else "Python code"
        lang = "sql" if is_sql_query else "python"
        style = "comprehensive" if is_sql_query else "concise"
        
        if is_sql_query:
            guidelines = "- Purpose, components, insights, use cases, and performance considerations"
        elif is_ml_query:
            guidelines = "- Key steps, logic, libraries, and techniques used\n- Model selection and parameters"
        else:
            guidelines = "- Break down each line/block of code\n- Explain purpose and functionality\n- Use simple, clear language"

        expected_output = llm.invoke([{"role": "user", "content": output_prompt_template.format(code=code)}]).content
        explanation = llm.invoke([{"role": "user", "content": explanation_prompt_template.format(
            style=style,
            type=code_type,
            lang=lang,
            code=code,
            guidelines=guidelines
        )}]).content

        # Make sure to include a content field for LangChain compatibility
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": "Here's the analysis code you requested:",  # Add a content field
                    "code": code,
                    "expected_output": expected_output,
                    "explanation": explanation
                }
            ]
        }
    else:
        if is_sql_query:
            error_message = (
                "The query seems unrelated to SQL or data analysis. Please provide a specific SQL query.\n"
                "Suggestions:\n"
                "- Try queries like 'Find top 10 customers by sales'\n"
                "- Analyze data from specific tables\n"
                "- Generate aggregation or join-based queries"
            )
        elif is_ml_query:
            error_message = (
                "Please provide a more specific machine learning or statistical analysis query.\n"
                "Suggestions:\n"
                "- Train a classification model\n"
                "- Perform regression analysis\n"
                "- Implement clustering algorithm\n"
                "- Calculate statistical metrics"
            )
        else:
            error_message = (
                "I specialize in data analysis tasks. Please ask a query related to:\n"
                "- Data manipulation with Pandas\n"
                "- NumPy array operations\n"
                "- Data transformation\n"
                "- Statistical analysis\n"
                "- Data cleaning and preprocessing\n"
                "Examples:\n"
                "- Calculate mean and median of a column\n"
                "- Filter DataFrame based on conditions\n"
                "- Perform group by aggregation"
            )
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": error_message
                }
            ]
        }



def logical_content(state:State):
    last_message = state["messages"][-1]
    messages = [
        {
            "role":"system",
            "content":"""
            You are a purely logical assistant. Focus only on facts and information.
            Provide clear, concise answers based on logic and evidence.
            Do not address emotions or provide emotional support.
            Be direct and straightforward in your responses.
            """
        },
        {
            "role":"user",
            "content":last_message.content
        }
    ]   

    reply = llm.invoke(messages) 
    return {
        "messages": [
            {
                "role":"assistant",
                "content":reply.content
            }
        ]
    }

graph_builder = StateGraph(State)
graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("therapist", therapist_agent)
graph_builder.add_node("logical", logical_content)
graph_builder.add_node("analysis", analysis_agent)
graph_builder.add_node("tools_llm", tool_calling_llm)
graph_builder.add_node("tools", ToolNode(tools))

graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")

graph_builder.add_conditional_edges(
        "router",
        lambda state: state.get("next"),
        {"therapist":"therapist", "analysis":"analysis", "logical":"logical", "tools_llm":"tools_llm"}
    )

graph_builder.add_edge("therapist", END)
graph_builder.add_edge("analysis",END)
graph_builder.add_edge("logical", END)
graph_builder.add_conditional_edges(
    "tools_llm",
    tools_condition,
    END
)
graph_builder.add_edge("tools","tools_llm")

graph = graph_builder.compile()


def run_chatbot():
    state = {
        "messages":[],
        "message_type":None
    }

    while True:
        user_input = input("Message: ")
        if user_input == "exit":
            print("Bye")
            break
        
        state["messages"] = state.get("messages", []) + [
            {
                "role": "user",
                "content": user_input
            }
        ]

        state = graph.invoke(state)

        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            print(f"Assistant: {last_message.content}")



if __name__ == "__main__":
    run_chatbot()
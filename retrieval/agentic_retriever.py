import os
import sys
import structlog;log=structlog.get_logger()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # not the nicest way of handling this, but oh well...
from retrieval.verified_retriever import Validation
from retrieval.retriever import Retriever, Metadata
from typing_extensions import TypedDict, Annotated, List, Tuple, Union, Generator
from rapidfuzz import process, fuzz

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition, InjectedState
from langgraph.types import Command

from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId

from utils import get_llm_client, AGENTIC_RETRIEVER_SYSTEM_PROMPT, handle_attached_files

script_dir = os.path.dirname(os.path.abspath(__file__))
set_llm_cache(SQLiteCache(database_path=os.path.join(script_dir, f"../cache/.langchain.db"))) # For evals, comment this out!


def union_metadata(left: List[Metadata], right: List[Metadata]) -> List[Metadata]:
    """
    Merge two lists of metadata documents, removing duplicates based on the title.
    """
    unique_metadata = {m.title: m for m in left + right} # dict for deduplication
    return list(unique_metadata.values())


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    query: Annotated[Union[str, dict], "user question/query, can be string or dict with text and files"]
    potential_metadata: Annotated[List[Metadata], union_metadata] # apparently only used when doing multiple tool calls at same node?
    validation: Annotated[Validation, "Result output from LLM retrieval agent"]
    retriever: Annotated[Retriever, "Retriever object"]
    search_query: Annotated[str, "Current search query being executed"] # for tracking current search

class AgenticRetriever(Retriever):
    """
    Can iteratively search for relevant datasets until it decides that it has found enough datasets or that no datasets are relevant.
    """
    def __init__(self, groupOwner, top_n, embedding_client=None, embedding_model=None, llm_client=None, hybrid_search=False, bm25_search=False):
        if llm_client is None:
            llm_client = get_llm_client("gpt-4.1")

        self.tools = [vector_search, report_results]
        # GPT-3.5-turbo, GPT-o1, and Gemini do not support parallel tool call argument
        # However, only AzureOpenAI have deployment_name attribute
        is_valid_azure_openai = hasattr(llm_client, "deployment_name") and llm_client.deployment_name not in ["gpt-35-turbo", "gpt-o1", "gpt-o3-mini-preview"]
        if is_valid_azure_openai:
            self.llm_with_tools = llm_client.bind_tools(self.tools, parallel_tool_calls=False)
        else:
            self.llm_with_tools = llm_client.bind_tools(self.tools)
        self.graph = self.get_compiled_graph()

        super().__init__(groupOwner, top_n, embedding_client, embedding_model, hybrid_search=hybrid_search, bm25_search=bm25_search)


    def initialize_state(self, state: State):
        # Handle case where query is not directly supplied to graph, but rather through a message
        set_query = False
        query = ""
        human_message_content = None
        
        # Not checking whether query is present means that a human message takes presence over setting the query directly
        if len(state.get("messages", [])) != 0 and isinstance(state["messages"][-1], HumanMessage):
            set_query = True
            query = state["messages"][-1].content
            human_message_content = state["messages"][-1].content
        else:
            query = state["query"]
            
        # Handle multimodal input from Gradio chat interface
        if isinstance(query, dict) and "text" in query:
            text_content = query["text"]
            files = query.get("files", [])
            
            if files:
                # Create multimodal content for HumanMessage
                content_parts = [{"type": "text", "text": text_content}]
                content_parts.extend(handle_attached_files(files))
                human_message_content = content_parts
            else:
                human_message_content = text_content
                
            # Update query to be just the text for downstream processing
            query = text_content
        elif isinstance(query, str):
            human_message_content = query
        else:
            # Fallback for other types
            human_message_content = str(query)
            query = str(query)

        messages = [
            SystemMessage(AGENTIC_RETRIEVER_SYSTEM_PROMPT),
            HumanMessage(content=human_message_content),
        ]

        # Setting validation to None ensures that the agent is routing correctly
        if set_query:
            return {"messages": messages, "query": query, "validation": None, "retriever": self, "potential_metadata": []}
        else:
            return {"messages": messages, "validation": None, "retriever": self, "potential_metadata": []}


    def agent(self, state: State):
        response = self.llm_with_tools.invoke(state["messages"])

        # Token counting
        self.total_tokens += response.usage_metadata["total_tokens"]
        self.total_input_tokens += response.usage_metadata["input_tokens"]
        self.total_output_tokens += response.usage_metadata["output_tokens"]
        if "output_token_details" in response.usage_metadata:
            reasoning_tokens = response.usage_metadata["output_token_details"].get("reasoning", 0)
            self.total_reasoning_tokens += reasoning_tokens
            # Gemini models do not add reasoning tokens to the output tokens, although they are charged at this rate, so let's fix that
            if hasattr(self.llm_with_tools, "model") and self.llm_with_tools.model and "gemini" in self.llm_with_tools.model:
                self.total_output_tokens += reasoning_tokens
        return {"messages": [response]}


    def route_back_from_tools(self, state: State):
        """
        When vector_search tool was called, route back to agent.
        When report_results tool was called, route to the end.
        """
        # We can detect if report tool was called by inspecting whether state's validation field is not None
        if state.get("validation", None) is not None:
            return END
        else:
            return "agent"


    def get_compiled_graph(self):
        """
        Build and return the compiled graph
        """
        # --- Build Graph ---
        tool_node = ToolNode(tools=self.tools)

        graph_builder = StateGraph(State)
        graph_builder.add_node("init", self.initialize_state)
        graph_builder.add_node("agent", self.agent)
        graph_builder.add_node("tools", tool_node)
        graph_builder.add_edge("init", "agent")

        # Adds conditional_edge to route to the ToolNode if the last message has tool calls. Otherwise, route to the end.
        # Routing to end should not happen TODO: investigate how this can be enforced
        graph_builder.add_conditional_edges(
            "agent",
            tools_condition,
        )

        # Adds conditional_edge to route back to the agent if the last tool call was vector_search, or to the end if the last tool call was report_results
        graph_builder.add_conditional_edges(
            "tools",
            self.route_back_from_tools,
            {"agent": "agent", END: END}, # interpret output of route_back_from_tools function
        )

        graph_builder.set_entry_point("init")
        return graph_builder.compile()
    
    def retrieve(self, query: Union[str, dict]) -> Generator[Union[str, Tuple[List[Metadata], str]], None, None]:
        """
        Retrieve the top n most relevant datasets for a user query.
        
        Args:
            query: Either a string query or a dictionary with 'text' and 'files' keys for multimodal input
            
        Yields:
            Either search query strings for intermediate searches or the final result tuple
        """
        events = self.graph.stream(
            {"query": query}, stream_mode="updates"
        )

        metadata = []

        for event in events:
            log.debug(event)

            if "tools" not in event:
                continue

            tools_payload = event["tools"]
            # This allows handling both parallel and single tool calls, as for parallel tool calls, the tools payload is a list of tool calls
            parts = tools_payload if isinstance(tools_payload, list) else [tools_payload]

            for part in parts:
                if "potential_metadata" in part:
                    self.performed_searches += 1
                    metadata = union_metadata(metadata, part["potential_metadata"])
                    
                    # Extract the search query and yield to frontend
                    search_query = part.get("search_query", "Unknown query")
                    yield search_query

                if part.get("messages"):
                    last_msg = part["messages"][-1]
                    if last_msg.name == "report_results":        
                        validation = part["validation"]
                        log.info(f"Agentic Retriever Results: ", valid=validation.valid, valid_datasets=validation.validDatasets)
                        log.info(f"Explanation: {validation.explanation}")
                        # For every dataset in validation, find the corresponding metadata
                        # As the LLM might not perfectly match the dataset titles, we use a fuzzy matching approach
                        relevant_titles = []
                        choices = [m.title.strip() for m in metadata]
                        for dataset_title in validation.validDatasets:
                            # Use fuzzy matching to find the best match for the dataset title
                            best_match = process.extractOne(query=dataset_title, choices=choices, scorer=fuzz.ratio)
                            if best_match:
                                relevant_titles.append(best_match[0])  # best_match is a tuple (title, score, index)

                        # Show token counts
                        log.info("Total Token counts", total_tokens=self.total_tokens, input_tokens=self.total_input_tokens,
                            output_tokens=self.total_output_tokens, reasoning_tokens=self.total_reasoning_tokens)
                        
                        # Yield final result
                        yield [m for m in metadata if m.title.strip() in relevant_titles], validation.explanation
                        return

        log.info("Agentic Retriever: nothing retrieved")
        yield [], "No relevant datasets found."



@tool
def report_results(
    canBeAnswered: Annotated[bool, "Whether dataset(s) can be used to answer question"],
    relevant_datasets: Annotated[List[str], "List of dataset titles that can be used to answer question"],
    explanation: Annotated[str, "Brief explanation of how dataset(s) can be used to answer question"],
    tool_call_id: Annotated[str, InjectedToolCallId] # not filled out by LLM
) -> Annotated[Validation, "Object to report search results in a structured way to the user"]:
    """Report the results of the search to the user."""
    # Update state with result/validation output
    return Command(
        update={
            "messages": [ToolMessage("Success", tool_call_id=tool_call_id)],
            "validation": Validation(valid=canBeAnswered, explanation=explanation, validDatasets=relevant_datasets),
        }
    )


@tool
def vector_search(
    retriever: Annotated[Retriever, InjectedState("retriever")], # not filled out by LLM
    tool_call_id: Annotated[str, InjectedToolCallId], # not filled out by LLM
    query: Annotated[str, "query in German"],
) -> Annotated[List[dict], "List of metadata of most relevant datasets"]:
    """Retrieve the metadata of the most relevant open government datasets for the given query."""
    metadata_docs = retriever.get_top_n_similar_docs(query, retriever.top_n)
    
    # Appends metadata documents that are potentially relevant to the state

    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=(
                        "Potentially relevant datasets:\n\n"
                        + "\n\n".join(str(m) for m in metadata_docs)
                    ),
                    name="vector_search",
                    tool_call_id=tool_call_id
                ),
                ],
            "potential_metadata": metadata_docs,
            "search_query": query,
        }
    )
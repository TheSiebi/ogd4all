import os
import sys
from typing_extensions import TypedDict, Annotated, List, Tuple, Generator
from typing import Optional
from pydantic import BaseModel, Field
import structlog;log=structlog.get_logger()
from rapidfuzz import process, fuzz

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # not the nicest way of handling this, but oh well...
from retrieval.retriever import Retriever, Metadata
from utils import get_llm_client

script_dir = os.path.dirname(os.path.abspath(__file__))
set_llm_cache(SQLiteCache(database_path=os.path.join(script_dir, f"../cache/.langchain.db"))) # For evals, comment this out!


class Validation(BaseModel):
    """Validation output from metadata"""
    valid: bool = Field(..., description="Whether dataset(s) can be used to answer question")
    explanation: str = Field(..., description="Explanation of how dataset(s) can be used to answer question")
    validDatasets: Optional[List[str]] = Field(..., description="List of datasets that can be used to answer question")


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    query: Annotated[str, "user question/query"]
    metadata: Annotated[List[Metadata], "List of metadata of most relevant datasets"]
    validation: Annotated[Validation, "Validation output from verifier"]


class VerifiedRetriever(Retriever):
    """
    Retrieves documents using a KNN approach, then verifies whether the retrieved documents can be used to answer the question.
    The verification is done using a language model that checks the metadata of the retrieved documents.
    """
    def __init__(self, groupOwner, top_n, embedding_client=None, embedding_model=None, llm_client=None, hybrid_search=False, bm25_search=False):
        if llm_client is None:
            llm_client = get_llm_client("gpt-4.1")

        self.llm_verifier = llm_client.with_structured_output(Validation, include_raw=True)
        self.graph = self.get_compiled_graph()

        super().__init__(groupOwner, top_n, embedding_client, embedding_model, hybrid_search=hybrid_search, bm25_search=bm25_search)

    def vector_search(self, state: State):
        """
        Retrieve the metadata of the most relevant open government datasets for answering a user's question.
        """
        # Handle case where query is not directly supplied to graph, but rather through a message
        set_query = False
        query = ""
        self.performed_searches += 1
        # Not checking whether query is present means that a human message takes presence over setting the query directly
        if len(state.get("messages", [])) != 0 and isinstance(state["messages"][-1], HumanMessage):
            set_query = True
            query = state["messages"][-1].content
        else:
            query = state["query"]

        metadata_docs = super().get_top_n_similar_docs(query, self.top_n)

        if set_query:
            return {"metadata": metadata_docs, "query": query}
        else:
            return {"metadata": metadata_docs}
        

    def llm_verificator(self, state: State):
        """
        Given user question and dataset, verify whether dataset can be used to answer question and produce answer in structured output
        """
        messages = [
            SystemMessage("""You will be given a user question and the metadata of multiple open datasets that are most relevant to the question.
                        Please verify whether any or multiple of these datasets can be used to answer the user's question, and if so, briefly explain how.
                        If there are datasets with the same data, please only use the most relevant one, typically the most recent one."""),
            HumanMessage(f"Question: {state['query']}\nMetadata of the most relevant datasets: {"\n\n".join(str(m) for m in state['metadata'])}"),
        ]

        response = self.llm_verifier.invoke(messages)

        # Token counting
        self.total_tokens += response["raw"].usage_metadata["total_tokens"]
        self.total_input_tokens += response["raw"].usage_metadata["input_tokens"]
        self.total_output_tokens += response["raw"].usage_metadata["output_tokens"]
        if "output_token_details" in response["raw"].usage_metadata:
            reasoning_tokens = response["raw"].usage_metadata["output_token_details"].get("reasoning", 0)
            self.total_reasoning_tokens += reasoning_tokens
            # Gemini models do not add reasoning tokens to the output tokens, although they are charged at this rate, so let's fix that
            if hasattr(self.llm_verifier, "model") and "gemini" in self.coding_client.model:
                self.total_output_tokens += reasoning_tokens

        validation = response["parsed"]

        # Filter metadata in state based on validation
        valid_metadata = [] 

        if validation.valid and validation.validDatasets:
            messages.append(AIMessage(f"**Valid datasets**: {', '.join(validation.validDatasets)}\n**Explanation**: {validation.explanation}"))

            # For every dataset in validation, find the corresponding metadata
            # As the LLM might not perfectly match the dataset titles, we use a fuzzy matching approach
            relevant_titles = []
            choices = [m.title.strip() for m in state["metadata"]]
            for dataset_title in validation.validDatasets:
                # Use fuzzy matching to find the best match for the dataset title
                best_match = process.extractOne(query=dataset_title, choices=choices, scorer=fuzz.ratio)
                if best_match:
                    relevant_titles.append(best_match[0])  # best_match is a tuple (title, score, index)
            valid_metadata = [m for m in state["metadata"] if m.title.strip() in relevant_titles]
        else:
            messages.append(AIMessage(f"Unfortunately, the available open data cannot be used to answer this question."))


        return {
            "messages": messages,
            "validation": validation,
            "metadata": valid_metadata,
        }


    def get_compiled_graph(self):
        """
        Build and return the compiled (Lang)graph
        """
        # Build the graph
        graph_builder = StateGraph(State)
        graph_builder.add_node("vector_search", self.vector_search)
        graph_builder.add_node("llm_verificator", self.llm_verificator)
        graph_builder.set_entry_point("vector_search")
        graph_builder.add_edge("vector_search", "llm_verificator")
        graph_builder.set_finish_point("llm_verificator")
        return graph_builder.compile()
    

    def retrieve(self, query: str) -> Generator[Tuple[List[Metadata], str], None, None]:
        events = self.graph.stream(
            {"query": query}, stream_mode="updates"
        )
        
        for event in events:
            if "llm_verificator" in event:
                validation = event["llm_verificator"]["validation"]
                log.info(f"Verifier Explanation: {validation.explanation}")
                yield event["llm_verificator"]["metadata"], validation.explanation # llm_verificator node takes care of filtering

        log.info("Verified Retriever: nothing retrieved")
        yield [], "No relevant datasets found."

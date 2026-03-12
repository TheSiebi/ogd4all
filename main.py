import gradio as gr
import argparse
import time
import os
import sys
import logging
import structlog;log=structlog.get_logger()
import folium
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
import copy
import utils
import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # not the nicest way of handling this, but oh well...
from retrieval.knn_retriever import KNNRetriever
from retrieval.verified_retriever import VerifiedRetriever
from retrieval.agentic_retriever import AgenticRetriever
from retrieval.retriever import Retriever

from generation.simple_analyzer import SimpleAnalyzer
from generation.simple_local_analyzer_v2 import SimpleLocalAnalyzerV2
from generation.iterative_local_analyzer import IterativeLocalAnalyzer

from utils import init_mappings, get_llm_client, SUPPORTED_LLMS

class RetrievalCheck(BaseModel):
    """Results of whether an additional dataset retrieval is required"""
    thought: str = Field(..., description="Short explanation whether additional retrieval is required.")
    retrievalRequired: bool = Field(..., description="Whether an additional dataset retrieval is required or not.")

class OGD4All():
    def __init__(self, groupOwner, retriever, analyzer_type, coding_llm, retrieval_check_client, streaming, timeout=60):
        self.groupOwner = groupOwner
        self.retriever = retriever
        self.analyzer_type = analyzer_type.lower()
        self.analyzer = None
        self.reset = True
        self.timeout = timeout
        self.map_component = None
        self.added_datasets = False
        self.coding_llm = coding_llm
        self.streaming = streaming
        self.retrieval_check_client = retrieval_check_client.with_structured_output(RetrievalCheck)


    def chat_fn(self, query, history):
        """
        Function to handle the chat interaction with the user.
        It takes a query and history of messages, processes them, and returns the response.

        :param retriever: The retriever object to fetch relevant metadata documents.
        :param analyzer: The analyzer object to analyze the data based on the query.
        :param query: The user query.
        :param history: The history of messages exchanged in the chat. Currently ignored, as the Analyzer keeps track of the context
        :return: The response to the user's query.
        """
        try:
            updated_map = gr.update()
            if self.reset:
                # Kill previous sandbox
                if self.analyzer is not None:
                    self.analyzer.finalize()
                    self.analyzer = None


                start_time = time.time()
                
                # We start by retrieving the relevant metadata documents based on the user's query.
                thought_msg_retrieval = gr.ChatMessage(
                    role="assistant",
                    content="",
                    metadata={"title": "Retrieving relevant open data...",
                            "status": "pending"},
                )

                yield thought_msg_retrieval, updated_map

                try:
                    search_count = 0
                    
                    for result in self.retriever.retrieve(query):
                        if isinstance(result, str):
                            # Intermediate search query string
                            search_count += 1
                            thought_msg_retrieval.content += f"**Query {search_count}**: {result}\n"
                            yield thought_msg_retrieval, updated_map
                        else:
                            # Final result tuple
                            metadata_docs, explanation = result
                            break
                        
                except Exception as e:
                    log.error("Error during dataset retrieval:", exc_info=True)
                    thought_msg_retrieval.content = "An error occurred while retrieving datasets."
                    thought_msg_retrieval.metadata["status"] = "done"
                    thought_msg_retrieval.metadata["title"] = "Retrieval failed"
                    error_msg = "I'm sorry, an error occurred during the retrieval of relevant datasets. Please try again."
                    self.reset = True
                    yield [thought_msg_retrieval, error_msg], updated_map
                    return
                thought_msg_retrieval.metadata["status"] = "done"
                thought_msg_retrieval.metadata["title"] = "Retrieval completed"
                thought_msg_retrieval.metadata["duration"] = time.time() - start_time

                if len(metadata_docs) == 0:
                    thought_msg_retrieval.content += "\nNo suitable datasets found."
                    yield [thought_msg_retrieval, explanation], updated_map
                    return

                thought_msg_retrieval.content += f"\nBased on datasets retrieved with the above {"query" if search_count == 1 else "queries"}, I will be using the following dataset{"" if len(metadata_docs) == 1 else "s"} to answer your question:\n"
                thought_msg_retrieval.content += "\n".join([f"- [{doc.title}]({doc.downloadURL})" for doc in metadata_docs])
                #thought_msg_retrieval.content += f"\n\n{explanation}" # Explanation does not seem very helpful and is currently not optimized for display towards the user

                start_time = time.time()
                thought_msg_coding_init = gr.ChatMessage(
                    role="assistant",
                    content="",
                    metadata={"title": "Initializing coding environment...",
                              "status": "pending"},
                )

                yield [thought_msg_retrieval, thought_msg_coding_init], updated_map

                # Initialize the analyzer based on the specified type
                if self.analyzer_type == "simple":
                    self.analyzer = SimpleAnalyzer(retriever.groupOwner, metadata_docs, timeout=self.timeout, coding_client=self.coding_llm)
                elif self.analyzer_type == "simple_local_v2":
                    self.analyzer = SimpleLocalAnalyzerV2(retriever.groupOwner, metadata_docs, coding_client=self.coding_llm, streaming=self.streaming)
                elif self.analyzer_type == "iterative_local":
                    self.analyzer = IterativeLocalAnalyzer(retriever.groupOwner, metadata_docs, coding_client=self.coding_llm, streaming=self.streaming)
                else:
                    log.error(f"Unknown analyzer type: {self.analyzer_type}. Exiting...")
                    sys.exit(1)

                thought_msg_coding_init.metadata["status"] = "done"
                thought_msg_coding_init.metadata["title"] = "Coding environment initialized"
                thought_msg_coding_init.metadata["duration"] = time.time() - start_time
                thought_msg_coding_init.content += f"I have initialized a persistent Python environment that allows me to analyze the user's question. "
                thought_msg_coding_init.content += f"I have loaded all required datasets and imported required libraries. "
                thought_msg_coding_init.content += f"The following code has been executed to print context about the datasets:\n```python{self.analyzer.setup_code}\n```"

                yield [thought_msg_retrieval, thought_msg_coding_init], updated_map
            else:
                # Run a check to see whether new datasets need to be retrieved
                prompt = f"""
                Are additional datasets required to answer the following question? If so, please provide a short explanation of why they are needed.
                You currently have access to the following datasets:
                {", ".join([str(doc) for doc in self.analyzer.metadata_docs])}

                Question: {query}
                """
                log.debug(f"Retrieval check prompt: {prompt}")
                messages = self.analyzer.messages.copy()
                messages.append(HumanMessage(prompt))
                try:
                    retrieval_check = self.retrieval_check_client.invoke(messages)
                except Exception as e:
                    log.error("Error during retrieval check:", exc_info=True)
                    yield "An error occurred while checking whether additional datasets are required.", updated_map
                    return

                if retrieval_check.retrievalRequired:
                    thought_msg_retrieval = gr.ChatMessage(
                        role="assistant",
                        content="",
                        metadata={"title": "Retrieving additional datasets...",
                                  "status": "pending"},
                    )
                    yield thought_msg_retrieval, updated_map

                    # Retrieve additional datasets
                    try:
                        search_count = 0
                        
                        for result in self.retriever.retrieve(query):
                            if isinstance(result, str):
                                # Intermediate search query string
                                search_count += 1
                                thought_msg_retrieval.content += f"**Query {search_count}**: {result}\n"
                                yield thought_msg_retrieval, updated_map
                            else:
                                # Final result tuple
                                metadata_docs, explanation = result
                                break
                                
                        if metadata_docs is None:
                            # Something went wrong, no final result was yielded
                            raise Exception("No final result received from retriever")
                            
                    except Exception as e:
                        log.error("Error during additional dataset retrieval:", exc_info=True)
                        thought_msg_retrieval.content = "An error occurred while retrieving additional datasets."
                        thought_msg_retrieval.metadata["status"] = "done"
                        thought_msg_retrieval.metadata["title"] = "Retrieval failed"
                        error_msg = "I'm sorry, an error occurred during the retrieval of relevant datasets. Please try again."
                        yield [thought_msg_retrieval, error_msg], updated_map
                        return
                    thought_msg_retrieval.metadata["status"] = "done"
                    thought_msg_retrieval.metadata["title"] = "Retrieval completed"

                    # Filter out already existing metadata documents based on title
                    existing_titles = {doc.title for doc in self.analyzer.metadata_docs}
                    extra_docs = [doc for doc in metadata_docs if doc.title not in existing_titles]

                    if len(extra_docs) == 0 and len(metadata_docs) == 0:
                        # Nothing relevant found
                        thought_msg_retrieval.content += "\nNo suitable additional datasets found. "
                        yield [thought_msg_retrieval, explanation], updated_map
                        return
                    elif len(extra_docs) == 0:
                        # Only existing datasets were found. Difference to previous case is that we still let analyzer run.
                        thought_msg_retrieval.content += "\nNo suitable additional datasets found. "
                        yield [thought_msg_retrieval, explanation], updated_map
                    else:
                        self.added_datasets = True
                        thought_msg_retrieval.content += f"\nBased on datasets retrieved with the above {"query" if search_count == 1 else "queries"}, I will be using the following additional dataset{"" if len(extra_docs) == 1 else "s"} to answer your question:\n"
                        thought_msg_retrieval.content += "\n".join([f"- [{doc.title}]({doc.downloadURL})" for doc in extra_docs])

                        yield thought_msg_retrieval, updated_map

                        start_time = time.time()
                        thought_msg_coding_extend = gr.ChatMessage(
                            role="assistant",
                            content="",
                            metadata={"title": "Updating coding environment...",
                                    "status": "pending"},
                        )

                        if isinstance(self.analyzer, SimpleLocalAnalyzerV2) or isinstance(self.analyzer, IterativeLocalAnalyzer):
                            yield [thought_msg_retrieval, thought_msg_coding_extend], updated_map 

                            self.analyzer.metadata_docs.extend(extra_docs)
                            self.analyzer.extend_sandbox([m.title for m in extra_docs])

                            thought_msg_coding_extend.metadata["status"] = "done"
                            thought_msg_coding_extend.metadata["title"] = "Coding environment updated"
                            thought_msg_coding_extend.metadata["duration"] = time.time() - start_time
                            thought_msg_coding_extend.content += f"I have updated the persistent Python environment with new datasets. "
                            thought_msg_coding_extend.content += f"The following code has been executed to print context about the new datasets:\n```python{self.analyzer.setup_code}\n```"
                            yield [thought_msg_retrieval, thought_msg_coding_extend], updated_map
                        else:
                            thought_msg_coding_extend.metadata["status"] = "done"
                            thought_msg_coding_extend.metadata["title"] = "Coding environment not updated"
                            thought_msg_coding_extend.metadata["duration"] = time.time() - start_time
                            thought_msg_coding_extend.content = "I am not able to extend the sandbox with additional datasets with the current analyzer type. Please reset the system to start a new analysis."
                            yield [thought_msg_retrieval, thought_msg_coding_extend], updated_map


            for out in self.analyzer.analyze(query):
                if not isinstance(out, list):
                    out = [out]

                produced_new_map = False
                new_map = None
                for i, item in enumerate(out[:]): # iterate over copy
                    if isinstance(item, folium.Map):
                        new_map = item
                        out.pop(i)
                        produced_new_map = True

                if produced_new_map and self.map_component is not None:
                    copied_map = copy.deepcopy(new_map)
                    folium.LayerControl().add_to(copied_map)
                    updated_map = gr.update(value=copied_map._repr_html_())

                if self.reset:
                    yield [thought_msg_retrieval, thought_msg_coding_init] + out, updated_map
                elif self.added_datasets:
                    yield [thought_msg_retrieval, thought_msg_coding_extend] + out, updated_map
                elif retrieval_check is not None and retrieval_check.retrievalRequired:
                    # Retrieval was performed, but no new datasets were added
                    yield [thought_msg_retrieval] + out, updated_map
                else:
                    yield out, updated_map

            self.added_datasets = False
            self.reset = False

        except Exception as e:
            log.error("Caught an exception in chat_fn: %s", e, exc_info=True, backtrace=True, diagnose=True)
            self.finalize()
            yield gr.ChatMessage(role="assistant", content="I am sorry, there has been an error processing your request. Please try again."), updated_map
            self.reset = True
            return
    

    def finalize(self):
        log.info("Finalizing OGD4All...")
        if self.analyzer is not None:
            self.analyzer.finalize()


def start_frontend(retriever: Retriever, analyzer_type: str, coding_llm, retrieval_check_client, streaming: bool = True):
    """Starts an interactive Gradio interface for OGD4All"""
    log.info("Starting OGD4All...")
    ogd4all = OGD4All(retriever.groupOwner, retriever, analyzer_type, coding_llm, retrieval_check_client, streaming, timeout=360)
    
    # Define custom CSS that will make the elements span full viewport height.
    custom_css = """
    .full-height {
        height: 80vh;
    }

    /* Map panel: propagate height through Gradio's wrapper divs */
    .map-panel {
        height: 80vh;
    }

    .map-panel .prose,
    .map-panel > div {
        height: 100% !important;
        max-height: 100% !important;
    }

    /* Override folium's padding-bottom aspect-ratio trick */
    .map-panel div[style*="padding-bottom"] {
        padding-bottom: 0 !important;
        height: 100% !important;
    }

    /* Hide folium's Jupyter notebook trust fallback message */
    .map-panel div[style*="padding-bottom"] > span {
        display: none;
    }

    .map-panel iframe {
        height: 100% !important;
        position: relative !important;
    }

    @media (max-width: 768px) {
        /* Allow the page to scroll instead of being clipped to viewport */
        html, body {
            height: auto !important;
            overflow-y: auto !important;
        }

        .gradio-container,
        .gradio-container > .main,
        .gradio-container > .main > .wrap {
            height: auto !important;
            min-height: 100vh;
            overflow-y: visible !important;
            overflow-x: hidden !important;
        }

        /* Remove fixed height from row/columns so stacked layout flows naturally */
        .full-height {
            height: auto !important;
            min-height: unset !important;
        }

        /* Map panel: fixed height that contributes to (scrollable) page length */
        .map-panel {
            height: 50vh;
            min-height: 300px;
        }
    }
    """


    map_placeholder = """
    <style>
        .map-placeholder {
            width: 100%;
            height: 100%;
            min-height: 300px;
            background: #ffffff;
            border-radius: 12px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 14px;
            border: 1px solid #e0e0e0;
            box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        }
        .map-placeholder svg { opacity: 0.25; }
        .map-placeholder span { font-size: 15px; color: #bbb; letter-spacing: 0.01em; }
        @media (prefers-color-scheme: dark) {
            .map-placeholder {
                background: rgba(255,255,255,0.06);
                border-color: rgba(255,255,255,0.1);
            }
            .map-placeholder svg { opacity: 0.3; stroke: #aaa; }
            .map-placeholder span { color: #666; }
        }
    </style>
    <div class="map-placeholder">
        <!-- Map icon -->
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M9 4L3 7v13l6-3 6 3 6-3V4l-6 3-6-3z" stroke="#555" stroke-width="1.5" stroke-linejoin="round"/>
            <line x1="9" y1="4" x2="9" y2="17" stroke="#555" stroke-width="1.5"/>
            <line x1="15" y1="7" x2="15" y2="20" stroke="#555" stroke-width="1.5"/>
        </svg>
        <span>OGD4All automatically generates interactive maps when relevant</span>
    </div>
    """

    with gr.Blocks(title="OGD4All", fill_height=True) as demo:
        map = gr.HTML(value=map_placeholder, render=False, elem_classes="map-panel")
        ogd4all.map_component = map
        with gr.Row(scale=1):
            with gr.Column(scale=1):
                gr.HTML("<center><h1><b>OGD4All</b>: Accessible Retrieval & Analysis of Geospatial Open Government Data</h1></center>")
        with gr.Row(elem_classes="full-height", scale=4):
            with gr.Column(scale=1, elem_classes="full-height"):
                chatbot = gr.Chatbot(scale=1, show_label=False)

                def clear_all():
                    ogd4all.reset = True # reset analyzer state
                    return gr.update(value=map_placeholder) # restore placeholder

                chatbot.clear(fn=clear_all, inputs=[], outputs=[map])

                gr.ChatInterface(
                    fn=ogd4all.chat_fn,
                    multimodal=False, # we manually handle multimodal input
                    textbox=gr.MultimodalTextbox(file_types=["image", ".pdf"], placeholder="Type a question...", file_count='multiple'),
                    examples=[
                        "Wo plant die Stadt Zürich, neue Bäume zu pflanzen?",
                        "Welche Hundefreilaufzone ist am nächsten zum Kunsthaus Zürich?",
                        "In welchem Quartier der Stadt Zürich hat es die höchste Dichte an Spielplätzen?",
                        "Wie hat sich die Anzahl Elektroautos in Zürich in den letzten 20 Jahren entwickelt?"
                    ],
                    chatbot=chatbot,
                    additional_outputs=[map],
                )
            with gr.Column(scale=1, elem_classes="full-height"):
                map.render()

    demo.launch(theme=gr.themes.Soft(font=[gr.themes.GoogleFont("Roboto"), "Arial", "sans-serif"]), css=custom_css)


if __name__ == "__main__":
    print(utils.CONSOLE_LOGO)

    parser = argparse.ArgumentParser(description="Chat with Geospatial Open Government Data.")
    parser.add_argument("--groupOwner", type=str, default="50000006", help="The groupOwner id whose metadata should be queried (default: 50000006).")
    parser.add_argument("--top_n", type=int, default=10, help="The number of documents to retrieve for a single KNN search (default: 10).")
    parser.add_argument("--retriever", type=str, choices=["agentic", "knn", "verified"], default="agentic", help="The retrieval strategy to use")
    parser.add_argument("--analyzer", type=str, choices=["simple_local_v2", "simple", "simple_local", "iterative_local"], default="iterative_local", help="The analyzer type to use")
    parser.add_argument("--retrieval_llm", choices=SUPPORTED_LLMS, default='gpt-4.1', help="The LLM to use for retrieval tasks.")
    parser.add_argument("--retrieval_check_llm", choices=SUPPORTED_LLMS, default='gpt-4o', help="The LLM to use when checking whether a follow-up retrieval is needed. Ideally quite fast.")
    parser.add_argument("--coding_llm", choices=[llm for llm in SUPPORTED_LLMS], default='gpt-4.1', help="The LLM to use for coding tasks/analysis.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--hybrid_search", action="store_true", help="Enable hybrid search with Milvus.")
    group.add_argument("--bm25_search", action="store_true", help="Enable BM25 search with Milvus.")
    parser.add_argument("--no_streaming", action="store_true", help="Disable streaming for the coding LLM. This enables validation of LLM responses and token counting, but makes the system feel less responsive.")
    args = parser.parse_args()
    
    log_name = f"main_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.retriever}_{args.analyzer}"
    utils.setup_logging(level=logging.INFO, log_filename=f"{log_name}.log")

    # Initialize utils
    init_mappings(args.groupOwner)

    # Initialize the retriever
    retriever = None
    if args.retriever.lower() == "verified":
        retriever = VerifiedRetriever(args.groupOwner, args.top_n, hybrid_search=args.hybrid_search, llm_client=get_llm_client(args.retrieval_llm), bm25_search=args.bm25_search)
    elif args.retriever.lower() == "knn":
        retriever = KNNRetriever(args.groupOwner, args.top_n, hybrid_search=args.hybrid_search, llm_client=get_llm_client(args.retrieval_llm), bm25_search=args.bm25_search)
    elif args.retriever.lower() == "agentic":
        retriever = AgenticRetriever(args.groupOwner, args.top_n, hybrid_search=args.hybrid_search, llm_client=get_llm_client(args.retrieval_llm), bm25_search=args.bm25_search)
    else:
        log.error(f"Unknown retriever type: {args.retriever}. Exiting...")
        sys.exit(1)

    start_frontend(retriever, args.analyzer, coding_llm=get_llm_client(args.coding_llm), retrieval_check_client=get_llm_client(args.retrieval_check_llm), streaming=not args.no_streaming)
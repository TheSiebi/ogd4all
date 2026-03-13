from typing_extensions import List, Any, Union
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from smolagents.local_python_executor import LocalPythonExecutor, fix_final_answer_code

import os
import sys
import gradio as gr
import time
import structlog;log=structlog.get_logger()
import textwrap
import folium
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
from geopy.geocoders import Nominatim, GoogleV3

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # not the nicest way of handling this, but oh well...
from retrieval.retriever import Metadata
from generation.analyzer import Analyzer, CodeAction, CodeAct
from utils import get_path_from_title, get_file_from_title, generate_iterative_system_prompt, clean, handle_attached_files


def setup_code_for_file(name: str, file_path: str, idx: int) -> str:
    """
    Generate code which shows layer and column information for a given file, if it is a geopackage dataset.
    For CSV data, it will just show column information.
    :param name: The name of the dataset.
    :param file_path: The path to the dataset.
    :param idx: The index of the dataset in the list of datasets.
    :return: The generated code as a string.
    """

    if file_path.endswith(".gpkg"):
        return textwrap.dedent(f"""
        path_{idx} = "{file_path}"
        layer_names = gpd.list_layers(path_{idx})['name'].to_list()
        print(f"Layers in dataset {name}: {{', '.join(layer_names)}}")
        
        for layer in layer_names:
            try:
                gdf_tmp = gpd.read_file(path_{idx}, layer=layer)
                print(f"**Layer {{layer}}**")
                print(f"{{gdf_tmp.shape[0]}} rows and {{gdf_tmp.shape[1]}} columns")
                print("Example data:")
                pd.set_option('display.max_columns', None)
                pd.set_option('display.expand_frame_repr', False)
                print(gdf_tmp.head())
                pd.reset_option('display.max_columns')
                pd.reset_option('display.expand_frame_repr')

                # Iterate over every attribute, get unique values and if number of unique values is small, print them, otherwise just print number and examples
                for col in gdf_tmp.columns:
                    col_type = gdf_tmp[col].dtype
                    num_unique = gdf_tmp[col].nunique()
                    try:
                        if col == 'geometry':
                            geom_type = gdf_tmp[col].geom_type.unique()
                            print(f"Column {{col}}: Type {{', '.join(geom_type)}}, Unique Values: {{num_unique}}")
                        elif num_unique <= 10:
                            print(f"Column {{col}}: Type {{col_type}}, Unique Values: {{', '.join(map(str, gdf_tmp[col].unique()))}}")
                        else:
                            print(f"Column {{col}}: Type {{col_type}}, Unique Values: {{gdf_tmp[col].nunique()}} in total, examples are {{', '.join(map(str, gdf_tmp[col].unique()[:5]))}}...")
                    except Exception as e:
                        print(f"Error while processing column {{col}}")
                        print(f"Column {{col}}: Type {{col_type}}, Unique Values: {{num_unique}}")
                        continue
            except Exception as e:
                print(f"Error reading layer {{layer}}: {{e}}")
        """)
    elif file_path.endswith(".csv"):
        varName = f"df_{clean(name)[:16]}"
        return textwrap.dedent(f"""
        path_{idx} = "{file_path}"
        {varName} = pd.read_csv(path_{idx})
        print(f"Dataset {name} has {{{varName}.shape[0]}} rows and {{{varName}.shape[1]}} columns")
        print("Example data:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.expand_frame_repr', False)
        print({varName}.head())
        pd.reset_option('display.max_columns')
        pd.reset_option('display.expand_frame_repr')

        # Iterate over every column, get unique values and if number of unique values is small, print them, otherwise just print number and examples
        for col in {varName}.columns:
            col_type = {varName}[col].dtype
            num_unique = {varName}[col].nunique()
            try:
                if num_unique <= 10:
                    print(f"Column {{col}}: Type {{col_type}}, Unique Values: {{', '.join(map(str, {varName}[col].unique()))}}")
                else:
                    print(f"Column {{col}}: Type {{col_type}}, Unique Values: {{{varName}[col].nunique()}} in total, examples are {{', '.join(map(str, {varName}[col].unique()[:5]))}}...")
            except Exception as e:
                print(f"Error while processing column {{col}}")
                print(f"Column {{col}}: Type {{col_type}}, Unique Values: {{num_unique}}")
                continue
        """)
    else:
        log.warning(f"Unsupported file type for {name}: {file_path}. Only .gpkg and .csv files are supported for now.")
        return ""

geolocator = GoogleV3(api_key=os.environ["GOOGLE_GEOCODING_API_KEY"]) if "GOOGLE_GEOCODING_API_KEY" in os.environ else Nominatim(user_agent="ogd4all")

def geocode(query: str):
    return geolocator.geocode(query)

class IterativeLocalAnalyzer(Analyzer):
    """
    Runs AI-generated code using smolagent's LocalPythonExecutor to analyze the data.
    Is able to run multiple analysis steps iteratively, until it concludes that the analysis is complete or reaches the maximum number of steps.
    """
    def __init__(self, groupOwner, metadata_docs: List[Metadata], coding_client=None, max_steps=20, streaming=False):
        super().__init__(groupOwner, metadata_docs, coding_client)

        if streaming:
            # Including raw is not possible with streaming
            # Also, we can't use CodeAct as it will require validation, which is not possible with streaming
            # Thus, we rely on the non-validated CodeAction
            self.coding_llm = self.coding_client.with_structured_output(CodeAction)
        else:
            self.coding_llm = self.coding_client.with_structured_output(CodeAct, include_raw=True)
        
        self.python_executor = LocalPythonExecutor(
            additional_authorized_imports=[
                "geopandas",
                "folium",
                "pandas",
                "numpy",
                "contextily",
                "geopy",
                "geopy.geocoders",
                "geopy.geocoders.Nominatim",
                "matplotlib",
                "matplotlib.pyplot",
                "matplotlib.cm",
                "shapely",
                "seaborn",
                "pyproj"
            ]
        )
        self.python_executor.send_tools({
            "geocode": geocode,
            "final_answer": self.final_answer,
            }) # could add more tools here

        self.max_steps = max_steps
        self.streaming = streaming
        self.current_final_answer = None
        self.setup_code, logs, file_names = self.init_sandbox()
        self.init_conversation(self.setup_code, logs, file_names)
        self.dataset_idx_offset += len(file_names)


    def finalize(self):
        log.info("Killing the sandbox...")


    def final_answer(self, answer):
        self.current_final_answer = answer


    def run_ai_generated_code(self, ai_generated_code: str) -> tuple[bool, List[Any], bool]:
        """
        Run the AI-generated code in the sandbox and return the results.
        
        :param ai_generated_code: The AI-generated code to run.
        :return: A tuple containing a boolean indicating success and a list of results.
        """

        # Ensure no variable is named final_answer
        ai_generated_code = fix_final_answer_code(ai_generated_code)

        log.info("Running code in sandbox...")
        log.info("AI-generated code:\n%s", ai_generated_code)
        try:
            vars, logs, is_final_answer = self.python_executor(ai_generated_code)
            log.info('Code execution finished!')

            if is_final_answer:
                if not isinstance(self.current_final_answer, list):
                    self.current_final_answer = [self.current_final_answer]

                # Process answers into suitable format for gradio
                # First join all list elements of type string
                textual_answers = [a for a in self.current_final_answer if isinstance(a, str)]
                textual_answers = "\n".join(textual_answers)

                gradio_elems = []
                for elem in self.current_final_answer:
                    if isinstance(elem, folium.Map):
                        gradio_elems.append(elem) # we wrap with gradio_folium in main.py, to avoid discarded updates
                        # There are some errors which happen only when the map is rendered, so we force a render here
                        try:
                            elem.get_root().render()
                        except Exception as e:
                            return False, [f"Error rendering folium map. Reinstantiate the map, then fix this error: {e}"], False
                    elif isinstance(elem, matplotlib.figure.Figure):
                        elem.canvas.draw() # Ensure the figure is rendered
                        img = np.asarray(elem.canvas.buffer_rgba())
                        pil_img = Image.fromarray(img)
                        gradio_elems.append(gr.Image(value=pil_img, type="pil", label="Matplotlib Figure"))
                    elif isinstance(elem, pd.DataFrame):
                        textual_answers += f"\n\n{elem.to_markdown()}"
                    # else:
                    #     try:
                    #         textual_answers += f"\n{elem}"
                    #     except Exception as e:
                    #         log.error('Error processing element of type %s: %s', type(elem), e)
                return True, [textual_answers] + gradio_elems, is_final_answer
            else:
                # Make vars a list if not already
                if not isinstance(vars, list):
                    vars = [vars]

                # Get all textual elements from vars and pandas dataframes as markdown
                textual_vars = []
                for var in vars:
                    if isinstance(var, str):
                        textual_vars.append(var)
                    elif isinstance(var, pd.DataFrame):
                        textual_vars.append(var.to_markdown())
                    elif var is not None:
                        textual_vars.append(str(var))

                # Return textual_vars and logs
                return True, [logs] + textual_vars, is_final_answer


        except Exception as e:
            log.error('AI-generated code had an error.')
            log.error(e)
            return False, [str(e)], False


    def init_sandbox(self) -> tuple[str, List[Any], List[str]]:
        """
        Initialize the interpreter by loading datasets and running setup code.

        :return: A tuple containing the setup code, the logs from the setup code, and the names of the loaded datasets.
        """

        try:
            file_names = [m.title for m in self.metadata_docs]
            cleaned_file_names = [get_file_from_title(self.groupOwner, file_name) for file_name in file_names]
            file_paths = [get_path_from_title(self.groupOwner, file_name) for file_name in file_names]

            # Combine file names and paths into a single list of tuples
            files = list(zip(file_names, file_paths))

            setup_code = textwrap.dedent(f"""
            import geopandas as gpd
            import folium
            import pandas as pd
            import matplotlib.pyplot as plt
            import contextily as cx

            # Initialize an interactive folium map
            m = folium.Map(zoom_start=13, control_scale=True, location=[47.3769, 8.5417], zoom_control="bottomleft", min_zoom=12, attributionControl=False)
            # Add base layers
            folium.TileLayer('Esri.WorldImagery', name='Satellite', control=True).add_to(m)
            folium.TileLayer('cartodb positron', name='CartoDB Positron', control=True).add_to(m)
            """)

            # Initial code for LLM to understand data:
            # - For every dataset, print layer names
            for idx, (file_name, file_path) in enumerate(files):
                name = file_name.split(".")[0]  # remove suffix
                setup_code += setup_code_for_file(name, file_path, idx)

            # Run code on sandbox
            success, logs, _ = self.run_ai_generated_code(setup_code)
            if not success:
                log.error("Caught an exception in setup code: %s", exc_info=True, backtrace=True, diagnose=True)
                self.finalize()
                sys.exit(1)
            else:
                log.info("Setup code ran successfully!")
                log.debug(logs)


            return setup_code, logs, cleaned_file_names
        except Exception as e:
            log.error("Caught an exception in sandbox init: %s", e, exc_info=True, backtrace=True, diagnose=True)
            self.finalize()
            sys.exit(1)
    
    
    def init_conversation(self, setup_code, logs, file_names):
        """
        Initialize the conversation with the analyzer

        :param logs: The logs from the setup code.
        :param file_names: The names of the uploaded datasets.
        """

        system_prompt = generate_iterative_system_prompt(file_names, self.metadata_docs, setup_code, logs)

        log.debug("System prompt generated")
        log.debug(system_prompt)

        self.messages.append(SystemMessage(system_prompt))


    def extend_sandbox(self, file_names):
        """
        Extend the sandbox with additional datasets.
        :param file_names: The names of the additional datasets.
        """
        try:
            # Combine file names and paths into a single list of tuples
            files = list(zip(file_names, [get_path_from_title(self.groupOwner, file_name) for file_name in file_names]))

            # Add code to load additional datasets
            setup_code = ""
            for idx, (file_name, file_path) in enumerate(files):
                name = file_name.split(".")[0]  # remove suffix
                setup_code += setup_code_for_file(name, file_path, idx + self.dataset_idx_offset)
            self.setup_code = setup_code
            self.dataset_idx_offset += len(file_names)

            # Run code on sandbox
            success, logs, _ = self.run_ai_generated_code(setup_code)
            if not success:
                log.error('Setup code had an error.')
                log.error(logs)
                self.finalize()
                sys.exit(1)
            else:
                log.info("Setup code ran successfully!")
                log.debug(logs)

            # Append the setup code and logs to the conversation
            msg_content = (
                f"Extended the sandbox with additional datasets: {', '.join(file_names)}\n\n"
                f"Setup code:\n```python\n{setup_code}\n```\n"
                f"Logs:\n{logs}"
            )
            print(msg_content)
            self.messages.append(AIMessage(msg_content))

        except Exception as e:
            log.error("Caught an exception in sandbox extend: %s", e, exc_info=True, backtrace=True, diagnose=True)
            self.finalize()
            sys.exit(1)


    def analyze(self, query: Union[str, dict]):
        # Handle multimodal input similar to AgenticRetriever
        text_query = ""
        human_message_content = None
        
        # Extract text and handle multimodal content
        if isinstance(query, dict) and "text" in query:
            text_content = query["text"]
            files = query.get("files", [])
            
            if files:
                # Create multimodal content for HumanMessage
                content_parts = [{"type": "text", "text": f"Request/Question: {text_content}\n"}]
                content_parts.extend(handle_attached_files(files))
                human_message_content = content_parts
            else:
                human_message_content = f"Request/Question: {text_content}\n"
                
            # Update query to be just the text for display purposes
            text_query = text_content
        elif isinstance(query, str):
            human_message_content = f"Request/Question: {query}\n"
            text_query = query
        else:
            # Fallback for other types
            human_message_content = f"Request/Question: {str(query)}\n"
            text_query = str(query)

        self.messages.append(HumanMessage(content=human_message_content))
        start_time = time.time()

        log.info("Invoking analyzer", user_question=text_query)

        thought_msg = gr.ChatMessage(
            role="assistant",
            content="",
            metadata={"title": "Analyzing the data...",
                      "status": "pending"},
        )

        yield thought_msg

        steps = 0
        prev_thought_msg_content = ""

        while steps < self.max_steps:
            steps += 1
            code = ""
            thought = ""
            try:
                if self.streaming:                   
                    for response in self.coding_llm.stream(self.messages):                       
                        thought_msg_content = prev_thought_msg_content
                        if "thought" in response:
                            thought = response["thought"]
                            thought_msg_content += f"**Thought**: {thought}\n"
                        if "code" in response:
                            code = response["code"].replace("```python", "").replace("```", "").strip()
                            thought_msg_content += f"**Code**: \n```python\n{code}\n``` \n"

                        thought_msg.content = thought_msg_content
                        yield thought_msg
                else:
                    response = self.coding_llm.invoke(self.messages)
                    codeAct = response["parsed"]
                    # Counting tokens
                    self.total_tokens += response["raw"].usage_metadata["total_tokens"]
                    self.total_input_tokens += response["raw"].usage_metadata["input_tokens"]
                    self.total_output_tokens += response["raw"].usage_metadata["output_tokens"]
                    if "output_token_details" in response["raw"].usage_metadata:
                        reasoning_tokens = response["raw"].usage_metadata["output_token_details"].get("reasoning", 0)
                        self.total_reasoning_tokens += reasoning_tokens
                        # Gemini models do not add reasoning tokens to the output tokens, although they are charged at this rate, so let's fix that
                        if hasattr(self.coding_client, "model") and "gemini" in self.coding_client.model:
                            self.total_output_tokens += reasoning_tokens

                    # As a sanity check, let's print the token counts
                    log.info("Total Token counts", total_tokens=self.total_tokens, input_tokens=self.total_input_tokens,
                            output_tokens=self.total_output_tokens, reasoning_tokens=self.total_reasoning_tokens)

                    # If the LLM wrapped the code with ```python ``` or ``` ```, remove it
                    code = codeAct.code.replace("```python", "").replace("```", "").strip()
                    thought = codeAct.thought

                    thought_msg.content += f"**Thought**: {thought}\n"
                    thought_msg.content += f"**Code**: \n```python\n{code}\n``` \n"
                    yield thought_msg
            except Exception as e:
                log.error("Caught an exception in LLM invocation: %s", e, exc_info=True, backtrace=True, diagnose=True)
                thought_msg.content += f"Error in LLM invocation"
                thought_msg.metadata["status"] = "done"
                thought_msg.metadata["title"] = "Error during analysis"
                thought_msg.metadata["duration"] = time.time() - start_time
                yield [thought_msg, "Unfortunately, there was an error during the LLM invocation. Please try again."]
                return

            success, response, is_final_answer = self.run_ai_generated_code(code)
            if not success:
                thought_msg.content += f"**Error**:\n```\n{response[0].replace("```", "")}\n```\n"
                new_msgs = [AIMessage("Thought: " + thought), AIMessage("**Code**: " + f"```python\n{code}\n```"), AIMessage(f"Error: \n{response[0]}"), HumanMessage("Please fix the error and try again.")]
                self.messages = self.messages + new_msgs
                yield thought_msg
            elif not is_final_answer:
                # Check if not all elements in response are empty strings
                if any(isinstance(r, str) and r.strip() for r in response):
                    thought_msg.content += f"**Output**:\n```\n{"\n".join(response)}\n```\n"
                thought_msg.metadata["status"] = "pending"
                thought_msg.metadata["title"] = f"Thinking step {steps} of at most {self.max_steps}"
                thought_msg.metadata["duration"] = time.time() - start_time

                new_msgs = [AIMessage("Thought: " + thought), AIMessage("**Code**: " + f"```python\n{code}\n```"), AIMessage(f"**Output**:\n{"\n".join(response)}")]
                self.messages = self.messages + new_msgs

                yield [thought_msg]
            else:
                thought_msg.metadata["status"] = "done"
                thought_msg.metadata["title"] = "Analysis completed"
                thought_msg.metadata["duration"] = time.time() - start_time

                new_msgs = [AIMessage("Thought: " + thought), AIMessage("**Code**: " + f"```python\n{code}\n```")]
                self.messages = self.messages + new_msgs

                yield [thought_msg] + response
                return
            
            prev_thought_msg_content = thought_msg.content
            
        thought_msg.metadata["status"] = "done"
        thought_msg.metadata["title"] = f"Maximum thinking steps ({self.max_steps}) exceeded"
        thought_msg.metadata["duration"] = time.time() - start_time
        yield [thought_msg, "Unfortunately, I was not able to find a solution to your request."]
        return
        

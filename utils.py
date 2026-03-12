import re
import os
import pandas as pd
import logging
import warnings
import re
import base64
import mimetypes
import pymupdf4llm
import structlog;log=structlog.get_logger()
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_azure_ai.chat_models.inference import AzureAIChatCompletionsModel
from langchain_ollama import ChatOllama
from azure.core.credentials import AzureKeyCredential
from typing import Tuple, List, Optional
from pydantic import Field, SecretStr


def init_mappings(groupOwner: str, excluded_extensions=[".tif"]):
    """
    Populates the mappings dict with mappings for title to file name and path based on the group owner.
    The mappings dict is used by `get_file_from_title` and `get_path_from_title`
    It also checks which files are actually available

    :param groupOwner: The group owner ID.
    """
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, f"./data/opendata/{groupOwner}/downloads.csv")

    extracted_data_path = os.path.join(script_dir, f"./data/opendata/{groupOwner}/extracted")
    available_files = set()
    for filename in os.listdir(extracted_data_path):
        available_files.add(filename)

    groupOwnerMapping = {}
    downloads_df = pd.read_csv(csv_path)

    # Filter out rows where the extracted_file is not in the available files, and report their amount
    filtered_df = downloads_df[downloads_df['extracted_file'].isin(available_files)]
    if len(filtered_df) < len(downloads_df):
        log.warning(f"Found {len(downloads_df) - len(filtered_df)} files in {csv_path} that are not available in {extracted_data_path}. They will be ignored.")

    for extension in excluded_extensions:
        # Filter out files with the specified extensions
        filtered_df = filtered_df[~filtered_df['extracted_file'].str.endswith(extension)]

    for index, row in filtered_df.iterrows():
        groupOwnerMapping[row["title"]] = row["extracted_file"]

    log.info(f"Will use {len(groupOwnerMapping)} files for group owner {groupOwner}.")

    mappings[groupOwner] = groupOwnerMapping


def clean(varStr):
    """
    Cleans a string to make it suitable for use as a variable name.
    - Converts to lowercase
    - Replaces special characters with underscores (taken from: https://stackoverflow.com/questions/3303312/how-do-i-convert-a-string-to-a-valid-variable-name-in-python)
    - Collapses multiple underscores into a single one
    - Replaces German umlauts with their respective two-letter combinations

    :param varStr: The string to be cleaned
    :return: The cleaned string
    """
    varStr = varStr.strip().lower()
    varStr = re.sub('ä', 'ae', varStr)
    varStr = re.sub('ö', 'oe', varStr)
    varStr = re.sub('ü', 'ue', varStr)
    varStr = re.sub(r'\W|^(?=\d)', '_', varStr)
    varStr = re.sub('_+', '_', varStr)  # Collapse multiple underscores to a single one
    return varStr


def does_title_exist(groupOwner: str, title: str):
    """
    Checks if a title exists in the mappings for a given group owner.

    :param groupOwner: The group owner ID.
    :param title: The title to check.
    :return: True if the title exists, False otherwise.
    """
    if groupOwner not in mappings:
        raise ValueError(f"Group owner {groupOwner} not initialized. Please call init_mappings(groupOwner) first.")
    
    return title in mappings[groupOwner]


def get_file_from_title(groupOwner: str, title: str):
    """
    Get file name based on the group owner and title.

    :param groupOwner: The group owner ID.
    :param title: The title of the dataset.
    :return: The generated file name
    """
    if groupOwner not in mappings:
        raise ValueError(f"Group owner {groupOwner} not initialized. Please call init_mappings(groupOwner) first.")
    
    if title not in mappings[groupOwner]:
        raise ValueError(f"Title '{title}' not found in mappings for group owner {groupOwner}.")
    
    return mappings[groupOwner][title]


def get_path_from_title(groupOwner: str, title: str):
    """
    Get a file path based on the group owner and title.

    :param groupOwner: The group owner ID.
    :param title: The title of the dataset.
    :return: The generated file path.
    """
    file = get_file_from_title(groupOwner, title)
    return f"./data/opendata/{groupOwner}/extracted/{file}"


def generate_system_prompt(file_names, metadata_docs, setup_code, logs):
    """
    Generates a system prompt for the model based on the provided file names, metadata documents, setup code, and logs.

    :param file_names: List of file names.
    :param metadata_docs: List of metadata documents.
    :param setup_code: The setup code that has already been run
    :param logs: The logs from the setup code.
    :return: The generated system prompt.
    """
    return f'''
    You have access to the following datasets:
    {", ".join([name for name in file_names])}

    Here is the metadata of these datasets:
    {"\n".join(str(m) for m in metadata_docs)}

    The following setup code has already been executed, which imports required libraries and prints the layer names of the datasets:
    {setup_code}

    Output:
    {logs}

    Your task:
    1. Write Python code that analyzes the dataset based on a user's request/question.
    2. Output the answer in text. In case you have geographical data available related to the user's question, please create a plot as well.
    3. The user will see what you are printing to the console and the plots, so please make sure to print only relevant information, and nicely format the output.
    4. You can format your output with Markdown, so you can use **bold** or *italics* to highlight important information, as well as use lists or tables (for tables, use pandas built-in to_markdown()) to present the data in a clear way.
    5. You can also use the contextily library to add a basemap to your plots, if needed. You can do so via `cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, crs=gdf.crs)` where `ax` is the axis of the plot and `gdf` is the geopandas dataframe.
    6. If you need to perform geocoding or reverse geocoding, you can use the instantiated `geolocator`, which is a geopy Nominatim geocoder. You can use it like this: `location = geolocator.geocode("Landesmusem, Zurich, Switzerland")` or `location = geolocator.reverse((<latitude>, <longitude>))`. The location object will contain fields longitude, latitude and address.
    7. If you do generate a plot, please make sure that it is large enough for almost full screen.

    You will now be given a user's request/question.
    You should first load the relevant layer into a dataframe and then analyze the data.
    Make sure to only use the language the user's question/request is written in from now on, both in the code and in the output.
    '''


def generate_system_prompt_v2(file_names, metadata_docs, setup_code, logs):
    """
    Generates a system prompt for the model based on the provided file names, metadata documents, setup code, and logs.

    :param file_names: List of file names.
    :param metadata_docs: List of metadata documents.
    :param setup_code: The setup code that has already been run
    :param logs: The logs from the setup code.
    :return: The generated system prompt.
    """
    return f'''
    You have access to the following datasets:
    {", ".join([name for name in file_names])}

    Here is the metadata of these datasets:
    {"\n".join(str(m) for m in metadata_docs)}

    The following setup code has already been executed, which imports required libraries and prints available layers and their structure and example data:
    {setup_code}

    Output:
    {logs}

    Your task:
    1. Write Python code that analyzes the dataset based on a user's request/question.
    2. The user will see what you are printing to the console, so please make sure to print only relevant information.
    3. You can format your output with Markdown, so you can use **bold** or *italics* to highlight important information, as well as use lists or tables (for tables, use pandas built-in to_markdown()) to present the data in a clear way.
    4. Only print pandas dataframes (using the to_markdown() method) if they are directly relevant to the user's question. Make sure to limit the number of rows and columns printed, as the available space is limited.
    5. Refrain from printing example data except if the user explicitly asks for it.
    6. Apart from printing the answer in text, you should output further modalities, including an interactive Folium map, and potentially a matplotlib figure object if the user asked for it or it makes sense. The last line of your code should be a list of all additional variables the user should see in the output.
       An example for a last line would be `answer = [fig, m]`. Do not include textual output in this list, rather use print statements for that.
    7. If you create a matplotlib figure, only include it in the answer list if it is relevant, but not in every subsequent question.
    8. A folium map is already instantiated and you can add layers to it. Update the map with new layers using geopandas' explore function, e.g.: `m = gdf.explore(m=m, name="Streets", tooltip=["str_name", "str_von", "str_bis"])`.
    9. Please do not add a layer control, this will be done automatically.
    10. If you need to perform geocoding, you have access to a `geocode` function, which takes as input a query string and returns a geopy location object in EPSG:4326 CRS. It internally uses an instantiated geocoder from geopy.
       You can use it like this: `location = geocode("Landesmusem, Zurich, Switzerland")`.

    You will now be given a user's request/question.
    You should first load the relevant layer into a dataframe and then analyze the data.
    Make sure to only use the language the user's question/request is written in from now on, both in the code and in the output.
    As previously mentioned, the last line of your code should be a list of all variables the user should see in the output, and acceptable types are (geo) dataframe, folium map, and string.
    '''


def generate_iterative_system_prompt(file_names, metadata_docs, setup_code, logs):
    """
    Generates a system prompt for the model based on the provided file names, metadata documents, setup code, and logs.

    :param file_names: List of file names.
    :param metadata_docs: List of metadata documents.
    :param setup_code: The setup code that has already been run
    :param logs: The logs from the setup code.
    :return: The generated system prompt.
    """
    return f'''
    You are an expert assistant who helps users solve analysis tasks about open government data using code blobs. You will be given a task to solve as best you can.
    To solve the task, you must plan forward to proceed in a series of steps, in a cycle of 'Thought:', 'Code:', and 'Output:' sequences, until you call `final_answer` to return the final answer.

    Here are the rules you should always follow to solve your task:
    1. Iteratively write Python code to inspect and analyze the dataset(s) based on a user's request/question. During each intermediate step, you can use 'print()' to save whatever important information you will then need. These print outputs will then appear in the 'Output:' field, which will be available as input for the next step.
    2. In the end, you have to return a final answer by calling the `final_answer` function. IMPORTANT: The user does not see your print statements, he only sees what you return in the final answer. Thus, make sure to always end your sequence of iterations with a call to final_answer.
    3. You can format your output for the final answer with Markdown, so you can use **bold** or *italics* to highlight important information, as well as use lists or tables (for tables, use pandas built-in to_markdown()) to present the data in a clear way.
    4. Only include pandas dataframes (using the to_markdown() method) in the final answer if they are directly relevant to the user's question. Make sure to limit the number of rows and columns printed, as the available space is limited.
    5. Apart from printing the final answer in text, you should output further modalities, including an interactive Folium map, and potentially a matplotlib figure object if the user asked for it or it makes sense. Your final answer should be a list of all variables the user should see in the output.
       An example for a final answer invocation would be `final_answer([text_answer, fig, m])`.
    6. If you create a matplotlib figure, only include it in the answer list if it is relevant, but not in every subsequent question. Same for the folium map.
    7. A folium map is already instantiated and you can add layers to it. Update the map with new layers using geopandas' explore function, e.g.: `m = gdf.explore(m=m, name="Streets", tooltip=["str_name", "str_von", "str_bis"])`.
    8. Please do not add a layer control, this will be done automatically.
    9. If you need to perform geocoding, you have access to a `geocode` function, which takes as input a query string and returns a geopy location object in EPSG:4326 CRS. It internally uses an instantiated geocoder from geopy.
       You can use it like this: `location = geocode("Landesmusem, Zurich, Switzerland")`.

    Here are a few examples:
    ---
    User request/question: "Wieviele öffentliche Spielplätze hat es im Escher-Wyss Quartier?"
    
    Thought: Ich benötige zuerst die Geometrie des Escher-Wyss Quartiers, um die Spielplätze darin zu zählen.
    Code:
    ```python
    # Lade das Escher-Wyss Quartier (Polygon)
    gdf_quartiere = gpd.read_file(path_1, layer='stzh.adm_statistische_quartiere_v')
    escher_wyss = gdf_quartiere[gdf_quartiere['qname'] == 'Escher-Wyss']

    # Überprüfe die Geometrie
    print(escher_wyss.geometry)
    ```
    Output: GeoSeries([], Name: geometry, dtype: geometry)

    Thought: Die Geometrie ist leer. Ich überprüfe die verfügbaren Quartiere.
    Code:
    ```python
    # Zeige alle verfügbaren Quartiere
    print(gdf_quartiere['qname'].unique())
    ```
    Output: ['Enge', 'Wollishofen', 'Altstadt', 'Escher Wyss', ...]

    Thought: Der Name des Quartiers ist "Escher Wyss" anstelle von "Escher-Wyss". Ich werde den Code anpassen.
    Code:
    ```python
    # Lade das Escher Wyss Quartier (Polygon)
    escher_wyss = gdf_quartiere[gdf_quartiere['qname'] == 'Escher Wyss']
    # Überprüfe die Geometrie
    print(escher_wyss.geometry)
    ```
    Output: 3    POLYGON ((2680009.144 1249565.021, 2680055.843...
            Name: geometry, dtype: geometry

    Thought: Jetzt habe ich die Geometrie des Escher Wyss Quartiers. Ich werde nun die Spielplätze zählen.
    Code:
    ```python
    # Lade die Spielplätze
    gdf_playgrounds = gpd.read_file(path_0, layer='stzh.poi_spielplatz_view')
    # Zähle die Spielplätze im Escher Wyss Quartier
    filtered_playgrounds = gdf_playgrounds[gdf_playgrounds.within(escher_wyss.geometry.iloc[0])]
    print(f"Spielplatz Anzahl: {{filtered_playgrounds.shape[0]}}")
    ```
    Output: 3

    Thought: Die Abfrage war erfolgreich und die Anzahl der Spielplätze im Escher Wyss Quartier ist 3. Ich werde diese Information formatieren, der folium Karte hinzufügen und als finale Antwort zurückgeben.
    Code:
    ```python
    m = escher_wyss.explore(m=m, color='red', style_kwds={{'fillOpacity':0.1}}, name="Escher Wyss Quartier")
    m = filtered_playgrounds.explore(m=m, color='blue', marker_kwds={'radius':5}, name="Spielplätze Escher Wyss", tooltip=["name", "geraete", "besonderes"])
    text_answer = f"Im Escher Wyss Quartier hat es {{filtered_playgrounds.shape[0]}} öffentliche Spielplätze."
    final_answer([text_answer, m])
    ```

    ---
    User request/question: "Sind die Parkplätze flächenmässig grösser im Parkhaus Migros Limmatplatz oder im Parkhaus Stampfenbach?"

    Thought: Ich benötige Geometrie der Parkplätze in den beiden Parkhäusern, und muss die Flächen berechnen. Ich beginne mit dem Parkhaus Stampfenbach.
    Code:
    ```python
    # Lade den Datensatz der Parkhäuser
    gdf_parking = gpd.read_file(path_0, layer='stzh.poi_parkhaus_view')
    # Filtere nach dem Parkhaus Stampfenbach
    stampfenbach = gdf_parking[gdf_parking['name'].str.contains("Stampfenbach")]
    print(stampfenbach.geometry)
    ```
    Output: 114    POINT (2683117.066 1249006.594)
            Name: geometry, dtype: geometry

    Thought: Der gegebene Datensatz enthält nur Punktgeometrie. Da auch keine der anderen Layer und Spalten einen Hinweis auf die Parkplatzgrösse gibt, kann ich diese Frage nicht beantworten.
    Code:
    ```python
    final_answer(["Ich kann diese Frage leider nicht beantworten, da im verwendeten Datensatz nur Punktgeometrien vorhanden sind und keine Informationen über die Fläche der Parkplätze."])
    ```

    ---
    User Request/Question: "Show me all of fountains in Zurich that have a trough made of marble."

    Thought: I need to load the fountains dataset and filter by the material of the trough, then display the results on a map.
    Code:
    ```python
    # Load the fountains dataset
    gdf_fountains = gpd.read_file(path_0, layer="wvz.wvz_brunnen")
    # Inspect what materials exist for the trough
    print(gdf_fountains['material_trog'].unique())
    ```
    Output: ['Kunststein' 'Holz' 'Granit' 'St. Imier-Marmor' 'St. Triphonmarmor' 'Jura-Laufener-Kalkstein' 'Cristallinamarmor' ...]

    Thought: There are multiple specific marble types present. I will filter for all of them.
    Code:
    ```python
    # Filter for fountains with trough made of marble
    marble_fountains = gdf_fountains[gdf_fountains['material_trog'].str.contains("Marmor", case=False, na=False)]
    print(len(marble_fountains))
    ```
    Output: 53

    Thought: I have found 53 fountains with troughs made of marble. I will now display them on a map and report the result to the user.
    Code:
    ```python
    m = marble_fountains.explore(m=m, color='grey', marker_kwds={'radius':5}, name="Marble Fountains", tooltip=["steinhauer", "material_trog", "historisches_baujahr"])
    text_answer = f"I have found {{len(marble_fountains)}} fountains in Zurich with troughs made of marble. Here is a map showing their locations."
    final_answer([text_answer, m])
    ```
    ---

    You have access to the following datasets:
    {", ".join([name for name in file_names])}

    Here is the metadata of these datasets:
    {"\n".join(str(m) for m in metadata_docs)}

    The following setup code has already been executed, which imports required libraries and prints available layers and their structure and example data:
    {setup_code}

    Output:
    {logs}

    You will now be given a user's request/question.
    You should first load the relevant layer into a dataframe and then analyze the data. Feel free to inspect the required data step-by-step.
    Make sure to only use the language the user's question/request is written in from now on, both in the code and in the output.
    As previously mentioned, conclude with a final_answer call with the arguments being a list of all variables the user should see in the output. Acceptable types are (geo) dataframe, folium map, and string.
    '''


def generate_rephrasal_prompt(num_rephrasings: int):
    """
    Generates a prompt for rephrasing a question.

    :param num_rephrasings: The number of rephrasings to generate.
    :return: The generated prompt.
    """
    return f"""
    Generate {num_rephrasings} slightly rephrased formulations of the question while keeping the meaning the same.
    Be very careful that you don't change the expected answer to the question, and also keep important terminology the same.
    """


ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

def plain_renderer(logger, name, event_dict):
    return f"{event_dict.get('timestamp')} [{event_dict.get('level')}] {ansi_escape.sub('', event_dict.get('event'))}"


def handle_attached_files(files: List[object]) -> List[dict]:
    """
    Handles images and PDF files attached to a message and converts them to appropriate message format.
    
    :param files: List of file objects or paths.
    :return: List of dictionaries representing the new content parts for appending to the message.
    """
    new_content_parts = []
    
    for file_info in files:
        # Handle file input - could be image, document, etc.
        if hasattr(file_info, 'path'):
            file_path = file_info.path
        elif isinstance(file_info, str):
            file_path = file_info
        else:
            # If it's a dict with path key
            file_path = file_info.get('path', str(file_info))
        
        # For images and PDFs, add as multimodal content
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.pdf')):
            try:               
                if file_path.lower().endswith('.pdf'):
                    # AzureOpenAI does not natively support PDF files, so we extract markdown text
                    # For now, we don't support images/graphics in PDFs
                    md_text = pymupdf4llm.to_markdown(file_path, ignore_images=True, ignore_graphics=True)
                    new_content_parts.append({
                        "type": "text",
                        "text": f"Content of PDF {os.path.basename(file_path)}:\n\n {md_text}"
                    })
                else:
                    # We can directly add image with base64 URL
                    base64_url = image_to_base64_url(file_path)
                    new_content_parts.append({
                        "type": "image_url", 
                        "image_url": {"url": base64_url}
                    })
            except Exception as e:
                # If file conversion fails, mention it in text instead
                log.warning(f"Failed to process file {file_path}: {e}")
                new_content_parts[0]["text"] += f"\n\nNote: Could not process file: {file_path}"
        # For other files, mention them in text (could be expanded later)
        else:
            new_content_parts[0]["text"] += f"\n\nAttached file in unsupported format: {file_path}"

    return new_content_parts


def image_to_base64_url(image_path: str) -> str:
    """
    Convert a local image to a base64 data URL.
    
    :param image_path: Path to the image.
    :return: Base64 data URL string.
    """
    try:
        # Get the MIME type
        mime_type, _ = mimetypes.guess_type(image_path)
        
        # Check if it's a supported image type
        if mime_type is None:
            # Try to infer from image extension
            image_ext = os.path.splitext(image_path)[1].lower()
            if image_ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']:
                mime_type = f'image/{image_ext[1:].replace("jpg", "jpeg")}'
            else:
                log.warning(f"Unknown image type for {image_path}, treating as binary")
                mime_type = 'application/octet-stream'
        
        # Validate supported types
        if not (mime_type.startswith('image/')):
            raise ValueError(f"Unsupported image type: {mime_type}. Only images are supported.")
        
        # Read and encode the image
        with open(image_path, 'rb') as image:
            encoded_string = base64.b64encode(image.read()).decode('utf-8')
        
        return f"data:{mime_type};base64,{encoded_string}"
    except Exception as e:
        log.error(f"Failed to convert image {image_path} to base64: {e}")
        raise ValueError(f"Could not process image: {image_path}")


def setup_logging(level=logging.INFO, log_filename=None):
    logging.captureWarnings(True)

    structlog.configure(
        processors=[
            structlog.dev.ConsoleRenderer()  # human-readable, or use JSONRenderer if preferred
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    handler = logging.StreamHandler()
    handler.setFormatter(structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(),
        foreign_pre_chain=[
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.processors.add_log_level,
        ]
    ))

    handlers = [handler]

    if log_filename is not None:
        # Ensure the logs directory exists
        logs_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data", "logs",
        )
        os.makedirs(logs_dir, exist_ok=True)
        log_file_path = os.path.join(
            logs_dir,
            os.path.basename(log_filename),
        )

        renderer = plain_renderer

        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setFormatter(structlog.stdlib.ProcessorFormatter(
            processor=renderer,
            foreign_pre_chain=[
                structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
                structlog.processors.add_log_level,
            ]
        ))
        handlers.append(file_handler)

    root_logger = logging.getLogger()
    root_logger.handlers = handlers
    root_logger.setLevel(level)


# Workaround to get tool-calling and structured output working with OpenRouter, adapted from:
# https://github.com/langchain-ai/langchain/discussions/27964#discussioncomment-12350857
class ChatOpenRouter(ChatOpenAI):
    openai_api_key: Optional[SecretStr] = Field(
        alias="api_key", default_factory=lambda: os.environ.get("OPENROUTER_API_KEY", None),
    )
    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"openai_api_key": "OPENROUTER_API_KEY"}

    def __init__(self,
                 openai_api_key: Optional[str] = None,
                 **kwargs):
        openai_api_key = openai_api_key or os.environ.get("OPENROUTER_API_KEY")
        super().__init__(base_url="https://openrouter.ai/api/v1", openai_api_key=openai_api_key, **kwargs)


def get_llm_client(llm_name: str, temperature: float = 0.0):
    """
    Returns an LLM client based on the provided name.
    Raises an error if the LLM name is not supported.
    If the model name contains a '/', then it is assumed to be an OpenRouter model.
    Otherwise, it is checked whether an AZURE_OPENAI_API_KEY is set, and if so, an AzureOpenAI client is returned.
    If not, it is checked whether an OPENAI_API_KEY is set, and if so, an OpenAI client is returned.
    If both are not set, an error is raised.

    :param llm_name: The model name of the LLM.
    :param temperature: The temperature for the LLM, default is 0.0. (lower temperature is better for tool usage)
    :return: The LLM client.
    """

    if llm_name not in SUPPORTED_LLMS:
        raise ValueError(f"LLM {llm_name} is not supported. Supported LLMs are: {SUPPORTED_LLMS}")
    
    azure_openai_endpoint = '/'.join(os.getenv("AZURE_OPENAI_ENDPOINT").split('/')[:-1]) + '/' + llm_name
    if llm_name in ['gpt-4.1', 'gpt-4.1-mini', 'gpt-4o', 'gpt-35-turbo']:
        if os.getenv("AZURE_OPENAI_API_KEY"):
            return AzureChatOpenAI(azure_deployment=llm_name, api_version="2024-10-21", temperature=temperature,
                                max_tokens=None, timeout=None, max_retries=2, azure_endpoint=azure_openai_endpoint)
        elif os.getenv("OPENAI_API_KEY"):
            return ChatOpenAI(model_name=llm_name, temperature=temperature, max_tokens=None, timeout=None, max_retries=2)
        else:
            raise ValueError("No API key found. Please set either AZURE_OPENAI_API_KEY or OPENAI_API_KEY environment variable.")
    elif llm_name in ['gpt-5', 'gpt-5-mini', 'gpt-5-nano', 'gpt-o1', 'gpt-o3-mini-preview']:
        # These models don't support the temperature parameter
        if os.getenv("AZURE_OPENAI_API_KEY"):
            return AzureChatOpenAI(azure_deployment=llm_name, api_version="2025-02-01-preview",
                                max_tokens=None, timeout=None, max_retries=2, azure_endpoint=azure_openai_endpoint)
        elif os.getenv("OPENAI_API_KEY"):
            return ChatOpenAI(model_name=llm_name, max_tokens=None, timeout=None, max_retries=2)
        else:
            raise ValueError("No API key found. Please set either AZURE_OPENAI_API_KEY or OPENAI_API_KEY environment variable.")
    elif llm_name in ['gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.5-flash-lite-preview-06-17']:
        return ChatGoogleGenerativeAI(model=llm_name, temperature=0, max_tokens=None, timeout=None, max_retries=2)
    elif llm_name in ['meta-llama-70B-instruct-latest', 'mistral-codestral', 'mistral-large']:
        azure_endpoint = os.getenv("AZURE_ENDPOINT") + '/' + llm_name
        return AzureAIChatCompletionsModel(
            endpoint=azure_endpoint,
            credential=AzureKeyCredential(os.environ.get("AZURE_API_KEY")),
            temperature=temperature, max_tokens=None, timeout=None, max_retries=2
        )
    elif llm_name in ['moonshotai/kimi-k2:free', 'meta-llama/llama-3.3-70b-instruct:free', 'qwen/qwen3-235b-a22b:free',
                      'qwen/qwen3-coder', 'openai/gpt-oss-120b', 'anthropic/claude-3.5-haiku', 'anthropic/claude-sonnet-4',
                      'meta-llama/llama-4-maverick']:
        return ChatOpenRouter(
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            model_name=llm_name,
        )
    elif llm_name in ['gpt-oss:20b']:
        return ChatOllama(
            model=llm_name,
            temperature=temperature,
            max_retries=2
        )
    else:
        raise ValueError("Mismatch between supported LLMs and currently implemented LLMs.")


def get_llm_pricing(llm_name: str) -> Tuple[float, float]:
    """
    Returns the pricing for the given LLM name.
    Raises an error if the LLM name is not supported.

    :param llm_name: The model name of the LLM.
    :return: The pricing for the LLM.
    """
    if llm_name not in LLM_PRICES:
        raise ValueError(f"Utils: {llm_name} is missing pricing information.")
     
    return LLM_PRICES[llm_name]


SUPPORTED_LLMS = ['gpt-5', 'gpt-5-mini', 'gpt-5-nano',
                  'gpt-4.1', 'gpt-4.1-mini', 'gpt-4o', 'gpt-o1', 'gpt-o3-mini-preview', 
                  'gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.5-flash-lite-preview-06-17',
                  'meta-llama-70B-instruct-latest', 'mistral-codestral', 'mistral-large',
                  'moonshotai/kimi-k2:free', 'meta-llama/llama-3.3-70b-instruct:free', 'qwen/qwen3-235b-a22b:free',
                  'qwen/qwen3-coder', 'openai/gpt-oss-120b', 'gpt-oss:20b',
                  'meta-llama/llama-4-maverick']
# meta-llama-70B-instruct-latest = Llama 3.3 70B Instruct

# Prices in USD per 1M tokens
LLM_PRICES = {
    # OpenAI: https://platform.openai.com/docs/pricing
    "gpt-5": (1.25, 10),
    "gpt-5-mini": (0.25, 2),
    "gpt-5-nano": (0.05, 0.4),
    'gpt-4.1': (2, 8),
    'gpt-4.1-mini': (0.4, 1.6),
    'gpt-4o': (2.5, 10),
    'gpt-o1': (15, 60),
    'gpt-o3-mini-preview': (1.1, 4.4),
    'text-embedding-3-large': (0.13, 0.13),
    # Google: https://ai.google.dev/gemini-api/docs/pricing
    # Both input and ouput are <= 200k tokens for this system
    'gemini-2.5-pro': (1.25, 10), 
    'gemini-2.5-flash': (0.3, 2.5),
    'gemini-2.5-flash-lite-preview-06-17': (0.1, 0.4),
    # Mistral: https://mistral.ai/pricing#api-pricing
    'mistral-codestral': (0.3, 0.9),
    'mistral-large': (2, 6),
    # Meta (free?)
    'meta-llama-70B-instruct-latest': (0, 0),
    # OpenRouter (free):
    'moonshotai/kimi-k2:free': (0, 0),
    'meta-llama/llama-3.3-70b-instruct:free': (0, 0),
    'qwen/qwen3-235b-a22b:free': (0, 0),
    # OpenRouter (paid):
    'qwen/qwen3-coder': (0.2, 0.8), # Qwen3-Coder-480B-A35B-Instruct
    'openai/gpt-oss-120b': (0.073, 0.29), # OpenAI GPT-OSS 120B
    'meta-llama/llama-4-maverick': (0.15, 0.6),
    # Ollama (self-hosted):
    'gpt-oss:20b': (0, 0),
}


LLM_JUDGE_SYSTEM_PROMPT = """
You task is to evaluate a question/answering system. You will be given a question, a reference answer and a predicted answer.
Your task is to evaluate whether the predicted answer correctly answers the question (thus aligns with the reference answer), or not.
For results involving numbers, you should accept various formats of the same number as correct, e.g. 42500, 42,500, 42'500, etc.
"""

AGENTIC_RETRIEVER_SYSTEM_PROMPT = """
You are a helpful retrieval assistant for the city of Zurich that given a user query, tries to find relevant open datasets that can be used to answer the user's question.
You have access to a database with metadata of all datasets of the city of Zurich.
Think step-by-step on how to best find relevant datasets.
You have two tools at your disposal:
1. `vector_search`: This tool allows you to search for relevant datasets. You can use it multiple times (at most 3). Use it with German queries for best results.
2. `report_results`: This tool allows you to report the relevant datasets (if any). You can use it as soon as you are confident that you have found all relevant datasets for answering the user's question, or after three searches in the database.
Your output is in the form of a validation object, which contains whether the question can be answered, the relevant datasets, and an explanation of how you arrived at the conclusion.

After performing a vector search, you will be provided with a list of metadata documents that match your query, including titles, descriptions and attributes.
You should check the titles and descriptions and also make sure that relevant attributes/classes are present for answering the query (e.g. geometry).
Try other formulations of the query if the results do not contain relevant datasets.
Your reported results will be passed to a system that loads the datasets and runs (geo)pandas Python code on them to answer the user's question.
You can assume that the system has capabilities to join datasets, filter them, or perform other operations (e.g. computing length or area based on geometry).
It may also make sense to perform separate searches for different aspects of a user's question, just keep the limit of three searches in mind.

Here are a few examples of user questions and good calls of the vector_search tool:
User question: In welchen Stadtkreisen gibt es keine Gerichte?
vector_search("Gerichte Standorte Zürich")
vector_search("Stadtkreise")

User question: Can you show me all benches in Zurich that are likely under the shade of a tree?
vector_search("Sitzbänke Standorte")
vector_search("Kataster Baumbestand inkl. Kronendurchmesser")

User question: How are you?
no search, just report that no relevant datasets were found

User question: Welche Kategorien von Verkehrszonen gibt es in Zürich?
vector_search("Verkehrszonen Kategorien")

User question: Welches Quartier hatte im Jahr 2022 die höchste Anzahl an Verkehrsunfällen mit Personenschäden?
vector_search("Verkehrsunfälle mit Personenschäden 2022")
vector_search("Quartiere in der Stadt Zürich")

Note in particular the pattern of queries with different aspects and also that you should find explicit datasets for explicitly requested geographical aggregations such as "Stadtkreise" or "Quartiere".

Remember:
Carefully check titles, descriptions and attributes of the datasets returned by the vector search.
If after at most three searches in the database, you still cannot identify any suitable datasets, you will use the report tool as well to inform the user that no relevant datasets were found.
Also, if there are datasets with the same data, please only use the most relevant one, typically the most recent one.
You also use the report tool in case the user does not ask a question.

Now Begin! If you solve the task correctly, you will receive a reward of $1,000,000.
"""

CONSOLE_LOGO = """
 ██████╗  ██████╗ ██████╗ ██╗  ██╗ █████╗ ██╗     ██╗     
██╔═══██╗██╔════╝ ██╔══██╗██║  ██║██╔══██╗██║     ██║     
██║   ██║██║  ███╗██║  ██║███████║███████║██║     ██║     
██║   ██║██║   ██║██║  ██║╚════██║██╔══██║██║     ██║     
╚██████╔╝╚██████╔╝██████╔╝     ██║██║  ██║███████╗███████╗
 ╚═════╝  ╚═════╝ ╚═════╝      ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝
  Accessible Retrieval & Analysis of Open Government Data
"""

mappings = {} # dict of dicts, for each groupOwner, a dict of title to file name

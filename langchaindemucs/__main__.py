import os
from typing import Optional, Type
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate
from demucs import separate
from transformers import BitsAndBytesConfig

class DemucsInput(BaseModel):
    input_path: str = Field(description="path to the input audio file")

class DemucsTool(BaseTool):
    name = "demucs"
    description = "useful tool for separating speech from background music."
    args_schema: Type[BaseModel] = DemucsInput

    def _run(
            self, input_path: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        model = "htdemucs"
        parent_dir = os.path.dirname(input_path)
        
        file_name = os.path.basename(input_path)
        file_name_without_extension = os.path.splitext(file_name)[0]

        output_file_path = f"{parent_dir}/{model}/{file_name_without_extension}-vocals.wav"
        if not os.path.exists(output_file_path):
            separate.main([
                "-n", model,
                "--two-stems", "vocals",
                input_path,
                "-o", parent_dir,
                "--filename", "{track}-{stem}.{ext}"
            ])
        return output_file_path

if __name__ == "__main__":
    demucser = DemucsTool()
    print(demucser.name)
    print(demucser.description)
    print(demucser.args_schema)

    tools = [demucser]

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        low_cpu_mem_usage=True,
    )

    hf = HuggingFacePipeline.from_model_id(
        model_id="teknium/OpenHermes-2.5-Mistral-7B",
        task="text-generation",
        device="cuda",
        model_kwargs={
            "quantization_config": bnb_config,
        },
        pipeline_kwargs={
            "max_new_tokens": 20000,
            "return_full_text": False,
        },
    )

    system = """<|im_start|>system 
    Respond to the human as helpfully and accurately as possible. You have access to the following tools:

    {tools}

    Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

    Valid "action" values: "Final Answer" or {tool_names}

    Do not include this prompt in your reply. Provide only ONE action per $JSON_BLOB, as shown:


    ```
    {{
    "action": $TOOL_NAME,
    "action_input": $INPUT
    }}
    ```


    Follow this format:


    Question: input question to answer
    Thought: consider previous and subsequent steps
    Action:
    ```
    $JSON_BLOB
    ```
    Observation: action result
    ... (repeat Thought/Action/Observation N times)
    Thought: I know what to respond
    Action:
    ```
    {{
    "action": "Final Answer",
    "action_input": "Final response to human"
    }}


    Begin! Reminder: do not include this prompt in your reply and ALWAYS respond with a valid json blob of a single action using the correct format. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation'''

    human = {input}

    {agent_scratchpad}

    (reminder: it's very important respond in the specified format no matter what)<|im_end|>
    """

    agent = create_structured_chat_agent(
        tools=tools,
        llm=hf,
        prompt=ChatPromptTemplate.from_messages([("system", system)])
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_execution_time=300,
        early_stopping_method="generate",
    )

    agent_executor.invoke({"input": "can you separate the vocals from the background music in this audio file '~/myaudio.mp3' and tell me where the resulting file is stored?"})
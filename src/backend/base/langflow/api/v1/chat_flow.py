import json
import os
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel

from langflow.api.utils import CurrentActiveUser, DbSession
from langflow.api.v1.flows import _new_flow
from langflow.services.database.models.flow.model import FlowCreate, FlowRead
from langflow.services.deps import get_storage_service
from langflow.services.storage.service import StorageService

router = APIRouter(prefix="/chat_flow", tags=["Chat Flow"])


class ChatFlowRequest(BaseModel):
    prompt: str
    openai_api_key: str | None = None
    provider: str = "openai"
    model_name: str | None = None
    base_url: str | None = None


def _get_chat_model(provider: str, api_key: str | None, model_name: str | None, base_url: str | None):
    """Factory function to initialize the appropriate LangChain chat model based on the provider."""
    try:
        if provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(api_key=api_key, model=model_name or "gpt-4o", base_url=base_url)
        if provider == "groq":
            from langchain_groq import ChatGroq

            return ChatGroq(api_key=api_key, model=model_name or "llama3-70b-8192", base_url=base_url)
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(api_key=api_key, model=model_name or "claude-3-5-sonnet-20240620", base_url=base_url)
        if provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                google_api_key=api_key, model=model_name or "gemini-2.0-flash", convert_system_message_to_human=True
            )
        if provider == "ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(base_url=base_url or "http://localhost:11434", model=model_name or "llama3.1")
        raise ValueError(f"Unsupported provider: {provider}")
    except ImportError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Dependency for {provider} not found. Please install the required package. Error: {e!s}",
        )


def _get_all_components(components_path: Path) -> str:
    """Helper to load components from the component_index.json file.
    Returns a formatted string list of available components.
    """
    try:
        with open(components_path) as f:
            data = json.load(f)

        components = []
        # Iterate over categories (the structure is [CategoryName, {ComponentName: Details...}])
        for category in data.get("entries", []):
            if isinstance(category, list) and len(category) == 2:
                comp_dict = category[1]
                for comp_name, details in comp_dict.items():
                    desc = details.get("description", "No description")
                    components.append(f"- {comp_name}: {desc}")

        return "\n".join(components)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error loading components: {e}")
        # Fallback to a basic list if file not found or error occurs
        return (
            "- ChatInput: Input from the user.\n"
            "- Prompt: To format the input.\n"
            "- OpenAIModel: LLM processing.\n"
            "- ChatOutput: Output to the user."
        )


@router.post("/", response_model=FlowRead, status_code=201)
async def create_chat_flow(
    *,
    session: DbSession,
    request: ChatFlowRequest,
    current_user: CurrentActiveUser,
    storage_service: Annotated[StorageService, Depends(get_storage_service)],
):
    """Create a new flow based on a natural language prompt using a Generative AI provider."""
    # Resolve API Key
    api_key = request.openai_api_key

    if not api_key:
        if request.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        elif request.provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
        elif request.provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
        elif request.provider == "google":
            api_key = os.getenv("GOOGLE_API_KEY")

    if request.provider != "ollama" and not api_key:
        raise HTTPException(
            status_code=400,
            detail=f"API key is required for {request.provider}. Please provide it in the request or set it in the environment variables.",
        )

    try:
        llm = _get_chat_model(request.provider, api_key, request.model_name, request.base_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # Calculate path relative to this file to find component_index.json
    # Ancestry: src/backend/base/langflow/api/v1/chat_flow.py -> src/langflow
    try:
        current_dir = Path(__file__).resolve().parent
        # Go up 5 levels from api/v1 to reach 'src' root
        src_root = current_dir.parents[4]

        component_index_path = src_root / "lfx" / "src" / "lfx" / "_assets" / "component_index.json"

        if not component_index_path.exists():
            logger.warning(f"Warning: Component index not found at {component_index_path}")
            components_list = _get_all_components(Path("invalid_path"))  # Triggers fallback
        else:
            components_list = _get_all_components(component_index_path)

    except Exception as e:  # noqa: BLE001
        logger.error(f"Error resolving component path: {e}")
        components_list = _get_all_components(Path("invalid_path"))  # Triggers fallback

    system_prompt = f"""
    You are an expert in Langflow and LangChain.
    Your task is to generate a valid Langflow JSON structure based on the user's request.
    The JSON should be a valid 'Flow' data structure with 'nodes' and 'edges'.

    The 'data' field of the Flow should contain:
    - 'nodes': A list of nodes. Each node must have:
        - 'id': A unique string ID (e.g., "ChatInput-1").
        - 'type': The component type (e.g., "ChatInput", "Prompt", "OpenAIModel", "ChatOutput").
        - 'position': {{'x': int, 'y': int}}.
        - 'data': A dictionary containing:
            - 'type': Same as the outer type.
            - 'node': A dictionary describing the node configuration (template).
                - 'template': A dictionary where keys are field names and values are dicts with 'value'.

    - 'edges': A list of edges connecting nodes. Each edge must have:
        - 'source': Source node ID.
        - 'target': Target node ID.
        - 'sourceHandle': Output handle name (usually "start" or specific output name).
        - 'targetHandle': Input handle name (usually "end" or specific input name).

    Available Components (Names and Descriptions):
    {components_list}

    Example Chain: ChatInput -> Prompt -> OpenAIModel -> ChatOutput.

    Return ONLY valid JSON. Do not include markdown code blocks.
    """

    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Create a Langflow flow for: {request.prompt}"),
        ]

        if request.provider == "mock":
            # Return a valid dummy flow for testing purposes
            content = """
            {
              "data": {
                "nodes": [
                  {
                    "id": "ChatInput-1",
                    "type": "ChatInput",
                    "position": {"x": 100, "y": 200},
                    "data": {
                      "type": "ChatInput",
                      "node": {
                        "template": {}
                      }
                    }
                  },
                  {
                    "id": "ChatOutput-1",
                    "type": "ChatOutput",
                    "position": {"x": 500, "y": 200},
                    "data": {
                       "type": "ChatOutput",
                       "node": {
                         "template": {}
                       }
                    }
                  }
                ],
                "edges": [
                  {
                    "source": "ChatInput-1",
                    "target": "ChatOutput-1",
                    "sourceHandle": "start",
                    "targetHandle": "end"
                  }
                ]
              }
            }
            """
        else:
            # Invoke the model
            response = await llm.ainvoke(messages)
            content = response.content

        if not content:
            raise HTTPException(status_code=500, detail="Failed to generate flow content.")

        flow_data = content

        # Clean up code blocks if present
        if "```json" in flow_data:
            flow_data = flow_data.split("```json")[1].split("```")[0].strip()
        elif "```" in flow_data:
            flow_data = flow_data.split("```")[1].split("```")[0].strip()

        try:
            flow_json = json.loads(flow_data)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to decode JSON from model output: {flow_data[:100]}..."
            ) from e

        # Basic parsing fix if LLM wrapped it incorrectly
        if "nodes" not in flow_json and "data" in flow_json:
            flow_json = flow_json["data"]

        # Create the Flow object
        flow_create = FlowCreate(
            name=f"Generated Flow: {request.prompt[:30]}",
            description=f"Auto-generated flow for: {request.prompt}",
            data=flow_json,
            is_component=False,
        )

        return await _new_flow(
            session=session, flow=flow_create, user_id=current_user.id, storage_service=storage_service
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

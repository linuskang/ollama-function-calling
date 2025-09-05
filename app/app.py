from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, List, Optional, Callable
from ollama import Client
from utils.tools import (
    get_reviews, get_info, get_top_bottom_bubblers,
    get_reviews_tool, get_information_tool, get_top_bottom_tool
)

client = Client(host='http://localhost:11434')

available_functions: Dict[str, Callable] = {
    'get_reviews': get_reviews,
    'get_info': get_info,
    'get_top_bottom_bubblers': get_top_bottom_bubblers,
}

app = FastAPI()

class ChatRequest(BaseModel):
    prompt: str
    conversation: Optional[List[Dict[str, str]]] = None  # Optional past messages

class ChatResponse(BaseModel):
    reply: str
    tool_calls: Optional[List[Dict[str, Any]]] = None

@app.post("/chat", response_model=ChatResponse)
async def ai_chat(request: ChatRequest):
    conversation = request.conversation or []
    conversation.append({"role": "user", "content": request.prompt})

    response = client.chat(
        model="gpt-oss:20b",
        messages=conversation,
        tools=[get_reviews_tool, get_information_tool, get_top_bottom_tool],
    )

    tool_call_data = []
    if response.message.tool_calls:
        for tool_call in response.message.tool_calls:
            function_name = tool_call.function.name
            args = tool_call.function.arguments
            if function_to_call := available_functions.get(function_name):
                # Run the tool
                result = function_to_call(**args)

                # Ask AI to interpret this result naturally
                follow_up = client.chat(
                    model="gpt-oss:20b",
                    messages=[
                        {"role": "system",
                         "content": "You are a helpful assistant that explains data in natural language."},
                        {"role": "user",
                         "content": f"The user asked: {request.prompt}. Here is the raw data from the tool {function_name}: {result}. Please explain it clearly in natural language."}
                    ]
                )
                conversation.append({"role": "assistant", "content": follow_up.message.content})
                return ChatResponse(reply=follow_up.message.content)

    return ChatResponse(reply=response.message.content, tool_calls=tool_call_data)

@app.get("/")
def hello():
    return {"message": "Hello, world!"}
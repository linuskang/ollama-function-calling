import requests
from typing import Dict, Any, Callable
from ollama import Client

client = Client(host='http://localhost:11434')

def get_reviews(bubbler_id: int) -> Dict[str, Any]:
    url = f"https://bubbly.lkang.au/api/reviews?bubblerId={bubbler_id}"
    response = requests.get(url, verify=False)
    if response.status_code != 200:
        return {"error": f"Failed to fetch reviews (status code {response.status_code})"}
    data = response.json()
    return {"bubblerId": bubbler_id, "reviews": data}

get_reviews_tool = {
    'type': 'function',
    'function': {
        'name': 'get_reviews',
        'description': 'Fetch reviews for a specific bubbler ID from the Bubbly API.',
        'parameters': {
            'type': 'object',
            'required': ['bubbler_id'],
            'properties': {
                'bubbler_id': {'type': 'integer', 'description': 'The ID of the bubbler to fetch reviews for.'}
            },
        },
    },
}

available_functions: Dict[str, Callable] = {
    'get_reviews': get_reviews,
}

print("Chat with AI (type 'exit' to quit)")
conversation = []

while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        break

    conversation.append({'role': 'user', 'content': user_input})

    response = client.chat(
        'gpt-oss:20b',
        messages=conversation,
        tools=[get_reviews_tool],
    )

    if response.message.tool_calls:
        for tool in response.message.tool_calls:
            if function_to_call := available_functions.get(tool.function.name):
                raw_data = function_to_call(**tool.function.arguments)
                follow_up = client.chat(
                    'gpt-oss:20b',
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that explains data in a human-friendly way."},
                        {"role": "user", "content": f"Here is the raw data from the function {tool.function.name}: {raw_data}. Please summarize or explain it naturally."}
                    ]
                )
                print("AI:", follow_up.message.content)
                conversation.append({'role': 'assistant', 'content': follow_up.message.content})
            else:
                print(f"Function {tool.function.name} not found")
    else:
        print("AI:", response.message.content)
        conversation.append({'role': 'assistant', 'content': response.message.content})
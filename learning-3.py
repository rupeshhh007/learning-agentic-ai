from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client()
def get_attendance(subject):
    attendance = {
        "CN": 68,
        "AI": 82,
        "DBMS": 76
    }

    return attendance.get(subject.upper(), "Subject not found")

attendance_tool = {
    "type": "function",
    "name": "get_attendance",
    "description": "Get the student's attendance percentage for a university subject.",
    "parameters": {
        "type": "object",
        "properties": {
            "subject": {
                "type": "string",
                "description": "The subject code, for example CN, AI, or DBMS."
            }
        },
        "required": ["subject"]
    }
}
tool_registry = {
    "get_attendance": get_attendance
}

def run_agent(user_message):

    current_input = user_message
    previous_interaction_id = None

    while True:

        interaction = client.interactions.create(
            model="gemini-3.8-flash",
            input=current_input,
            tools=[attendance_tool],
            previous_interaction_id=previous_interaction_id
        )

        tool_calls = []

        for step in interaction.steps:
            if step.type == "function_call":
                tool_calls.append(step)

        if not tool_calls:
            return interaction.output_text

        print("NUMBER OF TOOL CALLS:", len(tool_calls))

        for call in tool_calls:
            print(call.name, call.arguments)

        function_results = []

        for tool_call in tool_calls:
            function = tool_registry[tool_call.name]
            result = function(** tool_call.arguments)

            function_results.append(
                {
                    "type": "function_result",
                    "name": tool_call.name,
                    "call_id": tool_call.id,
                    "result": [
                        {
                            "type": "text",
                            "text": str(result)
                        }
                    ]
                }
            )
        current_input = function_results
        previous_interaction_id = interaction.id



answer = run_agent("Which attendance is higher, CN or AI?")
print(answer)



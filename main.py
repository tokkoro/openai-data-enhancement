import os

import openai
import valohai

valohai.prepare(
    step="gpt35-turbo-filter",
    image="python:3.10",
    default_inputs={
        "data_to_clean": "",
    },
    default_parameters={
        "token_limit_per_message": 800,
        "max_messages": 10,
        "max_tokens": 10_000,
        "chat_prompt": "Give me all questions in following text:\n{body}",
    }
)

logger = valohai.logger()

openai.api_key = os.getenv("OPENAI_API_KEY")

chars_per_token = 4
limit_per_message = valohai.parameters("token_limit_per_message").value
result = []
max_messages = valohai.parameters("max_messages").value
max_tokens = valohai.parameters("max_tokens").value
chat_prompt = valohai.parameters("chat_prompt").value
if "{body}" not in chat_prompt:
    raise ValueError("chat_prompt must contain {body} placeholder")
message_num = 0
total_tokens = 0

with open(valohai.inputs("data_to_clean").path(), "r") as data_to_clean:
    while message_num < max_messages or max_messages == 0:
        body = ""
        while len(body) / chars_per_token < limit_per_message:
            line = data_to_clean.readline()
            if not line:
                break
            body += line

        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                # {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": chat_prompt.format(body=body),
                }
            ]
        )

        matches = res["choices"][0]["message"]["content"]
        for question in matches.splitlines():
            if not question:
                continue
            result.append(question)

        usage = res["usage"]
        total_tokens += usage["total_tokens"]
        if total_tokens > max_tokens:
            break
        new_matches_count = matches.count('\n')
        print(f"New matches: {new_matches_count}")

        message_num += 1
        if not line:
            break

output_path = valohai.outputs("cleaned_data").path("cleaned_data.txt")
with open(output_path, "w") as f:
    f.write("\n".join(result))

logger.log("tokens_used", total_tokens)
logger.log("num_results", len(result))

logger.flush()

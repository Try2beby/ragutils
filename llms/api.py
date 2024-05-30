import traceback
import requests
import ujson as json
from typing import Any, Dict, List, Tuple, Union, Generator
from llmparty.server.v1.utils import parse_stream
from llmparty.prompt.format import Message


def api_generate(
    api_path: str = None, messages: List[Dict] = None, stream: bool = False, **params
) -> Union[Generator[Tuple[str, Dict], None, None], Tuple[str, None]]:
    try:
        parsed_messages = []
        for message in messages:
            if isinstance(message, Message):
                parsed_messages.append(message.model_dump())
            else:
                parsed_messages.append(message)

        request_data = dict(messages=parsed_messages, stream=True)
        request_data.update(params)
        reply = requests.post(
            api_path,
            json=request_data,
            headers={"Content-Type": "application/json"},
            stream=True,
        )

        full_text = ""
        error_message = None
        if reply.status_code != 200:
            error_message = "request error [code={}]".format(reply.status_code)
            raise RuntimeError(error_message)
        else:
            for line in parse_stream(reply.iter_lines()):
                chunk = json.loads(line)
                if "code" in chunk and chunk["code"] != 200:
                    if "message" in chunk:
                        error_message = chunk["message"]
                    else:
                        error_message = "unknown error"
                    raise RuntimeError(error_message)
                else:
                    if "content" in chunk["choices"][0]["delta"]:
                        new_text = chunk["choices"][0]["delta"]["content"]
                        if stream:
                            yield new_text, chunk
                        else:
                            full_text += new_text
            reply.close()

        if not stream:
            yield full_text, None

    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(
            "Unexpected error: {}\nrequest data: {}".format(e, request_data)
        )

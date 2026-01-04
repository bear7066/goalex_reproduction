import os
import json
from typing import List, Dict, Union
from tqdm import tqdm, trange
import time
from transformers import GPT2Tokenizer
from copy import deepcopy
import openai
import random
import numpy as np


openai.api_key = os.environ.get("OPENAI_API_KEY", "")
GPT2TOKENIZER = GPT2Tokenizer.from_pretrained("gpt2")

import http.client
import os

class OuterMedusaLLM:
    def __init__(self, model="gpt-oss-20b"):
        self.model = model
        self.cum_prompt_tokens = 0
        self.cum_completion_tokens = 0

        try:
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass
        except ImportError:
            pass
        self.OUTER_MEDUSA_ENDPOINT = os.environ.get("OUTER_MEDUSA_ENDPOINT", "")
        self.OUTER_MEDUSA_API_KEY = os.environ.get("OUTER_MEDUSA_API_KEY", "")

        if self.OUTER_MEDUSA_ENDPOINT == "":
            raise ValueError("OUTER_MEDUSA_CLIENT is not set")
        if self.OUTER_MEDUSA_API_KEY == "":
            raise ValueError("OUTER_MEDUSA_API_KEY is not set")

        # Parse and extract host[:port] from OUTER_MEDUSA_ENDPOINT if it is a URL
        from urllib.parse import urlparse

        parsed_endpoint = urlparse(
            self.OUTER_MEDUSA_ENDPOINT
            if self.OUTER_MEDUSA_ENDPOINT.startswith("http")
            else "https://" + self.OUTER_MEDUSA_ENDPOINT
        )
        host = (
            parsed_endpoint.netloc if parsed_endpoint.netloc else parsed_endpoint.path
        )
        # Remove any trailing slashes from host
        self.host = host.rstrip("/")

        conn = self._get_connection()
        try:
            response, response_data = self._get_response(
                conn,
                "GET",
                "/v1/models",
            )
            if response.status != 200:
                raise ValueError(
                    f"Request failed: {response.status} {response.reason} - {response_data}"
                )

        finally:
            conn.close()

        print(f"Connected to Outer Medusa client at {self.OUTER_MEDUSA_ENDPOINT}")

    def change_model(self, model: str):
        import json

        conn = self._get_connection()
        try:
            response, response_data = self._get_response(conn, "GET", "/v1/models")
            if response.status != 200:
                raise ValueError(
                    f"Request failed: {response.status} {response.reason} - {response_data}"
                )
        finally:
            conn.close()

        data = json.loads(response_data)
        models = data.get("data", [])
        if model in [m["id"] for m in models]:
            self.model = model
            return
        else:
            raise ValueError(f"Model {model} not found in available models")

    def __str__(self):
        return f"OuterMedusaLLM(model={self.model})"

    def _get_connection(self):
        return http.client.HTTPSConnection(self.host)

    def _get_response(self, conn, method, path, headers=None, body=None):
        if headers is None:
            headers = {
                "Authorization": f"Bearer {self.OUTER_MEDUSA_API_KEY}",
                "Content-Type": "application/json",
            }
        conn.request(method, path, body=body, headers=headers)
        response = conn.getresponse()
        response_data = response.read().decode()
        return response, response_data

    def generate(self, prompt: str, sys_prompt: str | None = None):
        import json

        path = "/v1/chat/completions"
        payload = {"model": self.model, "messages": []}
        if sys_prompt:
            payload["messages"].append({"role": "system", "content": sys_prompt})
        payload["messages"].append({"role": "user", "content": prompt})

        conn = self._get_connection()
        try:
            response, response_data = self._get_response(
                conn,
                "POST",
                path,
                body=json.dumps(payload),
            )
            if response.status != 200:
                raise ValueError(
                    f"Request failed: {response.status} {response.reason} - {response_data}"
                )
            data = json.loads(response_data)
            # Response shape: { "choices": [{ "message": { "content": ... } }], "usage": {"prompt_tokens": ..., "completion_tokens": ... } }
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            prompt_eval_count = data.get("usage", {}).get("prompt_tokens", 0)
            eval_count = data.get("usage", {}).get("completion_tokens", 0)

            # Update cumulative token counts
            self.cum_prompt_tokens += prompt_eval_count
            self.cum_completion_tokens += eval_count

            return content, prompt_eval_count, eval_count
        finally:
            conn.close()


DEFAULT_MESSAGE = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": None},
]


def chat_gpt_wrapper(**args) -> Union[None, List[str]]:
    """
    A wrapper for openai.ChatCompletion.create() that retries 10 times if it fails.

    Parameters
    ----------
    **args
        The arguments to pass to openai.ChatCompletion.create(). This includes things like the prompt, the model, temperature, etc.

    Returns
    -------
    List[str]
        The list of responses from the API.
    """

    if args.get("messages") is None:
        args["messages"] = deepcopy(DEFAULT_MESSAGE)
        args["messages"][1]["content"] = args["prompt"]
        del args["prompt"]

    for _ in range(10):
        try:
            model_name = args.get("model", "gpt-oss-20b")
            llm = OuterMedusaLLM(model=model_name)
            
            # Extract system prompt if present
            sys_prompt = None
            user_prompt = ""
            
            for msg in args["messages"]:
                if msg["role"] == "system":
                    sys_prompt = msg["content"]
                elif msg["role"] == "user":
                    user_prompt = msg["content"]

            content, prompt_tokens, completion_tokens = llm.generate(prompt=user_prompt, sys_prompt=sys_prompt)
            print(f"Generated content: {content[:50]}...")
            return [content]
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            print(e)
            time.sleep(10)

    return None


def estimate_querying_cost(
    num_prompt_toks: int, num_completion_toks: int, model: str
) -> float:
    """
    Estimate the cost of running the API, as of 2023-04-06.

    Parameters
    ----------
    num_prompt_toks : int
        The number of tokens in the prompt.
    num_completion_toks : int
        The number of tokens in the completion.
    model : str
        The model to be used.

    Returns
    -------
    float
        The estimated cost of running the API.
    """

    if model == "gpt-3.5-turbo":
        cost_per_prompt_token = 0.0005 / 1000
        cost_per_completion_token = 0.0015 / 1000
    elif model == "gpt-4":
        cost_per_prompt_token = 0.03 / 1000
        cost_per_completion_token = 0.06 / 1000
    elif model == "gpt-4-32k":
        cost_per_prompt_token = 0.06 / 1000
        cost_per_completion_token = 0.12 / 1000
    elif model.startswith("text-davinci-"):
        cost_per_prompt_token = 0.02 / 1000
        cost_per_completion_token = 0.02 / 1000
    elif model == "gpt-oss:20b":
        cost_per_prompt_token = 0
        cost_per_completion_token = 0
    else:
        # Defaults for unknown models or assume 0
        cost_per_prompt_token = 0
        cost_per_completion_token = 0

    cost = (
        num_prompt_toks * cost_per_prompt_token
        + num_completion_toks * cost_per_completion_token
    )
    return cost


class ChatGPTWrapperWithCost:
    """
    A class for openai.ChatCompletion.create() that retries when and records the cost of the API.
    """
    
    # Class-level variables to track usage across all instances
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0

    def __init__(self):
        self.num_queries = 0
        self.num_tokens = 0
        self.cost = 0.0

    @classmethod
    def get_global_usage(cls):
        return {
            "total_prompt_tokens": cls.total_prompt_tokens,
            "total_completion_tokens": cls.total_completion_tokens,
            "total_cost": cls.total_cost
        }

    def __call__(self, **args) -> Union[None, List[str]]:
        """
        A wrapper regarding OuterMedusaLLM that behaves like previous ChatGPT wrapper.

        Parameters
        ----------
        **args
            The arguments to pass.

        Returns
        -------
        List[str]
            The list of responses from the API.
        """

        if args.get("messages") is None:
            args["messages"] = deepcopy(DEFAULT_MESSAGE)
            args["messages"][1]["content"] = args["prompt"]
            del args["prompt"]

        for _ in range(10):
            try:
                model_name = args.get("model", "gpt-oss-20b")
                # Ensure model name is compatible if needed, or just pass it through
                
                llm = OuterMedusaLLM(model=model_name)
                
                 # Extract system prompt if present
                sys_prompt = None
                user_prompt = ""
                
                for msg in args["messages"]:
                    if msg["role"] == "system":
                        sys_prompt = msg["content"]
                    elif msg["role"] == "user":
                        user_prompt = msg["content"]
                
                content, prompt_tokens, completion_tokens = llm.generate(prompt=user_prompt, sys_prompt=sys_prompt)
                
                self.num_queries += 1
                self.num_tokens += (prompt_tokens + completion_tokens)
                
                # Calculate cost for this call
                current_cost = estimate_querying_cost(
                    prompt_tokens,
                    completion_tokens,
                    model_name,
                )
                self.cost += current_cost
                
                # Update global stats
                ChatGPTWrapperWithCost.total_prompt_tokens += prompt_tokens
                ChatGPTWrapperWithCost.total_completion_tokens += completion_tokens
                ChatGPTWrapperWithCost.total_cost += current_cost

                # Compatibility: return list of content
                return [content]
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                print(e)
                time.sleep(10)

        return None


def gpt3wrapper(max_repeat=20, **arguments) -> Union[None, object]:
    """
    A wrapper for using OuterMedusaLLM in a completion-like style.
    NOTE: Return type changed from openai object to dict-like object to maintain minimum compat if needed, 
    but mainly just returns a mock object with 'choices' if possible or simple string if usages allow.
    However, seeing gpt3wrapper_texts below, it expects an object with .choices[0].text or .choices[].text
    
    Let's return a simple class that mimics the structure needed.
    """
    
    class MockChoice:
        def __init__(self, text):
            self.text = text
            
    class MockResponse:
        def __init__(self, texts):
            self.choices = [MockChoice(t) for t in texts]

    i = 0
    while i < max_repeat:
        try:
            model_name = arguments.get("model", "gpt-oss-20b")
            llm = OuterMedusaLLM(model=model_name)
            
            prompts = arguments.get("prompt")
            if isinstance(prompts, str):
                prompts = [prompts]
                
            results = []
            for p in prompts:
                content, _, _ = llm.generate(prompt=p)
                results.append(content)
            
            return MockResponse(results)

        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            print(arguments.get("prompt", ""))
            print(e)
            print("now sleeping")
            time.sleep(30)
            i += 1
    return None


def gpt3wrapper_texts(max_repeat=20, **arguments) -> Union[None, str, List[str]]:
    """
    A wrapper for openai.Completion.create() that returns the text of the response.

    Parameters
    ----------
    max_repeat : int, optional
        The maximum number of times to retry the API call, by default 20
    **arguments
        The arguments to pass to openai.Completion.create(). This includes things like the prompt, the model, temperature, etc.

    Returns
    -------
    Union[None, str, List[str]]
        The text of the response. If the prompt is a list, then the response is a list of strings. Otherwise, it is a single string. If the API call fails, then None is returned.
    """

    response = gpt3wrapper(max_repeat=max_repeat, **arguments)
    if response is None:
        return None
    if type(arguments["prompt"]) == list:
        return [r.text for r in response.choices]
    else:
        return response.choices[0].text


def gpt3wrapper_texts_batch_iter(max_repeat=20, bsize=20, verbose=False, **arguments):
    """
    A wrapper for gpt3wrapper_texts that batches the prompts.

    Parameters
    ----------
    max_repeat : int, optional
        The maximum number of times to retry the API call, by default 20
    bsize : int, optional
        The batch size, by default 20
    verbose : bool, optional
        Whether to print a progress bar, by default False
    **arguments
        The arguments to pass to gpt3wrapper_texts. This includes things like the prompt, the model, temperature, etc.

    Yields
    -------
    str
        The response from the API.
    """

    # make sure the prompt is a list
    prompt = arguments["prompt"]
    assert type(prompt) == list

    # batch the prompts
    num_batches = (len(prompt) - 1) // bsize + 1
    iterator = trange(num_batches) if verbose else range(num_batches)
    for i in iterator:
        arg_copy = deepcopy(arguments)
        arg_copy["prompt"] = prompt[i * bsize : (i + 1) * bsize]

        # make the API call
        response = gpt3wrapper(max_repeat=max_repeat, **arg_copy)

        # yield the response
        if response is None:
            for _ in range(len(arg_copy["prompt"])):
                yield None
        else:
            for text in [r.text for r in response.choices]:
                yield text


def parse_description_responses(response: str) -> List[str]:
    """
    Parse the description responses from the proposer model.

    Parameters
    ----------
    response : str
        The response from the proposer model, each description is separated by a newline, surrounded by quotes. We will extract the description within the quotes for each line.

    Returns
    -------
    List[str]
        A list of descriptions.
    """
    descriptions = []
    for line_id, line in enumerate(response.split("\n- ")):
        # find the two quotes
        start, end = (line.find('"') if line_id != 0 else -1), line.rfind('"')
        description = line[start + 1 : end]
        if description != "":
            descriptions.append(description)

    return descriptions


def get_context_length(model: str) -> int:
    """
    Get the context length for the given model.

    Parameters
    ----------
    model : str
        The model in the API to be used.

    Returns
    -------
    int
        The context length.
    """

    if model in ("text-davinci-002", "text-davinci-003"):
        return 4096
    if model == "gpt-4":
        return 8000
    elif model == "gpt-4-32k":
        return 32000
    elif model == "gpt-3.5-turbo":
        return 4096
    elif model in ["gpt-oss:20b", "gpt-oss-20b"]:
        return 2048
    else:
        raise ValueError(f"Unknown model {model}")


def get_length_in_gpt2_tokens(text: str) -> int:
    """
    Get the length of a text in GPT2 tokens.

    Parameters
    ----------
    text : str
        The text.

    Returns
    -------
    int
        The length of the text in GPT2 tokens.
    """
    return len(GPT2TOKENIZER.encode(text))


def get_avg_length(texts: List[str], max_num_samples=500) -> float:
    """
    Get the average length of texts in a list of texts.

    Parameters
    ----------
    texts : List[str]
        A list of texts.
    max_num_samples : int
        The maximum number of texts to sample to compute the average length.

    Returns
    -------
    float
        The average length of texts.
    """
    if len(texts) > max_num_samples:
        sampled_texts = random.sample(texts, max_num_samples)
    else:
        sampled_texts = texts
    avg_length = np.mean([get_length_in_gpt2_tokens(t) for t in sampled_texts])
    return avg_length


def parse_template(template: str) -> str:
    """
    A helper function to parse the template, which can be either a string or a path to a file.
    """
    if os.path.exists(template):
        with open(template, "r") as f:
            return f.read()
    else:
        return template


# hyperparameters
# in expectation the prompt will have the length (CONTEXT_LENGTH - CORPUS_OVERHEAD) * (1 - CORPUS_BUFFER_FRACTION) to leave room for the overflow and the completion
CORPUS_OVERHEAD = 1024
CORPUS_BUFFER_FRACTION = 0.25


def get_max_num_samples_in_proposer(texts: List[str], proposer_model: str) -> int:
    """
    Get the maximal number of in-context samples based on the context length.Leave a buffer of 25% of the relative context length and 1024 tokens for the absolute context length

    Parameters
    ----------
    texts : List[str]
        A list of texts.

    proposer_model : str
        The model used to propose descriptions.

    Returns
    -------
    int
        The maximal number of in-context samples.
    """
    max_corpus_pair_length = (get_context_length(proposer_model) - CORPUS_OVERHEAD) * (
        1 - CORPUS_BUFFER_FRACTION
    )
    avg_length = get_avg_length(texts)
    max_num_samples = int(max_corpus_pair_length / avg_length)
    return max_num_samples

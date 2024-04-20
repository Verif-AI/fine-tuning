import json
import logging

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

class Llama2Wrapper:
    def __init__(self, client=None) -> None:
        self.client = client

    def invoke_with_text(self, prompt):
        # init llama2 runtime client
        client = self.client or boto3.client(
            service_name = "bedrock-runtime",
            region_name = "us-east-1"
        )

        # invoke llama2 with the text prompt
        base_model_id = "meta.llama2-70b-chat-v1"
        provisioned_custom_model_arn = "arn:aws:bedrock:us-east-1:992382490885:custom-model/meta.llama2-70b-v1:0:4k/k09t52jlvdza"

        try:
            response = client.invoke_model(
                modelId=provisioned_custom_model_arn,
                modelId=base_model_id,
                body=json.dumps(
                    {
                        "prompt": prompt,
                        "max_gen_len": 512,
                        "temperature": 0.5,
                        "top_p": 0.9,
                    }
                ),
            )

            # process and print the response
            result = json.loads(response.get("body").read())
          
            input_tokens = result["prompt_token_count"]
            output_tokens = result["generation_token_count"]
            output = result["generation"]

            #print("Invocation details:")
            #print(f"- The input length is {input_tokens} tokens.")
            #print(f"- The output length is {output_tokens} tokens.")

            #print(f"- The model returned 1 response(s):")
            print(output)

            return output

        except ClientError as err:
            logger.error(
                "Couldn't invoke Llama 2 Chat 70B. Here's why: %s: %s",
                err.response["Error"]["Code"],
                err.response["Error"]["Message"],
            )
            raise

def usage_demo():
    print("-" * 88)
    print("Welcome to the Amazon Bedrock Runtime demo with Llama 2 Chat 70B.")
    print("-" * 88)

    client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
    wrapper = Llama2Wrapper(client)

    # invoke Llama 2 Chat 70B with a text prompt
    text_prompt = "Hi, write a short sentence about yourself"
    print(f"Invoking Llama 2 Chat 70B with '{text_prompt}'...")
    wrapper.invoke_with_text(text_prompt)
    print("-" * 88)

def verify_model():
    print("-" * 88)
    print("Verify Verify.AI model with Llama 2 Chat 70B.")
    print("-" * 88)

    client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
    wrapper = Llama2Wrapper(client)

    # test
    TESTING_DATA = "data/politifact_for_fine_tuning_testing.jsonl"
    with open(TESTING_DATA, encoding="utf-8") as file:
        for idx, doc in enumerate(file):
            json_doc = json.loads(doc)
            text_prompt = json_doc["prompt"] + ", please give an answer TRUE or FALSE."
            #print(f"Invoking Llama 2 Chat 70B with '{text_prompt}'...")
            completion = json_doc["completion"]
            output = wrapper.invoke_with_text(text_prompt)
            print("No : {}, Model answer : {}, Correct answer : {}".format(idx + 1, output, completion))
            print("-" * 88)

if __name__ == "__main__":
    #usage_demo()
    verify_model()
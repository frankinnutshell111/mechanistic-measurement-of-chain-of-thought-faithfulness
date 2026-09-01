import json

from transformers import AutoTokenizer


MODEL_ID = "Qwen/Qwen3-14B"
JSONL_PATH = "results/paired_dataset1.jsonl"


def decode_generated_text(tokenizer, result):
    if "generated_tokens" in result:
        generated_tokens = result["generated_tokens"]
    elif "full_tokens" in result and "prompt_tokens" in result:
        generated_tokens = result["full_tokens"][len(result["prompt_tokens"]):]
    else:
        raise KeyError(
            "Result has no generated_tokens or full_tokens/prompt_tokens to decode."
        )

    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

with open(JSONL_PATH, "r", encoding="utf-8") as file:
    for index, line in enumerate(file, start=1):
        data = json.loads(line)

        print(f"RECORD {index} | ID: {data['id']}")
        print("\nOriginal generation:\n")
        print(decode_generated_text(tokenizer, data["results"]))
        print("\nHinted generation:\n")
        print(decode_generated_text(tokenizer, data["hinted_results"]))
        print("\n" + "=" * 100 + "\n")

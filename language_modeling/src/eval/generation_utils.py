import torch


def generate_greedy_with_kv_cache(model, 
                                input_ids,
                                max_new_tokens=512,
):
    """
    Generates text using greedy decoding with key-value cache support.

    Args:
        model (PreTrainedModel): The language model (LLM) used for generation.
        input_ids(torch.Tensor): [Batch_size, question_len] question tensor.
    Returns:
        str: The generated text.
    """
    # store the generated tokens 
    generated_ids = input_ids
    past_key_values = None

    with torch.no_grad():
        # 1. Prefill stage
        outputs = model(input_ids=input_ids,
                         use_cache=True, 
                         past_key_values=None)
        logits = outputs.logits
        past_key_values = outputs.past_key_values

        # Greedy decoding.
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

        # 2. Generation loop: generate tokens one at a time using key-value cache
        for _ in range(1, max_new_tokens):
            outputs = model(input_ids=next_token_id,
                            past_key_values=past_key_values,
                            use_cache=True)
            logits = outputs.logits
            past_key_values = outputs.past_key_values
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)

            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

            # stopping condition
            if next_token_id == model.config.eos_token_id:
                break

    return generated_ids
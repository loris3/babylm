import torch
SFT_MESSAGE_KEY = "messages"
INPUT_IDS_KEY = "input_ids"
ATTENTION_MASK_KEY = "attention_mask"
LABELS_KEY = "labels"

def sft_tulu_tokenize_and_truncate_v1(row, tokenizer, max_seq_length=4096):
    """taken more or less directly from https://github.com/allenai/open-instruct/blob/main/open_instruct/finetune.py
       (who took it directly from https://github.com/allenai/open-instruct/blob/ba11286e5b9eb00d4ce5b40ef4cac1389888416a/open_instruct/finetune.py#L385)"""
    messages = row["messages"]

    if len(messages) == 0:
        raise ValueError("messages field is empty.")
    input_ids = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=True,
        return_tensors="pt",
        return_special_tokens_mask=True, truncation=True, padding="max_length", max_length=4096,
        add_generation_prompt=False,
    )
    labels = input_ids.clone()
    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            # we calculate the start index of this non-assistant message
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer.apply_chat_template(
                    conversation=messages[:message_idx],  # here marks the end of the previous messages
                    tokenize=True,
                    return_tensors="pt",
                     return_special_tokens_mask=True, truncation=True, padding="max_length", max_length=4096,
                    add_generation_prompt=False,
                ).shape[1]
            # next, we calculate the end index of this non-assistant message
            if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant":
                # for intermediate messages that follow with an assistant message, we need to
                # set `add_generation_prompt=True` to avoid the assistant generation prefix being included in the loss
                # (e.g., `<|assistant|>`)
                message_end_idx = tokenizer.apply_chat_template(
                    conversation=messages[: message_idx + 1],
                    tokenize=True,
                    return_tensors="pt",
                     return_special_tokens_mask=True, truncation=True, padding="max_length", max_length=4096,
                    add_generation_prompt=True,
                ).shape[1]
            else:
                # for the last message or the message that doesn't follow with an assistant message,
                # we don't need to add the assistant generation prefix
                message_end_idx = tokenizer.apply_chat_template(
                    conversation=messages[: message_idx + 1],
                    tokenize=True,
                    return_tensors="pt",
                     return_special_tokens_mask=True, truncation=True, padding="max_length", max_length=4096,
                    add_generation_prompt=False,
                ).shape[1]
            # set the label to -100 for the non-assistant part
            labels[:, message_start_idx:message_end_idx] = -100
            if max_seq_length and message_end_idx >= max_seq_length:
                break
    attention_mask = torch.ones_like(input_ids)

    # changes: no .flatten()
    row[INPUT_IDS_KEY] = input_ids
    row[LABELS_KEY] = labels
    row[ATTENTION_MASK_KEY] = attention_mask

    return row

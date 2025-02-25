from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers_interpret import SequenceClassificationExplainer

if __name__ == "__main__":
    text_to_check = "Testowany tekst, bardzo długi i skomplikowany"
    model_name = "sdadas/polish-roberta-large-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained("./SCEIZKA_DO_WYTRENOWANEGO_MODELU")

    # Create the explainer
    cls_explainer = SequenceClassificationExplainer(
        model=model,
        tokenizer=tokenizer
    )

    # In case of long texts, sentence explainer cannot process them at once
    encoding = tokenizer(
        text_to_check,
        max_length=512,
        truncation=True,
        stride=256,
        return_overflowing_tokens=True,
        padding="max_length",
        return_tensors="pt"
    )

    n_chunks = encoding["input_ids"].shape[0]
    print(f"Text was split into {n_chunks} chunks.")

    # Process each chunk with the explainer
    all_chunk_attributions = []
    for i in range(n_chunks):
        # Previous attempt
        # chunk_text = tokenizer.decode(encoding["input_ids"][i], skip_special_tokens=False)
        #reencoded = tokenizer(chunk_text, max_length=514, truncation=True, padding="max_length", return_tensors="pt")

        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][i])
        chunk_text = tokenizer.convert_tokens_to_string(tokens)

        print(f"\n--- Attributions for Chunk {i + 1} ---")
        # Pass the reconstructed text to the explainer.
        chunk_attributions = cls_explainer(chunk_text)
        all_chunk_attributions.append(chunk_attributions)
        print(chunk_attributions)
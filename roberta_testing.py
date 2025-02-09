from typing import List
import sys
from dao.attribute import DAOAttributePL
from models.attribute import AttributePLInDB
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers_interpret import SequenceClassificationExplainer

if __name__ == "__main__":
    try:
        text_id = str(sys.argv[1])
        if len(text_id) < 5:
            print("ObjectID too short")
            exit(1)
    except IndexError:
        print("ObjectID was not provided")
        exit(1)

    dao_attribute: DAOAttributePL = DAOAttributePL(collection_name="attributes-24-12-16-recalc-24-12-22.1-pgryka")
    selected_attribute: AttributePLInDB = dao_attribute.find_by_id(text_id)
    if selected_attribute is None:
        print(f"No object found with {text_id} ID")
        exit(1)
    text_to_check = selected_attribute.stylometrix_metrics.text
    model_name = "sdadas/polish-roberta-large-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained("./final_results")

    # Create the explainer
    cls_explainer = SequenceClassificationExplainer(
        model=model,
        tokenizer=tokenizer
    )

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
        # Decode the i-th chunk (this gives you a text representation of the already-tokenized chunk)
        chunk_text = tokenizer.decode(encoding["input_ids"][i], skip_special_tokens=True)

        # === IMPORTANT: Re-tokenize with explicit truncation and padding ===
        # This step ensures that when the explainer tokenizes the text internally, it uses
        # the same settings as during training. Otherwise, the explainer’s default tokenization
        # may produce a sequence length (e.g. 523 tokens) that mismatches what the model expects.
        inputs = tokenizer(
            chunk_text,
            max_length=tokenizer.model_max_length,  # or use 512 if that was your training setting
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        # Decode back to text so that when the explainer tokenizes it, it yields exactly the same tokens.
        normalized_chunk_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

        print(f"\n--- Attributions for Chunk {i + 1} ---")
        # Pass the normalized text to the explainer
        chunk_attributions = cls_explainer(normalized_chunk_text)
        all_chunk_attributions.append(chunk_attributions)
        print(chunk_attributions)
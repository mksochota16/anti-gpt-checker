from typing import List
from dao.attribute import DAOAttributePL
from models.attribute import AttributePLInDB
from datasets import Dataset
import torch
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForSequenceClassification


def tokenize_function(example):
    encoding = tokenizer(
        example['text'],
        max_length=512,
        truncation=True,
        stride=256,
        return_overflowing_tokens=True,
        padding="max_length",
        return_tensors="pt"
    )
    n_chunks = encoding["input_ids"].shape[0]

    return {
        "input_ids": [encoding["input_ids"][i].tolist() for i in range(n_chunks)],
        "attention_mask": [encoding["attention_mask"][i].tolist() for i in range(n_chunks)],
        "label": [example["label"]] * n_chunks
    }
def tokenize_function_batch(batch):
    """
    Processes a batch of examples and tokenizes each text using a sliding window.
    Each long text may produce several chunks; this function flattens them so that
    every chunk becomes a separate example with its own scalar label.

    Args:
        batch (dict): A dictionary with keys "text" and "label", where each value is a list.

    Returns:
        dict: A dictionary with keys "input_ids", "attention_mask", and "label",
              where each value is a list of length equal to the total number of chunks produced.
    """
    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    # Loop over each example in the batch.
    for text, label in zip(batch["text"], batch["label"]):
        # Use the built-in sliding window functionality.
        encoding = tokenizer(
            text,
            max_length=512,
            truncation=True,
            stride=256,
            return_overflowing_tokens=True,
            padding="max_length",
            return_tensors="pt"
        )
        # Number of chunks produced for this text.
        num_chunks = encoding["input_ids"].shape[0]
        # For each chunk, add its token ids, attention mask, and the same scalar label.
        for i in range(num_chunks):
            all_input_ids.append(encoding["input_ids"][i].tolist())
            all_attention_masks.append(encoding["attention_mask"][i].tolist())
            all_labels.append(label)

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "label": all_labels
    }

def compute_metrics(eval_pred):
    """Compute accuracy or other metrics after each evaluation."""
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    accuracy = (predictions == labels).float().mean()
    return {"accuracy": accuracy.item()}

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Rename 'label' to 'labels' if present
        if 'label' in inputs:
            inputs['labels'] = inputs.pop('label')
        return super().compute_loss(model, inputs, return_outputs=return_outputs)


if __name__ == "__main__":
    dao_attribute: DAOAttributePL = DAOAttributePL(collection_name="attributes-24-12-16-recalc-24-12-22.1-pgryka")

    attributes_generated: List[AttributePLInDB] = dao_attribute.find_many_by_query({"is_generated": True})
    attributes_real: List[AttributePLInDB] = dao_attribute.find_many_by_query({"is_generated": False})

    dicts_generated = [{"text": attribute.stylometrix_metrics.text, "label": 1} for attribute in attributes_generated]
    dicts_real = [{"text": attribute.stylometrix_metrics.text, "label": 0} for attribute in attributes_real]
    combined = dicts_generated + dicts_real
    dataset_whole = Dataset.from_list(combined)
    split_dataset = dataset_whole.train_test_split(test_size=0.3)

    # Extract the train and test subsets
    train_dataset_before_tokenizer = split_dataset["train"]
    test_dataset_before_tokenizer = split_dataset["test"]

    model_name = "sdadas/polish-roberta-large-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # For parallel
    # model = AutoModelForMaskedLM.from_pretrained(model_name, num_labels=2)
    # For sequential
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)



    train_dataset = train_dataset_before_tokenizer.map(tokenize_function_batch, batched=True, remove_columns=["text"])
    test_dataset = test_dataset_before_tokenizer.map(tokenize_function_batch, batched=True, remove_columns=["text"])

    train_dataset = train_dataset.with_format("torch")
    test_dataset = test_dataset.with_format("torch")




    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,  # MLM stands for masked language modeling
        mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir="./results2",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        load_best_model_at_end=True,
    )


    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    results = trainer.evaluate(test_dataset)
    print(results)
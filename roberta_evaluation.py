#!/usr/bin/env python
import torch
from typing import List
from dao.attribute import DAOAttributePL
from models.attribute import AttributePLInDB
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


def predict_text(model, tokenizer, text):
    """
    Tokenize the input text with a sliding window (using stride) and
    aggregate the logits from all produced chunks to generate a single prediction.
    """
    encoding = tokenizer(
        text,
        max_length=512,
        truncation=True,
        stride=256,
        return_overflowing_tokens=True,
        padding="max_length",
        return_tensors="pt"
    )

    # Extract input_ids and attention_mask for all chunks
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # outputs.logits has shape (num_chunks, num_labels)
    logits = outputs.logits
    # Average logits over all chunks to obtain a single set of logits for the text
    avg_logits = logits.mean(dim=0, keepdim=True)
    probabilities = torch.softmax(avg_logits, dim=-1)
    predicted_label = torch.argmax(probabilities, dim=-1).item()
    return predicted_label, probabilities.squeeze().tolist()


def main():
    # Specify the directory where the trained model was saved.
    model_path = "./final_results"

    # Load the tokenizer and model from the saved directory.
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()  # Set the model to evaluation mode.

    dao_attribute: DAOAttributePL = DAOAttributePL(collection_name="attributes-24-12-16-recalc-24-12-22.1-pgryka")

    attributes_generated: List[AttributePLInDB] = dao_attribute.find_many_by_query({"is_generated": True})
    attributes_real: List[AttributePLInDB] = dao_attribute.find_many_by_query({"is_generated": False})

    dicts_generated = [{"text": attribute.stylometrix_metrics.text, "label": 1} for attribute in attributes_generated]
    dicts_real = [{"text": attribute.stylometrix_metrics.text, "label": 0} for attribute in attributes_real]

    test_data = dicts_generated[:100] + dicts_real[:100]

    true_labels = []
    pred_labels = []

    # Process each sample and collect predictions.
    for sample in test_data:
        text = sample["text"]
        true_label = sample["label"]

        predicted_label, probabilities = predict_text(model, tokenizer, text)
        true_labels.append(true_label)
        pred_labels.append(predicted_label)

        print(f"Input text: {text}")
        print(f"True label: {true_label}")
        print(f"Predicted label: {predicted_label}")
        print(f"Probabilities: {probabilities}")
        print("-" * 50)

    # Compute basic metrics using scikit-learn.
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average="weighted", zero_division=0)
    recall = recall_score(true_labels, pred_labels, average="weighted", zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)

    print("Evaluation Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, zero_division=0))


if __name__ == "__main__":
    main()

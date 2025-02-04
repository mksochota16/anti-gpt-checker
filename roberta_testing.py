from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers_interpret import SequenceClassificationExplainer


tokenizer = RobertaTokenizer.from_pretrained("./final_results")
model = RobertaForSequenceClassification.from_pretrained("./results")

# Create the explainer
cls_explainer = SequenceClassificationExplainer(
    model=model,
    tokenizer=tokenizer
)

text = "To jest przykładowe zdanie do sprawdzenia."

# Get word attributions
word_attributions = cls_explainer(text)

# word_attributions is a list of tuples: [(token_1, attribution_score_1), (token_2, ...), ...]
print(word_attributions)

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

    # Get word attributions
    word_attributions = cls_explainer(text_to_check)

    # word_attributions is a list of tuples: [(token_1, attribution_score_1), (token_2, ...), ...]
    print(word_attributions)

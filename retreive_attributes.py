from config import init_all_polish_models
from analysis.nlp_transformations import preprocess_text
from analysis.attribute_retriving import perform_full_analysis

if __name__ == "__main__":
    init_all_polish_models()

    test_text = """
    W cichym miasteczku Eldridge jesień zaczynała malować ulice w odcieniach bursztynu i złota.  """

    text_to_analyse = preprocess_text(test_text)
    analysis_result = perform_full_analysis(text_to_analyse, 'pl')

    print(analysis_result.dict())

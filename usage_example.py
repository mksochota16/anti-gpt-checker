from config import init_all_polish_models

from analysis.attribute_retriving import perform_full_analysis
from analysis.nlp_transformations import preprocess_text
from services.utils import suppress_stdout

if __name__ == "__main__":
    init_all_polish_models()
    text_to_analyse = "THIS IS A SAMPLE TEXT TO ANALYZE"

    text_to_analyse = preprocess_text(text_to_analyse)
    with suppress_stdout():
        analysis_result = perform_full_analysis(text_to_analyse, 'pl')

    print(analysis_result)
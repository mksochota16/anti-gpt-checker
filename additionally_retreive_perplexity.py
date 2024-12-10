from config import init_polish_perplexity_model

from typing import List
from tqdm import tqdm

from dao.attribute import DAOAttributePL

from models.attribute import AttributePLInDB

from analysis.attribute_retriving import calculate_perplexity
from services.utils import suppress_stdout

if __name__ == "__main__":
    init_polish_perplexity_model()
    dao_attributes = DAOAttributePL()
    # select only those attributes that have not been processed yet
    not_processed_perplexity: List[AttributePLInDB] = dao_attributes.find_many_by_query({'perplexity': None})

    for attribute_in_db in tqdm(not_processed_perplexity, total=len(not_processed_perplexity),
                                desc=f'Calculating perplexity of lab reports', unit='Lab reports', miniters=1):
        text_to_analyse = attribute_in_db.stylometrix_metrics.text
        with suppress_stdout():
            perplexity_base, perplexity = calculate_perplexity(text_to_analyse, 'pl', return_both=True, force_use_cpu=True)
        dao_attributes.update_one({'_id': attribute_in_db.id},
                                  {"$set": {'perplexity_base': perplexity_base, 'perplexity': perplexity}})

from config import init_polish_perplexity_model, init_spacy_polish_nlp_model, init_language_tool_pl

from typing import List
from tqdm import tqdm

from dao.lab_report import DAOLabReport
from dao.attribute import DAOAttributePL

from models.lab_report import LabReportInDB
from models.attribute import AttributePL, AttributePLInDB

from analysis.attribute_retriving import perform_full_analysis
from analysis.nlp_transformations import remove_report_tags, replace_whitespaces, replace_links_with_text
from services.utils import suppress_stdout

if __name__ == "__main__":
    #init_polish_perplexity_model()
    init_spacy_polish_nlp_model()
    init_language_tool_pl()
    dao_lab_reports = DAOLabReport()
    dao_attributes = DAOAttributePL()
    real_lab_reports: List[LabReportInDB] = dao_lab_reports.find_many_by_query({'is_generated': False})
    generated_lab_reports: List[LabReportInDB] = dao_lab_reports.find_many_by_query({'is_generated': True})
    alreadyprocessed_lab_reports = dao_attributes.find_many_by_query({})
    alreadyprocessed_lab_reports_ids = [report.referenced_doc_id for report in alreadyprocessed_lab_reports]

    real_lab_reports = [report for report in real_lab_reports if report.id not in alreadyprocessed_lab_reports_ids]
    generated_lab_reports = [report for report in generated_lab_reports if
                             report.id not in alreadyprocessed_lab_reports_ids]

    for real_lab_report in tqdm(real_lab_reports, total=len(real_lab_reports),
                                desc=f'Calculating real lab reports statistics', unit='Lab reports', miniters=1):
        text_to_analyse = remove_report_tags(real_lab_report.plaintext_content)
        text_to_analyse = replace_whitespaces(text_to_analyse)
        text_to_analyse = replace_links_with_text(text_to_analyse, replacement="")
        with suppress_stdout():
            analysis_result = perform_full_analysis(text_to_analyse, 'pl')
        attribute_to_insert = AttributePL(
            referenced_db_name='lab_reports',
            referenced_doc_id=real_lab_report.id,
            language="pl",
            is_generated=False,
            is_personal=None,
            **analysis_result.dict()
        )
        dao_attributes.insert_one(attribute_to_insert)

    for generated_lab_report in tqdm(generated_lab_reports, total=len(generated_lab_reports),
                                     desc=f'Calculating generated lab reports statistics', unit='Lab reports',
                                     miniters=1):
        text_to_analyse = remove_report_tags(generated_lab_report.plaintext_content)
        text_to_analyse = replace_whitespaces(text_to_analyse)
        text_to_analyse = replace_links_with_text(text_to_analyse, replacement="")
        with suppress_stdout():
            analysis_result = perform_full_analysis(text_to_analyse, 'pl')
        attribute_to_insert = AttributePL(
            referenced_db_name='lab_reports',
            referenced_doc_id=generated_lab_report.id,
            language="pl",
            is_generated=True,
            is_personal=None,
            **analysis_result.dict()
        )
        dao_attributes.insert_one(attribute_to_insert)
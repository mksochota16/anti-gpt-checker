import string
from typing import Tuple, List

import re

import nltk
from nltk.corpus import stopwords

from config import RELATIVE_PATH_TO_PROJECT, MINIMAL_SENTENCE_LENGTH, SUSPICIOUS_SENTENCE_LENGTH, \
    MAXIMAL_SENTENCE_LENGTH


def build_default_charset() -> set[str]:
    """
    Returns a set of characters allowed in LaTeX-friendly Polish text.
    Includes:
      - Polish alphabet (lowercase + uppercase)
      - Digits
      - Common ASCII letters
      - Whitespace
      - Common punctuation
      - LaTeX-friendly symbols (m-dash, n-dash, tilde, caret, etc.)
    """
    polish_lower = "ąćęłńóśźż"
    polish_upper = polish_lower.upper()
    ascii_letters = string.ascii_letters
    digits = string.digits
    whitespace = string.whitespace
    # Common punctuation and LaTeX-safe symbols
    punctuation = (
        ".,;:!?\"'()[]{}<>/\\|@#$%^&*_+=-~`–—―„”"
    )
    itemize_chars = "*-•\u2022\u2013\u2014"
    return set(polish_lower + polish_upper + ascii_letters + digits + whitespace + punctuation + itemize_chars)


POLISH_CHARSET = build_default_charset()

def filter_text(text: str) -> str:
    """
    Filters text to only include characters from charset.
    For each character not in charset:
      - If there is a space before or after, delete the character
        and leave one space.
      - If there is no space before or after, replace the character with one space.
    Consecutive spaces are collapsed.
    """
    out_chars = []
    indexes_to_remove = []
    n = len(text)

    for i, ch in enumerate(text):
        if ch in POLISH_CHARSET:
            out_chars.append(ch)
        else:
            # Check if there's a space before or after the invalid character
            left_space = (i > 0 and text[i - 1] == " ")
            right_space = (i + 1 < n and text[i + 1] == " ") or (i + 1 < n and text[i + 1] not in POLISH_CHARSET)

            # If there's a space on either side, just skip this character
            # (the existing space will be kept)
            if not (left_space or right_space):
                # No adjacent space, so replace with a space
                out_chars.append(" ")
            if left_space and right_space:
                indexes_to_remove.append(len(out_chars))

    # Collapse consecutive spaces
    result = []
    for i, ch in enumerate(out_chars):
        if i in indexes_to_remove:
            continue
        else:
            result.append(ch)

    return "".join(result)

def lemmatize_text(text: str, lang_code: str) -> Tuple[str, List[str]]:
    """
    Lemmatize the text using the appropriate Spacy NLP model
    :param text: text to be lemmatized
    :param lang_code: language of the text
    :return: str and list of lemmatized words
    """
    from config import SPACY_POLISH_NLP_MODEL, SPACY_ENGLISH_NLP_MODEL
    if lang_code == "pl" and SPACY_POLISH_NLP_MODEL is None:
        raise ValueError("Polish NLP model is not initialized")
    elif lang_code == "en" and SPACY_ENGLISH_NLP_MODEL is None:
        raise ValueError("English NLP model is not initialized")

    if lang_code == "pl":
        nlp = SPACY_POLISH_NLP_MODEL
    elif lang_code == "en":
        nlp = SPACY_ENGLISH_NLP_MODEL
    else:
        raise ValueError(f"Language {lang_code} is not supported")

    doc = nlp(text)
    lemma_list = [token.lemma_ for token in doc]
    lemma_text = " ".join(lemma_list)
    return lemma_text, lemma_list


def remove_stopwords_punctuation_emojis_and_splittings(lemmatize_text: str, lang_code: str) -> List[str]:
    """
    Remove stopwords and punctuation from the text
    :param lemmatize_text: lemmatize_text to be cleaned
    :param lang_code: language of the text
    :return: list of cleaned words
    """
    lemmatize_text = deemojify(lemmatize_text)
    lemmatize_text = remove_footers(lemmatize_text)
    lemmatize_text = re.sub(r'[=]+', ' ', lemmatize_text)
    lemmatize_text = re.sub(r'[-]+', ' ', lemmatize_text)
    lemmatize_text = re.sub(r'\d+', ' ', lemmatize_text)
    lemmatize_text = re.sub(r'[^\w\s]', '', lemmatize_text)
    tokens = lemmatize_text.split()
    if lang_code == "pl":
        # read polish stopwords from file
        with open(f"{RELATIVE_PATH_TO_PROJECT}static/polish.stopwords.txt", "r") as file:
            stop_words = file.read().splitlines()
    elif lang_code == "en":
        stop_words = set(stopwords.words('english'))
    else:
        raise ValueError(f"Language {lang_code} is not supported")
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
    return tokens


def clean_text(input_text: str) -> str:
    # Replace lines with only a dot or spaces with a blank line
    cleaned_text = re.sub(r'^[\s.]*$', '\n', input_text, flags=re.MULTILINE)
    # Replace multiple blank lines with a single blank line
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
    # Remove leading and trailing blank lines
    cleaned_text = cleaned_text.strip('\n')
    return cleaned_text


def deemojify(text: str) -> str:
    # Unicode ranges for emojis
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251" 
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r'', text)


def separate_previous_conversation(text: str) -> Tuple[str, str]:
    lines = text.split('\n')
    previous_messages = []
    response = []

    for line in lines:
        # Check if the line is a previous message or part of the response
        if line.startswith('>'):
            previous_messages.append(line)
        else:
            response.append(line)

    # Join the lists back into strings
    previous_text = '\n'.join(previous_messages)
    response_text = '\n'.join(response)

    return previous_text, response_text

def remove_footers(text: str) -> str:
    # Remove the footer
    return re.sub(r"\w{2}\., \d{1,2} \w{3} \d{4} o \d{2}:\d{2} .+ <.+@.+> napisał(a|\(a\)|):", "", text)


def replace_links_with_text(text: str, replacement: str="LINK") -> str:
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+|\bwww\.\S+\.\S+'
    replaced_text = re.sub(url_pattern, replacement, text)
    return replaced_text


def replace_IP_addresses_with_text(text: str, replacement: str = "ADRES INTERNETOWY") -> str:
    """
    Replace all IPv4, IPv6, and MAC addresses in text with a replacement string.
    Works across newlines and anywhere in the text.

    Args:
        text: Input text containing addresses
        replacement: String to replace addresses with (default: "ADRES INTERNETOWY")

    Returns:
        Text with all addresses replaced
    """

    # IPv4 pattern (with optional CIDR notation)
    # Matches standard IPv4 addresses like 192.168.1.1 or 10.0.0.0/24
    ipv4_pattern = r'\b(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.(?:25[0-5]|2[0-4]\d|[01]?\d?\d)(?:/(?:3[0-2]|[12]?\d))?\b'

    # IPv6 pattern (with optional CIDR notation /0-128)
    # Matches full and compressed IPv6 addresses
    ipv6_pattern = r'(?:\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b|' \
                   r'\b(?:[0-9a-fA-F]{1,4}:){1,7}:\b|' \
                   r'\b(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}\b|' \
                   r'\b(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}\b|' \
                   r'\b(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}\b|' \
                   r'\b(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}\b|' \
                   r'\b(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}\b|' \
                   r'\b[0-9a-fA-F]{1,4}:(?:(?::[0-9a-fA-F]{1,4}){1,6})\b|' \
                   r'\b:(?:(?::[0-9a-fA-F]{1,4}){1,7}|:)\b|' \
                   r'\bfe80:(?::[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}\b|' \
                   r'\b::(?:ffff(?::0{1,4}){0,1}:){0,1}(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\b|' \
                   r'\b(?:[0-9a-fA-F]{1,4}:){1,4}:(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\b)' \
                   r'(?:/(?:12[0-8]|1[01][0-9]|[1-9]?[0-9]))?'  # Optional CIDR /0-128

    # MAC address pattern
    mac_pattern = r'\b(?:[0-9a-fA-F]{2}[:-]){5}[0-9a-fA-F]{2}\b|' \
                  r'\b(?:[0-9a-fA-F]{4}\.){2}[0-9a-fA-F]{4}\b|' \
                  r'\b[0-9a-fA-F]{12}\b'

    # Replace all patterns
    text = re.sub(ipv4_pattern, replacement, text)
    text = re.sub(ipv6_pattern, replacement, text)
    text = re.sub(mac_pattern, replacement, text)

    return text


def split_into_sentences(text: str, lang_code: str) -> List[str]:
    if lang_code == "pl":
        sentence_tokenizer = nltk.data.load('tokenizers/punkt/polish.pickle')
    elif lang_code == "en":
        sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    else:
        raise ValueError(f"Language {lang_code} is not supported")

    sentences = sentence_tokenizer.tokenize(text)
    sentences_normal = [sentence for sentence in sentences if (MINIMAL_SENTENCE_LENGTH < len(sentence.split(" ")) <= SUSPICIOUS_SENTENCE_LENGTH)]
    sentence_split = []
    for sentence in [sentence for sentence in sentences if (SUSPICIOUS_SENTENCE_LENGTH < len(sentence.split(" ")))]:
        sentence_split.extend(split_text_on_regex_match(sentence))

    split_sentences = sentences_normal + sentence_split

    if len(split_sentences) == 0:
        split_sentences = [text]

    return split_sentences

def replace_meaningful_report_tags(text: str) -> str:
    # Replace tags with placeholders
    text = text.replace("<<Imię>>", "IMIĘ").replace("<<Nazwisko>>", "NAZWISKO").replace("<<adres e-mail>>", "ADRES E-MAIL")
    text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    return text
def remove_report_tags(text: str) -> str:
    # Remove the tags from the report
    tags = ["<<Imię>>", "<<Nazwisko>>", "<<nr albumu>>", "<<adres e-mail>>", "<<tabela>>", "<<obrazek>>", "<<obcy język>>"]
    for tag in tags:
        text = text.replace(tag, "")
    text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    return text

def replace_whitespaces(text: str) -> str:
    # Replace multiple whitespaces with a single whitespace
    return text.replace('\u200B', "")

def is_abbreviation(s: str) -> bool:
    # Optimization: Check if the first character is uppercase
    if not s or not s[0].isupper():
        return False
    # Regex to capture the initial part and optional trailing lowercase part.
    # Initial part: letters (uppercase/lowercase) and digits
    # Optional part: '-' followed by 1 to 3 lowercase letters
    pattern = re.compile(r'^([A-Za-z0-9]+)(?:-([a-z]{1,3}))?$')
    match = pattern.match(s)
    if not match:
        return False

    first_part = s.split('-')[0]

    # Count lowercase letters in the first part
    lowercase_count = sum(1 for c in first_part if c.islower())
    length = len(first_part)

    # Apply the lowercase letter rule:
    # If length > 2, lowercase_count must be < floor(length/2)
    # If length <= 2, no lowercase letters allowed
    if length > 2:
        if lowercase_count >= length / 2:
            return False
    else:
        if lowercase_count > 0:
            return False

    # If there's a trailing part, it's guaranteed by the regex to be lowercase 1-3 chars
    # No further checks needed for trailing part.

    return True

def remove_multiple_dots(text: str) -> str:
    # Matches sequences of 4+ dots with optional spaces between them.
    # This is common in listings at the start of a report.
    return re.sub(r'(\s)?\.(?:\s*\.){3,}', '', text)

def preprocess_text(text: str) -> str:
    text_to_analyse = replace_whitespaces(text)
    text_to_analyse = replace_meaningful_report_tags(text_to_analyse)
    text_to_analyse = remove_report_tags(text_to_analyse)
    text_to_analyse= replace_IP_addresses_with_text(text_to_analyse)
    text_to_analyse = replace_links_with_text(text_to_analyse, replacement="LINK")
    text_to_analyse = text_to_analyse.replace("\n ", " ")
    text_to_analyse = text_to_analyse.replace("-\n", "")
    text_to_analyse = remove_multiple_dots(text_to_analyse)
    text_to_analyse = filter_text(text_to_analyse)
    text_to_analyse = re.sub(r'\s+', ' ', text_to_analyse).strip() #Replace all kinds of whitespace with a single space.
    return text_to_analyse

POLISH_CHARS = r'[A-ząęóżźćńłśĄĘÓŹŻĆŃŁŚ]' # tu tylko testowałem czy są linie bez tekstu żadnego (były chyba nawet >10 charow jakie formułki matematyczne np.: "1/9  ≈ 0,(1), 0,077983 ≅ 0,1.")

CAPTION_LIKE = ['Zdj.', 'Wyc.', 'SS.', 'Dz.U.', 'Ad.', 'Wg.', 'Rys.', 'Rysunek', 'Zdjęcie', 'Zadanie', 'Część', 'Grafika', 'Schemat', 'Rys', 'Zrzut', 'Obrazek', 'Tabela', 'Wydruk', 'Tab']

CAPTION_ONLY_PATTERN = r'^(?i)(?:' + '|'.join(re.escape(item) for item in CAPTION_LIKE) + r')((\s*\d+)|(\s*\d+\.))*$' # to do łapania krótkich linijek tylko; np Rysunek 8. (...)

CAPTION_TEXT_PATTERN = r'(?:' + '|'.join(re.escape(item) for item in CAPTION_LIKE) + r')' # to używałem do krojenia

ITEMIZE_PATTERN  = r"(?:^|\s)\s*([\*\-\•\u2022\u2013\u2014])\s+"

ROMAN_NUMERALS_PATTERN = r"^[IVXLCDM]+\.?$" # to do łapania numeracji sekcji cyframi rzymskimi pisane; btw są też itemizey robione cyframi rzymskimi xD i) ii) iii) iv) itd

LAW_RELATED_PATTERN = r'^\d+\s([A-ząęóżźćńłśĄĘÓŹŻĆŃŁŚ]+\.)+[A-ząęóżźćńłśĄĘÓŹŻĆŃŁŚ]?\.?$' # to do krótkich linii z tymi śmiesznymi skórtami prawnymi

ALL_SPLIT_PATTERNS = re.compile(f"({CAPTION_TEXT_PATTERN}|{ITEMIZE_PATTERN}|{ROMAN_NUMERALS_PATTERN})")


def split_text_on_regex_match(text: str, pattern=ALL_SPLIT_PATTERNS):
    parts = []
    last_end = 0
    for match in list(re.finditer(pattern, text)):
        if match.start() > (last_end + 20):
            start_text = text[last_end:match.start()]
            if MINIMAL_SENTENCE_LENGTH < len(start_text.split(" ")) < MAXIMAL_SENTENCE_LENGTH:
                parts.append(start_text.strip())

        part_to_add = match.group(0) + ' ' + text[match.end():].split(match.group(0), 1)[0].strip().split('\n')[0].strip()
        if MINIMAL_SENTENCE_LENGTH < len(part_to_add.split(" ")) < MAXIMAL_SENTENCE_LENGTH:
            parts.append(part_to_add.strip())
        last_end = match.start() + len(part_to_add)

    if last_end + 20 < len(text):
        end_text = text[last_end:]
        if MINIMAL_SENTENCE_LENGTH < len(end_text.split(" ")) < MAXIMAL_SENTENCE_LENGTH:
            parts.append(text[last_end:].strip())

    return parts

def get_text_for_punctuation_analysis(text: str) -> str:
    file_extensions_pattern = re.compile(r'(\b[\w-]+)((\.\w+)+)')
    caption_related_punctuation_pattern = r'\b(?:' + '|'.join(re.escape(item) for item in CAPTION_LIKE) + r')\s+\d+[.:]+\s+'
    numbering_related_punctuation_pattern = r'\d+[.]\d+|\d+[.]|[.]\d+'
    hidden_files_punctuation_pattern = r'\s+(?:\.\w+)'

    cleaned_text = text
    cleaned_text = file_extensions_pattern.sub(r'\1', cleaned_text)
    cleaned_text = re.sub(caption_related_punctuation_pattern, '', cleaned_text)
    cleaned_text = re.sub(numbering_related_punctuation_pattern, '', cleaned_text)
    cleaned_text = re.sub(hidden_files_punctuation_pattern, '', cleaned_text)

    return cleaned_text

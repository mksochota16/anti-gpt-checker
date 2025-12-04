from typing import Optional

from pydantic import BaseModel

class GrammaticalFormsPL(BaseModel):
    G_N:Optional[float] = None  # Nouns
    G_V:Optional[float] = None  # Verbs
    G_ADJ:Optional[float] = None  # Adjectives
    G_ADV:Optional[float] = None  # Adverbs
    G_PRO:Optional[float] = None  # Pronouns
    G_PRO_PRS:Optional[float] = None  # Personal pronouns
    G_PRO_REL:Optional[float] = None  # Relative pronouns
    G_PRO_DEM:Optional[float] = None  # Demonstrative pronouns
    G_PRO_INT:Optional[float] = None  # Interrogative pronouns
    G_PRO_IND:Optional[float] = None  # Indefinite pronouns
    G_PRO_TOT:Optional[float] = None  # Total pronouns
    G_PRO_NEG:Optional[float] = None  # Negative pronouns
    G_PRO_POS:Optional[float] = None  # Possessive pronouns
    G_NUM:Optional[float] = None  # Numerals
    G_CNUM:Optional[float] = None  # Collective numerals
    G_PART:Optional[float] = None  # Particles
    G_ADP:Optional[float] = None  # Adpositions
    G_INTJ:Optional[float] = None  # Interjections
    G_SYM:Optional[float] = None  # Symbols
    G_ABBR:Optional[float] = None  # Abbreviations
    G_CONJ:Optional[float] = None  # Conjunctions
    G_CCONJ:Optional[float] = None  # Coordinating conjunctions
    G_SCONJ:Optional[float] = None  # Subordinating conjunctions
    G_OTHER:Optional[float] = None  # Other parts of speech

class InflectionPL(BaseModel):
    IN_ADJ_POS:Optional[float] = None  # Adjectives in positive degree
    IN_ADJ_COM:Optional[float] = None  # Adjectives in comparative degree
    IN_ADJ_SUP:Optional[float] = None  # Adjectives in superlative degree
    IN_ADV_POS:Optional[float] = None  # Adverbs in positive degree
    IN_ADV_COM:Optional[float] = None  # Adverbs in comparative degree
    IN_ADV_SUP:Optional[float] = None  # Adverbs in superlative degree
    IN_N_1NOM:Optional[float] = None  # Nouns in the nominative case
    IN_N_2GEN:Optional[float] = None  # Nouns in the genitive case
    IN_N_3DAT:Optional[float] = None  # Nouns in the dative case
    IN_N_4ACC:Optional[float] = None  # Nouns in the accusative case
    IN_N_5INS:Optional[float] = None  # Nouns in the instrumental case
    IN_N_6LOC:Optional[float] = None  # Nouns in the locative case
    IN_N_7VOC:Optional[float] = None  # Nouns in the vocative case
    IN_N_SG:Optional[float] = None  # Singular nouns
    IN_N_PL:Optional[float] = None  # Plural nouns
    IN_N_MS:Optional[float] = None  # Singular masculine nouns
    IN_N_MP:Optional[float] = None  # Nouns in masculine personal gender (plural)
    IN_N_FS:Optional[float] = None  # Singular feminine nouns
    IN_N_NMP:Optional[float] = None  # Nouns in non-masculine personal gender (plural)
    IN_N_NS:Optional[float] = None  # Singular neutral nouns
    IN_PRO_1NOM:Optional[float] = None  # Pronouns in the nominative case
    IN_PRO_2GEN:Optional[float] = None  # Pronouns in the genitive case
    IN_PRO_3DAT:Optional[float] = None  # Pronouns in the dative case
    IN_PRO_4ACC:Optional[float] = None  # Pronouns in the accusative case
    IN_PRO_5INS:Optional[float] = None  # Pronouns in the instrumental case
    IN_PRO_6LOC:Optional[float] = None  # Pronouns in the locative case
    IN_PRO_7VOC:Optional[float] = None  # Pronouns in the vocative case
    IN_PRO_1S:Optional[float] = None  # First person singular pronouns
    IN_PRO_2S:Optional[float] = None  # Second person singular pronouns
    IN_PRO_3S:Optional[float] = None  # Third person singular pronouns
    IN_PRO_1P:Optional[float] = None  # First person plural pronouns
    IN_PRO_2P:Optional[float] = None  # Second person plural pronouns
    IN_PRO_3P:Optional[float] = None  # Third person plural pronouns
    IN_V_1S:Optional[float] = None  # Verbs in 1 person singular
    IN_V_2S:Optional[float] = None  # Verbs in 2 person singular
    IN_V_3S:Optional[float] = None  # Verbs in 3 person singular
    IN_V_1P:Optional[float] = None  # Verbs in 1 person plural
    IN_V_2P:Optional[float] = None  # Verbs in 2 person plural
    IN_V_3P:Optional[float] = None  # Verbs in 3 person plural
    IN_V_FIN:Optional[float] = None  # Finite verbs
    IN_V_INF:Optional[float] = None  # Infinitive verbs
    IN_V_QUASI:Optional[float] = None  # Quasi-verbs
    IN_V_IMPERS:Optional[float] = None  # Impersonal verb forms
    IN_V_IMPERS_PERF:Optional[float] = None  # Impersonal verb forms in perfective aspect
    IN_V_IMPERS_IMPERF:Optional[float] = None  # Impersonal verb forms in imperfective aspect
    IN_V_MOD:Optional[float] = None  # Modal verbs
    IN_V_PACT:Optional[float] = None  # Active adjectival participles
    IN_V_PPAS:Optional[float] = None  # Passive adjectival participles
    IN_V_PPAS_PERF:Optional[float] = None  # Passive adjectival participles in perfective aspect
    IN_V_PPAS_IMPERF:Optional[float] = None  # Passive adjectival participles in imperfective aspect
    IN_V_PCON:Optional[float] = None  # Present adverbial participles
    IN_V_PANT:Optional[float] = None  # Perfect adverbial participles
    IN_V_PERF:Optional[float] = None  # Verbs in perfect aspect
    IN_V_IMPERF:Optional[float] = None  # Verbs in imperfect aspect
    IN_V_ACT:Optional[float] = None  # Verbs in active voice
    IN_V_PASS:Optional[float] = None  # Verbs in passive voice
    IN_V_GER:Optional[float] = None  # Gerunds
    IN_V_PRES:Optional[float] = None  # Verbs in present tense
    IN_V_PAST:Optional[float] = None  # Verbs in past tense
    IN_V_FUTS:Optional[float] = None  # Verbs in simple future tense
    IN_V_FUTC:Optional[float] = None  # Verbs in future complex tense
    IN_V_FUT:Optional[float] = None  # Verbs in future tense
    IN_V_IMP:Optional[float] = None  # Verbs in imperative mood
    IN_V_COND:Optional[float] = None  # Verbs in conditional

# Continuation of Inflection class
# Syntactic Features
class SyntacticPL(BaseModel):
    SY_FMWE:Optional[float] = None  # Flat multiword expressions
    SY_APPM:Optional[float] = None  # Appositional modifiers
    SY_S_DE:Optional[float] = None  # Words in declarative sentences
    SY_S_EX:Optional[float] = None  # Words in exclamatory sentences
    SY_S_IN:Optional[float] = None  # Words in interrogative sentences
    SY_S_NEG:Optional[float] = None  # Words in negative sentences
    SY_S_ELL:Optional[float] = None  # Words in ellipsis-ending sentences
    SY_S_VOC:Optional[float] = None  # Words in sentences with a noun in the vocative case
    SY_S_NOM:Optional[float] = None  # Words in nominal sentences
    SY_S_INF:Optional[float] = None  # Words in infinitive-only sentences without finite verbs
    SY_NPRED:Optional[float] = None  # Nominal predicates
    SY_MOD:Optional[float] = None  # Words within modifiers
    SY_NPHR:Optional[float] = None  # Words in nominal phrases
    SY_INV_OBJ:Optional[float] = None  # OVS word order
    SY_INV_EPI:Optional[float] = None  # Inverted epithet
    SY_INIT:Optional[float] = None  # Words being the initial token in a sentence
    SY_QUOT:Optional[float] = None  # Words in quotation marks
    SY_SIMILE_ADJ:Optional[float] = None  # Similes (adjective)

# Punctuation Features
class PunctuationPL(BaseModel):
    PUNCT_TOTAL:Optional[float] = None  # Total punctuation
    PUNCT_BI_NOUN:Optional[float] = None  # Punctuation following a noun
    PUNCT_BI_VERB:Optional[float] = None  # Punctuation following a verb

# Lexical Features
class LexicalPL(BaseModel):
    L_NAME:Optional[float] = None  # Proper names
    L_NAME_M:Optional[float] = None  # Masculine proper nouns
    L_NAME_F:Optional[float] = None  # Feminine proper nouns
    L_NAME_ENT:Optional[float] = None  # Named entities
    L_PLACEN_GEOG:Optional[float] = None  # Place and geographical names
    L_PERSN:Optional[float] = None  # Person names
    L_PERSN_M:Optional[float] = None  # Masculine person names
    L_PERSN_F:Optional[float] = None  # Feminine person names
    L_ORGN:Optional[float] = None  # Organization names
    L_ETHN:Optional[float] = None  # Ethnonyms and demonyms
    L_GEOG_ADJ:Optional[float] = None  # Adjectives derived from geographical names
    L_DATE:Optional[float] = None  # Dates
    L_VULG:Optional[float] = None  # Vulgarisms
    L_INTENSIF:Optional[float] = None  # Degree modifiers of Greek origin
    L_ERROR:Optional[float] = None  # Common linguistic errors
    L_ADVPHR:Optional[float] = None  # Adverbial phrases
    L_ADV_TEMP:Optional[float] = None  # Adverbs of time
    L_ADV_DUR:Optional[float] = None  # Adverbs of duration
    L_ADV_FREQ:Optional[float] = None  # Adverbs of frequency
    L_SYL_G1:Optional[float] = None  # One-syllable words
    L_SYL_G2:Optional[float] = None  # Two-syllable words
    L_SYL_G3:Optional[float] = None  # Three-syllable words
    L_SYL_G4:Optional[float] = None  # Words formed of 4 or more syllables
    L_TTR_IA:Optional[float] = None  # Type-token ratio for non-lemmatized tokens
    L_TTR_LA:Optional[float] = None  # Type-token ratio for lemmatized tokens
    L_CONT_A:Optional[float] = None  # Incidence of content words
    L_CONT_T:Optional[float] = None  # Content words types
    L_CONT_L:Optional[float] = None  # Content words lemma types
    L_FUNC_A:Optional[float] = None  # Incidence of function words
    L_FUNC_T:Optional[float] = None  # Function words types
    L_FUNC_L:Optional[float] = None  # Function words lemma types
    L_STOP:Optional[float] = None  # Incidence of stop words
    L_TCCT1:Optional[float] = None  # Tokens covering 1% of most common types
    L_TCCT5:Optional[float] = None  # Tokens covering 5% of most common types

# Psycholinguistic Features
class PsycholinguisticsPL(BaseModel):
    PS_M_POSa:Optional[float] = None  # Words having more than mean positivity
    PS_M_POSb:Optional[float] = None  # Words having less than mean positivity
    PS_M_NEGa:Optional[float] = None  # Words having more than mean negativity
    PS_M_NEGb:Optional[float] = None  # Words having less than mean negativity
    PS_M_REFa:Optional[float] = None  # Words having more than mean reflectiveness
    PS_M_REFb:Optional[float] = None  # Words having less than mean reflectiveness
    PS_M_AUTa:Optional[float] = None  # Words having more than mean automaticity
    PS_M_AUTb:Optional[float] = None  # Words having less than mean automaticity
    PS_M_AROa:Optional[float] = None  # Words having more than mean arousal
    PS_M_AROb:Optional[float] = None  # Words having less than mean arousal
    PS_M_SIGa:Optional[float] = None  # Words having more than mean significance
    PS_M_SIGb:Optional[float] = None  # Words having less than mean significance

# Descriptive Features
class DescriptivePL(BaseModel):
    DESC_ADJ_CP:Optional[float] = None  # Compound adjectives
    DESC_ADJ:Optional[float] = None  # Adjectival description of properties
    DESC_ADV:Optional[float] = None  # Adverbial description of properties
    DESC_APOS_NPHR:Optional[float] = None  # Descriptive apostrophe with a nominal phrase
    DESC_APOS_VERB:Optional[float] = None  # Apostrophe containing a verb
    DESC_APOS_ADJ:Optional[float] = None  # Descriptive apostrophe with an adjective
    DESC_ADV_ADJ:Optional[float] = None  # Adverbs followed by adjectives
    DESC_ADV_ADV:Optional[float] = None  # Adverb pairs incidence
    DESC_PRON_VOC:Optional[float] = None  # Personal pronoun followed by a noun in the vocative case
    DESC_PRON_ADJ_VOC:Optional[float] = None  # Personal pronoun followed by an adjective and a noun in the vocative case

# Graphical Features
class GraphicalPL(BaseModel):
    GR_UPPER:Optional[float] = None  # Capital letters
    GR_EMOJI:Optional[float] = None  # Emojis
    GR_EMOT:Optional[float] = None  # Emoticons
    GR_LENNY:Optional[float] = None  # Lenny faces
    GR_MENTION:Optional[float] = None  # Direct mentions with @
    GR_HASH:Optional[float] = None  # Hashtags
    GR_LINK:Optional[float] = None  # Hyperlinks


# Prefix to model mapping (some example mappings)
PREFIX_MAP_PL = {
    "G_": "grammatical_forms",
    "IN_": "inflection",
    "SY_": "syntactic",
    "PUNCT_": "punctuation",
    "L_": "lexical",
    "PS_": "psycholinguistics",
    "DESC_": "descriptive",
    "GR_": "graphical"
}

class AllStyloMetrixFeaturesPL(BaseModel):
    text: Optional[str]
    grammatical_forms: GrammaticalFormsPL
    inflection: InflectionPL
    syntactic: SyntacticPL
    punctuation: PunctuationPL
    lexical: LexicalPL
    psycholinguistics: PsycholinguisticsPL
    descriptive: DescriptivePL
    graphical: GraphicalPL


    def __init__(self, with_prepare: bool = False, **data):
        if with_prepare:
            # Initialize sub-models with empty dicts
            init_data = {
                "grammatical_forms": {},
                "inflection": {},
                "syntactic": {},
                "punctuation": {},
                "lexical": {},
                "psycholinguistics": {},
                "descriptive": {},
                "graphical": {}
            }

            # Distribute data to corresponding sub-models based on prefix
            for key, value in data.items():
                for prefix, model_name in PREFIX_MAP_PL.items():
                    if key.startswith(prefix):
                        init_data[model_name][key] = value
                        break

            super().__init__(
                text=data.get("text"),
                grammatical_forms=GrammaticalFormsPL(**init_data["grammatical_forms"]),
                inflection=InflectionPL(**init_data["inflection"]),
                syntactic=SyntacticPL(**init_data["syntactic"]),
                punctuation=PunctuationPL(**init_data["punctuation"]),
                lexical=LexicalPL(**init_data["lexical"]),
                psycholinguistics=PsycholinguisticsPL(**init_data["psycholinguistics"]),
                descriptive=DescriptivePL(**init_data["descriptive"]),
                graphical=GraphicalPL(**init_data["graphical"])
            )
        else:
            super().__init__(
                text=data.get("text"),
                grammatical_forms=GrammaticalFormsPL(**data["grammatical_forms"]),
                inflection=InflectionPL(**data["inflection"]),
                syntactic=SyntacticPL(**data["syntactic"]),
                punctuation=PunctuationPL(**data["punctuation"]),
                lexical=LexicalPL(**data["lexical"]),
                psycholinguistics=PsycholinguisticsPL(**data["psycholinguistics"]),
                descriptive=DescriptivePL(**data["descriptive"]),
                graphical=GraphicalPL(**data["graphical"])
            )


    def create(self, **data):
        # Initialize sub-models with empty dicts
        init_data = {
            "grammatical_forms": {},
            "inflection": {},
            "syntactic": {},
            "punctuation": {},
            "lexical": {},
            "psycholinguistics": {},
            "descriptive": {},
            "graphical": {}
        }

        # Distribute data to corresponding sub-models based on prefix
        for key, value in data.items():
            for prefix, model_name in PREFIX_MAP_PL.items():
                if key.startswith(prefix):
                    init_data[model_name][key] = value
                    break

        # Initialize each sub-model with its respective data
        super().__init__(
            text=data.get("text"),
            grammatical_forms=GrammaticalFormsPL(**init_data["grammatical_forms"]),
            inflection=InflectionPL(**init_data["inflection"]),
            syntactic=SyntacticPL(**init_data["syntactic"]),
            punctuation=PunctuationPL(**init_data["punctuation"]),
            lexical=LexicalPL(**init_data["lexical"]),
            psycholinguistics=PsycholinguisticsPL(**init_data["psycholinguistics"]),
            descriptive=DescriptivePL(**init_data["descriptive"]),
            graphical=GraphicalPL(**init_data["graphical"])
        )


class PartOfSpeechEN(BaseModel):
    POS_VERB:Optional[float] = None  # Verbs
    POS_NOUN:Optional[float] = None  # Nouns
    POS_ADJ:Optional[float] = None  # Adjectives
    POS_ADV:Optional[float] = None  # Adverbs
    POS_DET:Optional[float] = None  # Determiners
    POS_INTJ:Optional[float] = None  # Interjections
    POS_CONJ:Optional[float] = None  # Conjunctions
    POS_PART:Optional[float] = None  # Particles
    POS_NUM:Optional[float] = None  # Numerals
    POS_PREP:Optional[float] = None  # Prepositions
    POS_PRO:Optional[float] = None  # Pronouns

class LexicalEN(BaseModel):
    L_REF:Optional[float] = None  # References
    L_HASHTAG:Optional[float] = None  # Hashtags
    L_MENTION:Optional[float] = None  # Mentions
    L_RT:Optional[float] = None  # Retweets
    L_LINKS:Optional[float] = None  # Links
    L_CONT_A:Optional[float] = None  # Content words
    L_FUNC_A:Optional[float] = None  # Function words
    L_CONT_T:Optional[float] = None  # Content words types
    L_FUNC_T:Optional[float] = None  # Function words types
    L_PLURAL_NOUNS:Optional[float] = None  # Nouns in plural
    L_SINGULAR_NOUNS:Optional[float] = None  # Nouns in singular
    L_PROPER_NAME:Optional[float] = None  # Proper names
    L_PERSONAL_NAME:Optional[float] = None  # Personal names
    L_NOUN_PHRASES:Optional[float] = None  # Incidence of noun phrases
    L_PUNCT:Optional[float] = None  # Punctuation
    L_PUNCT_DOT:Optional[float] = None  # Punctuation - dots
    L_PUNCT_COM:Optional[float] = None  # Punctuation - commas
    L_PUNCT_SEMC:Optional[float] = None  # Punctuation - semicolons
    L_PUNCT_COL:Optional[float] = None  # Punctuation - colons
    L_PUNCT_DASH:Optional[float] = None  # Punctuation - dashes
    L_POSSESSIVES:Optional[float] = None  # Nouns in possessive case
    L_ADJ_POSITIVE:Optional[float] = None  # Adjectives in positive degree
    L_ADJ_COMPARATIVE:Optional[float] = None  # Adjectives in comparative degree
    L_ADJ_SUPERLATIVE:Optional[float] = None  # Adjectives in superlative degree
    L_ADV_POSITIVE:Optional[float] = None  # Adverbs in positive degree
    L_ADV_COMPARATIVE:Optional[float] = None  # Adverbs in comparative degree
    L_ADV_SUPERLATIVE:Optional[float] = None  # Adverbs in superlative degree
    PS_CONTRADICTION:Optional[float] = None  # Opposition, limitation, contradiction
    PS_AGREEMENT:Optional[float] = None  # Agreement, similarity
    PS_EXAMPLES:Optional[float] = None  # Examples, emphasis
    PS_CONSEQUENCE:Optional[float] = None  # Consequence, result
    PS_CAUSE:Optional[float] = None  # Cause, purpose
    PS_LOCATION:Optional[float] = None  # Location, space
    PS_TIME:Optional[float] = None  # Time
    PS_CONDITION:Optional[float] = None  # Condition, hypothesis
    PS_MANNER:Optional[float] = None  # Manner

class SyntacticEN(BaseModel):
    SY_QUESTION:Optional[float] = None  # Number of words in interrogative sentences
    SY_NARRATIVE:Optional[float] = None  # Number of words in narrative sentences
    SY_NEGATIVE_QUESTIONS:Optional[float] = None  # Words in negative questions
    SY_SPECIAL_QUESTIONS:Optional[float] = None  # Words in special questions
    SY_TAG_QUESTIONS:Optional[float] = None  # Words in tag questions
    SY_GENERAL_QUESTIONS:Optional[float] = None  # Words in general questions
    SY_EXCLAMATION:Optional[float] = None  # Number of words in exclamatory sentences
    SY_IMPERATIVE:Optional[float] = None  # Words in imperative sentences
    SY_SUBORD_SENT:Optional[float] = None  # Words in subordinate sentences
    SY_SUBORD_SENT_PUNCT:Optional[float] = None  # Punctuation in subordinate sentences
    SY_COORD_SENT:Optional[float] = None  # Words in coordinate sentences
    SY_COORD_SENT_PUNCT:Optional[float] = None  # Punctuation in coordinate sentences
    SY_SIMPLE_SENT:Optional[float] = None  # Tokens in simple sentences
    SY_INVERSE_PATTERNS:Optional[float] = None  # Incidents of inverse patterns
    SY_SIMILE:Optional[float] = None  # Simile
    SY_FRONTING:Optional[float] = None  # Fronting
    SY_IRRITATION:Optional[float] = None  # Incidents of continuous tenses as irritation markers
    SY_INTENSIFIER:Optional[float] = None  # Intensifiers
    SY_QUOT:Optional[float] = None  # Words in quotation marks

class VerbTensesEN(BaseModel):
    VT_PRESENT_SIMPLE:Optional[float] = None  # Present Simple tense
    VT_PRESENT_PROGRESSIVE:Optional[float] = None  # Present Continuous tense
    VT_PRESENT_PERFECT:Optional[float] = None  # Present Perfect tense
    VT_PRESENT_PERFECT_PROGR:Optional[float] = None  # Present Perfect Continuous tense
    VT_PRESENT_SIMPLE_PASSIVE:Optional[float] = None  # Present Simple passive
    VT_PRESENT_PROGR_PASSIVE:Optional[float] = None  # Present Continuous passive
    VT_PRESENT_PERFECT_PASSIVE:Optional[float] = None  # Present Perfect passive
    VT_PAST_SIMPLE:Optional[float] = None  # Past Simple tense
    VT_PAST_SIMPLE_BE:Optional[float] = None  # Past Simple 'to be' verb
    VT_PAST_PROGR:Optional[float] = None  # Past Continuous tense
    VT_PAST_PERFECT:Optional[float] = None  # Past Perfect tense
    VT_PAST_PERFECT_PROGR:Optional[float] = None  # Past Perfect Continuous tense
    VT_PAST_SIMPLE_PASSIVE:Optional[float] = None  # Past Simple passive
    VT_PAST_POGR_PASSIVE:Optional[float] = None  # Past Continuous passive
    VT_PAST_PERFECT_PASSIVE:Optional[float] = None  # Past Perfect passive
    VT_FUTURE_SIMPLE:Optional[float] = None  # Future Simple tense
    VT_FUTURE_PROGRESSIVE:Optional[float] = None  # Future Continuous tense
    VT_FUTURE_PERFECT:Optional[float] = None  # Future Perfect tense
    VT_FUTURE_PERFECT_PROGR:Optional[float] = None  # Future Perfect Continuous tense
    VT_FUTURE_SIMPLE_PASSIVE:Optional[float] = None  # Future Simple passive
    VT_FUTURE_PROGR_PASSIVE:Optional[float] = None  # Future Continuous passive
    VT_FUTURE_PERFECT_PASSIVE:Optional[float] = None  # Future Perfect passive
    VT_WOULD:Optional[float] = None  # Would verb simple
    VT_WOULD_PASSIVE:Optional[float] = None  # Would verb passive
    VT_WOULD_PROGRESSIVE:Optional[float] = None  # Would verb continuous
    VT_WOULD_PERFECT:Optional[float] = None  # Would verb perfect
    VT_WOULD_PERFECT_PASSIVE:Optional[float] = None  # Would verb perfect passive
    VT_SHOULD:Optional[float] = None  # Should verb simple
    VT_SHOULD_PASSIVE:Optional[float] = None  # Should verb simple passive
    VT_SHALL:Optional[float] = None  # Shall verb simple
    VT_SHALL_PASSIVE:Optional[float] = None  # Shall verb simple passive
    VT_SHOULD_PROGRESSIVE:Optional[float] = None  # Should verb continuous
    VT_SHOULD_PERFECT:Optional[float] = None  # Should verb perfect
    VT_SHOULD_PERFECT_PASSIVE:Optional[float] = None  # Should verb perfect passive
    VT_MUST:Optional[float] = None  # Must verb simple
    VT_MUST_PASSIVE:Optional[float] = None  # Must verb simple passive
    VT_MUST_PROGRESSIVE:Optional[float] = None  # Must verb continuous
    VT_MUST_PERFECT:Optional[float] = None  # Must verb perfect
    VT_MST_PERFECT_PASSIVE:Optional[float] = None  # Must verb perfect passive
    VT_CAN:Optional[float] = None  # Can verb simple
    VT_CAN_PASSIVE:Optional[float] = None  # Can verb simple passive
    VT_COULD:Optional[float] = None  # Could verb simple
    VT_COULD_PASSIVE:Optional[float] = None  # Could verb simple passive
    VT_CAN_PROGRESSIVE:Optional[float] = None  # Can verb continuous
    VT_COULD_PROGRESSIVE:Optional[float] = None  # Could verb continuous
    VT_COULD_PERFECT:Optional[float] = None  # Could + perfect infinitive
    VT_COULD_PERFECT_PASSIVE:Optional[float] = None  # Could verb perfect passive
    VT_MAY:Optional[float] = None  # May verb simple
    VT_MAY_PASSIVE:Optional[float] = None  # May verb simple passive
    VT_MIGHT:Optional[float] = None  # Might verb simple
    VT_MIGHT_PASSIVE:Optional[float] = None  # Might verb simple passive
    VT_MAY_PROGRESSIVE:Optional[float] = None  # May verb continuous
    VT_MIGTH_PERFECT:Optional[float] = None  # Might verb perfect
    VT_MIGHT_PERFECT_PASSIVE:Optional[float] = None  # Might verb perfect passive
    VT_MAY_PERFECT_PASSIVE:Optional[float] = None  # May verb perfect passive

class StatisticsEN(BaseModel):
    ST_TYPE_TOKEN_RATIO_LEMMAS:Optional[float] = None  # Type-token ratio for words lemmas
    ST_HERDAN_TTR:Optional[float] = None  # Herdan's TTR
    ST_MASS_TTR:Optional[float] = None  # Mass TTR
    ST_SENT_WRDSPERSENT:Optional[float] = None  # Difference between the number of words and the number of sentences
    ST_SENT_DIFFERENCE:Optional[float] = None  # Symmetric difference between nodes in sentences per doc
    ST_REPETITIONS_WORDS:Optional[float] = None  # Repetitions of words in text
    ST_REPETITIONS_SENT:Optional[float] = None  # Repetitions of sentences in text
    ST_SENT_D_VP:Optional[float] = None  # Statistics between VPs
    ST_SENT_D_NP:Optional[float] = None  # Statistics between NPs
    ST_SENT_D_PP:Optional[float] = None  # Statistics between PPs
    ST_SENT_D_ADJP:Optional[float] = None  # Statistics between ADJPs
    ST_SENT_D_ADVP:Optional[float] = None  # Statistics between ADVPs

class PronounsEN(BaseModel):
    L_I_PRON:Optional[float] = None  # 'I' pronoun
    L_HE_PRON:Optional[float] = None  # 'He' pronoun
    L_SHE_PRON:Optional[float] = None  # 'She' pronoun
    L_IT_PRON:Optional[float] = None  # 'It' pronoun
    L_YOU_PRON:Optional[float] = None  # 'You' pronoun
    L_WE_PRON:Optional[float] = None  # 'We' pronoun
    L_THEY_PRON:Optional[float] = None  # 'They' pronoun
    L_ME_PRON:Optional[float] = None  # 'Me' pronoun
    L_YOU_OBJ_PRON:Optional[float] = None  # 'You' object pronoun
    L_HIM_PRON:Optional[float] = None  # 'Him' object pronoun
    L_HER_OBJECT_PRON:Optional[float] = None  # 'Her' object pronoun
    L_IT_OBJECT_PRON:Optional[float] = None  # 'It' object pronoun
    L_US_PRON:Optional[float] = None  # 'Us' pronoun
    L_THEM_PRON:Optional[float] = None  # 'Them' pronoun
    L_MY_PRON:Optional[float] = None  # 'My' pronoun
    L_YOUR_PRON:Optional[float] = None  # 'Your' pronoun
    L_HIS_PRON:Optional[float] = None  # 'His' pronoun
    L_HER_PRON:Optional[float] = None  # 'Her' possessive pronoun
    L_ITS_PRON:Optional[float] = None  # 'Its' possessive pronoun
    L_OUR_PRON:Optional[float] = None  # 'Our' possessive pronoun
    L_THEIR_PRON:Optional[float] = None  # 'Their' possessive pronoun
    L_YOURS_PRON:Optional[float] = None  # 'Yours' pronoun
    L_THEIRS_PRON:Optional[float] = None  # 'Theirs' pronoun
    L_HERS_PRON:Optional[float] = None  # 'Hers' pronoun
    L_OURS_PRON:Optional[float] = None  # 'Ours' possessive pronoun
    L_MYSELF_PRON:Optional[float] = None  # 'Myself' pronoun
    L_YOURSELF_PRON:Optional[float] = None  # 'Yourself' pronoun
    L_HIMSELF_PRON:Optional[float] = None  # 'Himself' pronoun
    L_HERSELF_PRON:Optional[float] = None  # 'Herself' pronoun
    L_ITSELF_PRON:Optional[float] = None  # 'Itself' pronoun
    L_OURSELVES_PRON:Optional[float] = None  # 'Ourselves' pronoun
    L_YOURSELVES_PRON:Optional[float] = None  # 'Yourselves' pronoun
    L_THEMSELVES_PRON:Optional[float] = None  # 'Themselves' pronoun
    L_FIRST_PERSON_SING_PRON:Optional[float] = None  # First person singular pronouns
    L_SECOND_PERSON_PRON:Optional[float] = None  # Second person pronouns
    L_THIRD_PERSON_SING_PRON:Optional[float] = None  # Third person singular pronouns
    L_THIRD_PERSON_PLURAL_PRON:Optional[float] = None  # Third person plural pronouns

class GeneralEN(BaseModel):
    VF_INFINITIVE:Optional[float] = None  # Incidence of verbs in infinitive
    G_PASSIVE:Optional[float] = None  # Passive voice
    G_ACTIVE:Optional[float] = None  # Active voice
    G_PRESENT:Optional[float] = None  # Present tenses
    G_PAST:Optional[float] = None  # Past tenses
    G_FUTURE:Optional[float] = None  # Future tenses active
    G_MODALS_SIMPLE:Optional[float] = None  # Modal verbs simple
    G_MODALS_CONT:Optional[float] = None  # Modal verbs continuous
    G_MODALS_PERFECT:Optional[float] = None  # Modal verbs perfect

class HurtlexEN(BaseModel):
    AN:Optional[float] = None  # Animals
    DDP:Optional[float] = None  # Cognitive disabilities and diversity
    SVP:Optional[float] = None  # Words related to the seven deadly sins
    CDS:Optional[float] = None  # Derogatory words
    DDF:Optional[float] = None  # Physical disabilities and diversity
    IS:Optional[float] = None  # Words related to social and economic disadvantage
    PS:Optional[float] = None  # Negative stereotypes ethnic slurs
    RE:Optional[float] = None  # Felonies and words related to crime
    ASF:Optional[float] = None  # Female genitalia
    ASM:Optional[float] = None  # Male genitalia
    OM:Optional[float] = None  # Words related to homosexuality
    RCI:Optional[float] = None  # Locations and demonyms
    DMC:Optional[float] = None  # Moral and behavioral defects
    OR:Optional[float] = None  # Plants
    QAS:Optional[float] = None  # Words with potential negative connotations
    PA:Optional[float] = None  # Professions and occupations
    PR:Optional[float] = None  # Words related to prostitution


# Map of prefixes to the specific model names
PREFIX_MAP_EN = {
    "POS_": "part_of_speech",
    "L_": ["lexical", "pronouns"],
    "SY_": "syntactic",
    "VT_": "verb_tenses",
    "ST_": "statistics",
    "G_": "general",
    "VF_": "general",
}

class AllStyloMetrixFeaturesEN(BaseModel):
    text: Optional[str]
    part_of_speech: PartOfSpeechEN
    lexical: LexicalEN
    syntactic: SyntacticEN
    verb_tenses: VerbTensesEN
    statistics: StatisticsEN
    pronouns: PronounsEN
    general: GeneralEN
    hurtlex: HurtlexEN

    def __init__(self, with_prepare: bool = False, **data):
        if with_prepare:
            model_data = {
                "part_of_speech": {},
                "lexical": {},
                "syntactic": {},
                "verb_tenses": {},
                "statistics": {},
                "pronouns": {},
                "general": {},
                "hurtlex": {}
            }

            # Distribute the incoming data to the respective model data structures
            for key, value in data.items():
                model_key = None
                for prefix in PREFIX_MAP_EN:
                    if key.startswith(prefix):
                        if isinstance(PREFIX_MAP_EN[prefix], list):
                            if key.endswith("_PRON"):
                                model_key = PREFIX_MAP_EN[prefix][1]
                            else:
                                model_key = PREFIX_MAP_EN[prefix][0]
                        else:
                            model_key = PREFIX_MAP_EN[prefix]
                        break
                else:
                    if len(key) <= 3:
                        model_key = 'hurtlex'

                if model_key:
                    model_data[model_key][key] = value

            # Initialize each specific model with its data
            super().__init__(
                text=data.get("text"),
                part_of_speech=PartOfSpeechEN(**model_data["part_of_speech"]),
                lexical=LexicalEN(**model_data["lexical"]),
                syntactic=SyntacticEN(**model_data["syntactic"]),
                verb_tenses=VerbTensesEN(**model_data["verb_tenses"]),
                statistics=StatisticsEN(**model_data["statistics"]),
                pronouns=PronounsEN(**model_data["pronouns"]),
                general=GeneralEN(**model_data["general"]),
                hurtlex=HurtlexEN(**model_data["hurtlex"])
            )
        else:
            super().__init__(
                text=data.get("text"),
                part_of_speech=PartOfSpeechEN(**data["part_of_speech"]),
                lexical=LexicalEN(**data["lexical"]),
                syntactic=SyntacticEN(**data["syntactic"]),
                verb_tenses=VerbTensesEN(**data["verb_tenses"]),
                statistics=StatisticsEN(**data["statistics"]),
                pronouns=PronounsEN(**data["pronouns"]),
                general=GeneralEN(**data["general"]),
                hurtlex=HurtlexEN(**data["hurtlex"])
            )

    def create(self, **data):
        model_data = {
            "part_of_speech": {},
            "lexical": {},
            "syntactic": {},
            "verb_tenses": {},
            "statistics": {},
            "pronouns": {},
            "general": {},
            "hurtlex": {}
        }

        # Distribute the incoming data to the respective model data structures
        for key, value in data.items():
            model_key = None
            for prefix in PREFIX_MAP_EN:
                if key.startswith(prefix):
                    if isinstance(PREFIX_MAP_EN[prefix], list):
                        if key.endswith("_PRON"):
                            model_key = PREFIX_MAP_EN[prefix][1]
                        else:
                            model_key = PREFIX_MAP_EN[prefix][0]
                    else:
                        model_key = PREFIX_MAP_EN[prefix]
                    break
            else:
                if len(key) <= 3:
                    model_key = 'hurtlex'

            if model_key:
                model_data[model_key][key] = value

        # Initialize each specific model with its data
        super().__init__(
            text=data.get("text"),
            part_of_speech=PartOfSpeechEN(**model_data["part_of_speech"]),
            lexical=LexicalEN(**model_data["lexical"]),
            syntactic=SyntacticEN(**model_data["syntactic"]),
            verb_tenses=VerbTensesEN(**model_data["verb_tenses"]),
            statistics=StatisticsEN(**model_data["statistics"]),
            pronouns=PronounsEN(**model_data["pronouns"]),
            general=GeneralEN(**model_data["general"]),
            hurtlex=HurtlexEN(**model_data["hurtlex"])
        )

from typing import List

import numpy as np

from scipy.stats import gaussian_kde, percentileofscore
from matplotlib import pyplot as plt

from api.api_models.response import HistogramData, HistogramDataDTO
from api.server_config import API_ATTRIBUTES_REFERENCE_COLLECTION_NAME, API_MONGODB_DB_NAME, API_HISTOGRAMS_PATH, \
    API_MOST_IMPORTANT_ATTRIBUTES, LLM_REFERENCE_MODEL
from api.api_models.lightbulb_score import LightbulbScoreType
from dao.attribute import DAOAttributePL
from models.attribute import AttributePLInDB

dao_attribute_reference: DAOAttributePL = DAOAttributePL(collection_name=API_ATTRIBUTES_REFERENCE_COLLECTION_NAME, db_name=API_MONGODB_DB_NAME)

GENERATED_FLAT_DICT = None
REAL_FLAT_DICT = None
GENERATED_PARTIAL_FLAT_DICT = None
REAL_PARTIAL_FLAT_DICT = None

def load_reference_attributes() -> None:
    global GENERATED_FLAT_DICT, REAL_FLAT_DICT, GENERATED_PARTIAL_FLAT_DICT, REAL_PARTIAL_FLAT_DICT
    if GENERATED_FLAT_DICT is not None and REAL_FLAT_DICT is not None and GENERATED_PARTIAL_FLAT_DICT is not None and REAL_PARTIAL_FLAT_DICT is not None:
        return
    if LLM_REFERENCE_MODEL is None:
        generated: List[AttributePLInDB] = dao_attribute_reference.find_many_by_query({"is_generated": True})
    else:
        generated: List[AttributePLInDB] = dao_attribute_reference.find_many_by_query({"is_generated": True, "llm_model_name": LLM_REFERENCE_MODEL})

    real: List[AttributePLInDB] = dao_attribute_reference.find_many_by_query({"is_generated": False})

    GENERATED_FLAT_DICT = [(x.to_flat_dict_normalized(), 1) for x in generated]
    REAL_FLAT_DICT = [(x.to_flat_dict_normalized(), 0) for x in real]
    print(f"LOADED {len(GENERATED_FLAT_DICT) + len(REAL_FLAT_DICT)} document attributes from reference collection")

    GENERATED_PARTIAL_FLAT_DICT = []
    for sample in generated:
        if sample.partial_attributes is not None and len(sample.partial_attributes) > 0:
            for partial_attribute in sample.partial_attributes:
                GENERATED_PARTIAL_FLAT_DICT.append((partial_attribute.attribute.to_flat_dict_normalized(), 1))

    REAL_PARTIAL_FLAT_DICT = []
    for sample in real:
        if sample.partial_attributes is not None and len(sample.partial_attributes) > 0:
            for partial_attribute in sample.partial_attributes:
                REAL_PARTIAL_FLAT_DICT.append((partial_attribute.attribute.to_flat_dict_normalized(), 0))


    print(f"LOADED {len(GENERATED_PARTIAL_FLAT_DICT) + len(REAL_PARTIAL_FLAT_DICT)} paragraph attributes from reference collection")

def plot_two_hists(data1, data2, title, metric_name="Metric", num_bin=21, min_value=0, max_value=5, top=0.5,
                   additional_value=None, file_name=""):
    # Truncate data to max_value if needed
    data1_to_plot = [d if d < max_value else max_value for d in data1]
    data2_to_plot = [d if d < max_value else max_value for d in data2]

    w = (max_value - min_value) / num_bin
    bins = np.arange(min_value, max_value + w, w)

    weights1 = np.ones_like(data1_to_plot) / len(data1_to_plot)
    weights2 = np.ones_like(data2_to_plot) / len(data2_to_plot)

    plt.hist(data1_to_plot, bins=bins, weights=weights1, alpha=0.7, label='Generated', color='red')
    plt.hist(data2_to_plot, bins=bins, weights=weights2, alpha=0.7, label='Real', color='blue')
    if additional_value is not None:
        plt.axvline(additional_value, color='red', linestyle='--', linewidth=1, label='Sample value')

    plt.title(title)
    plt.xlim([min_value, max_value])
    plt.ylim(top=top)
    plt.xlabel(f'{metric_name} value')
    plt.ylabel('Lab reports share')
    plt.legend()
    plt.savefig(f'{API_HISTOGRAMS_PATH}/{file_name}.png')
    plt.clf()

def compute_histogram_data(attribute_name: str, num_bin=21,
                           min_value=None, max_value=None, additional_value=None, normalize=False, is_partial_attribute:bool=False) -> HistogramDataDTO:
    if is_partial_attribute:
        if is_attribute_available_in_partial_attributes(attribute_name):
            data_gen = [attribute[0][attribute_name] for attribute in GENERATED_PARTIAL_FLAT_DICT]
            data_real = [attribute[0][attribute_name] for attribute in REAL_PARTIAL_FLAT_DICT]
        else:
            raise ValueError(f"Attribute named {attribute_name} not available for partial analysis")
    else:
        data_gen = [attribute[0][attribute_name] for attribute in GENERATED_FLAT_DICT]
        data_real = [attribute[0][attribute_name] for attribute in REAL_FLAT_DICT]

    if min_value is None:
        min_value = min(np.percentile(data_gen, 5), np.percentile(data_real, 5))
    if max_value is None:
        max_value = max(np.percentile(data_gen, 95), np.percentile(data_real, 95))

    # Clip values
    data_gen = [d if d < max_value else max_value for d in data_gen]
    data_real = [d if d < max_value else max_value for d in data_real]

    w = (max_value - min_value) / num_bin
    bins = np.linspace(min_value, max_value, num_bin).tolist()

    if normalize:
        weights_gen = np.ones_like(data_gen) / len(data_gen)
        weights_real = np.ones_like(data_real) / len(data_real)

        counts_gen, _ = np.histogram(data_gen, bins=bins, weights=weights_gen)
        counts_real, _ = np.histogram(data_real, bins=bins, weights=weights_real)
    else:
        counts_gen, _ = np.histogram(data_gen, bins=bins)
        counts_real, _ = np.histogram(data_real, bins=bins)

    histogram_llm = HistogramData(
        feature=attribute_name,
        data_type="llm-generated",
        bins=bins,
        counts=counts_gen.tolist()
    )

    histogram_human = HistogramData(
        feature=attribute_name,
        data_type="human-written",
        bins=bins,
        counts=counts_real.tolist()
    )

    dto = HistogramDataDTO(
        llm=histogram_llm,
        human=histogram_human,
        # additional_value=additional_value,
        min_value=min_value,
        max_value=max_value,
        num_bins=num_bin,
        object_hash = ""
    )

    dto.object_hash = dto.calculate_histogram_hash()

    return dto



def compare_2_hists(attribute_name: str, min_value=None, max_value=None, top=0.41, num_bin=21,
                    additional_value=None, file_name:str= "", title:str= "") -> None:
    data_gen = [attribute[0][attribute_name] for attribute in GENERATED_FLAT_DICT]
    data_real = [attribute[0][attribute_name] for attribute in REAL_FLAT_DICT]
    if min_value is None:
        min_value = 0  # min(min(data_gen), min(data_real))
    if max_value is None:
        max_value = max(np.percentile(data_gen, 95), np.percentile(data_real, 95))

    plot_two_hists(data_gen, data_real, title=title, metric_name=attribute_name,
                   min_value=min_value, max_value=max_value, top=top, num_bin=num_bin,
                   additional_value=additional_value, file_name=file_name)


def _relative_density(value: float,
                      real_kde: gaussian_kde,
                      gen_kde:  gaussian_kde) -> float:
    """
    Raw score in [-1,1]:  +1 → the value sits only under the human curve,
                          -1 → the value sits only under the LLM curve,
                           0 → equally plausible under both.
    """
    p_real = real_kde.evaluate([value])[0]
    p_gen  = gen_kde.evaluate([value])[0]

    if p_real + p_gen == 0:        # totally unseen value
        return 0.0

    return (p_real - p_gen) / (p_real + p_gen)   # ∈ (-1,1)

def boost_with_cosine(score, boost=1.0, power=0.7):
    """
    Multiply by a smooth cosine bump centered at 0.
    - boost sets multiplier at 0.
    - power < 1 makes growth faster near 0.
    """
    s = float(np.clip(score, -1.0, 1.0))
    base = 0.5 * (1.0 + np.cos(np.pi * s))  # 1 at 0, 0 at ±1
    shaped = base ** power                  # adjust growth
    m = 1.0 + (boost - 1.0) * shaped
    return float(np.clip(s * m, -1.0, 1.0))

SQRT_2PI = np.sqrt(2.0 * np.pi)

def _silverman_bandwidth(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = max(len(v), 1)
    if n <= 1:
        return 1.0
    std = np.std(v)
    iqr = np.subtract(*np.percentile(v, [75, 25]))
    sigma = min(std, iqr / 1.349) if (std > 0 and iqr > 0) else max(std, 1e-6)
    return max(0.9 * sigma * n ** (-1.0 / 5.0), 1e-6)

def _kde_pdf_np(values: np.ndarray, x: float, h: float | None = None, eps: float = 1e-12) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return 0.0
    if not h or not np.isfinite(h) or h <= 0:
        h = _silverman_bandwidth(v)
    z = (x - v) / h
    pdf = np.sum(np.exp(-0.5 * z * z)) / (n * h * SQRT_2PI)
    return float(max(pdf, eps))

def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(max(1.4826 * mad, 1e-6))

def _sigmoid(u: float) -> float:
    if u >= 0:
        e = np.exp(-u)
        return 1.0 / (1.0 + e)
    else:
        e = np.exp(u)
        return e / (1.0 + e)

def relative_likelihood_score(
    value: float,
    real_values: np.ndarray,
    gen_values: np.ndarray,
    category,                        # LightbulbScoreType
    *,
    temperature: float = 1.0,
    tail_floor: float = 1e-9,
    share_bandwidth: bool = True,
    density_knee_frac: float = 0.35,
    density_sharpness: float = 3.0,
    density_gamma: float = 0.7,
    tail_knee_frac: float = 0.05,
    tail_sharpness: float = 2.0,
    tail_gamma: float = 0.6,
    distance_tau: float = 4.0,
    distance_beta: float = 1.5,
    purity_boost: float = 0.75,
    purity_power: float = 1.0,
    exclusive_bonus: float = 0.25,
    exclusive_gate_frac: float = 0.15,
    use_cosine_boost: bool = False,
) -> float:
    rv = np.asarray(real_values, dtype=float)
    gv = np.asarray(gen_values, dtype=float)
    rv = rv[np.isfinite(rv)]
    gv = gv[np.isfinite(gv)]

    if rv.size == 0 and gv.size == 0:
        return 0.0

    if share_bandwidth and rv.size and gv.size:
        h_shared = np.median([_silverman_bandwidth(rv), _silverman_bandwidth(gv)])
        hr = hg = max(h_shared, 1e-6)
    else:
        hr = _silverman_bandwidth(rv) if rv.size else 1.0
        hg = _silverman_bandwidth(gv) if gv.size else 1.0

    f_r = (_kde_pdf_np(rv, value, hr) if rv.size else 0.0) + tail_floor
    f_g = (_kde_pdf_np(gv, value, hg) if gv.size else 0.0) + tail_floor

    # 1. Equal priors, independent of sample counts
    pi_r = 0.5
    pi_g = 0.5

    # 2. No prior bias in log odds
    llr = np.log(f_r) - np.log(f_g)
    log_prior_odds = 0.0
    z = (llr + log_prior_odds) / max(temperature, 1e-6)
    p_h = _sigmoid(z)
    bi_score = float(2.0 * p_h - 1.0)

    # 3. Mixture density also uses equal priors
    mix_density = pi_r * f_r + pi_g * f_g

    med_r = float(np.median(rv)) if rv.size else value
    med_g = float(np.median(gv)) if gv.size else value
    f_ref_r = (_kde_pdf_np(rv, med_r, hr) if rv.size else 0.0) + tail_floor
    f_ref_g = (_kde_pdf_np(gv, med_g, hg) if gv.size else 0.0) + tail_floor
    f_ref_peak = max(f_ref_r, f_ref_g)

    purity = abs(f_r - f_g) / (f_r + f_g)

    knee_main = density_knee_frac * f_ref_peak
    gate_main = _sigmoid(
        density_sharpness * (np.log(mix_density) - np.log(knee_main + 1e-12))
    )
    density_weight_main = float(gate_main ** density_gamma)

    dom = max(f_r, f_g)
    knee_tail = tail_knee_frac * f_ref_peak
    gate_tail = _sigmoid(
        tail_sharpness * (np.log(dom) - np.log(knee_tail + 1e-12))
    )
    density_weight_tail = float(gate_tail ** tail_gamma)

    density_weight = float(
        (1.0 - purity) * density_weight_main + purity * density_weight_tail
    )

    z_r = abs(value - med_r) / _mad(rv) if rv.size else np.inf
    z_g = abs(value - med_g) / _mad(gv) if gv.size else np.inf
    z_min = min(z_r, z_g)
    distance_weight = float(
        np.exp(- (z_min / max(distance_tau, 1e-6)) ** distance_beta)
    )

    amplifier = 1.0 + purity_boost * (purity ** purity_power) * gate_main
    supported = bi_score * density_weight * distance_weight * amplifier

    dom_norm = float(np.clip(dom / max(f_ref_peak, 1e-12), 0.0, 1.0))
    bonus_gate = 1.0 if (purity > 0.85 and dom_norm >= exclusive_gate_frac) else 0.0
    signed_bonus = np.sign(bi_score) * exclusive_bonus * dom_norm * bonus_gate

    bi_score = float(np.clip(supported + signed_bonus, -1.0, 1.0))

    if category == LightbulbScoreType.HUMAN_WRITTEN:
        out = float(np.clip(bi_score, 0.0, 1.0))
    elif category == LightbulbScoreType.LLM_GENERATED:
        out = float(np.clip(bi_score, -1.0, 0.0))
    else:
        out = float(np.clip(bi_score, -1.0, 1.0))

    if use_cosine_boost:
        return boost_with_cosine(out)
    return out


def calculate_lightbulb_score(attribute_value,
                              attribute_name,
                              category=LightbulbScoreType.BIDIRECTIONAL,
                              is_chunk_attribute: bool = False) -> float:
    """
    Returns a scalar whose range depends on *category*.

    BIDIRECTIONAL : [-1, 1]   (+ → human-like, − → LLM-like)
    HUMAN_WRITTEN : [0, 1]   (close to 1 → confidently human)
    LLM_GENERATED : [ -1, 0]   (close to  -1 → confidently LLM)
    """
    if not is_chunk_attribute:
        gen_values = [attribute[0][attribute_name] for attribute in GENERATED_FLAT_DICT]
        real_values = [attribute[0][attribute_name] for attribute in REAL_FLAT_DICT]
    else:
        if is_attribute_available_in_partial_attributes(attribute_name):
            gen_values = [attribute[0][attribute_name] for attribute in GENERATED_PARTIAL_FLAT_DICT]
            real_values = [attribute[0][attribute_name] for attribute in REAL_PARTIAL_FLAT_DICT]
        else:
            raise ValueError(f"Attribute named {attribute_name} not available for partial analysis")

    gen_values = np.array(gen_values)
    real_values = np.array(real_values)

    raw = relative_likelihood_score(attribute_value, real_values, gen_values, category)

    if category == LightbulbScoreType.BIDIRECTIONAL:
        return float(np.clip(raw, -1, 1))

    if category == LightbulbScoreType.LLM_GENERATED:
        return float(np.clip(raw, -1, 0))

    if category == LightbulbScoreType.HUMAN_WRITTEN:
        return float(np.clip(raw,  0, 1))

    raise ValueError(f"Unknown category: {category}")

def is_attribute_available_in_partial_attributes(attribute_name):
    return attribute_name in GENERATED_PARTIAL_FLAT_DICT[0][0]
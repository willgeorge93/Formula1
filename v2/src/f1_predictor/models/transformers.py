"""ColumnTransformer configuration matching notebook pipeline."""

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def create_column_transformer() -> ColumnTransformer:
    """
    Create the ColumnTransformer matching the notebook configuration.

    From notebook (col_trans in Modelling.ipynb):
    - CountVectorizer for weather (NLP)
    - OneHotEncoder for 9 categorical features
    - StandardScaler for 5 numerical features

    Total: 15 transformations

    Returns:
        Configured ColumnTransformer
    """
    transformers = [
        # Text feature: Weather (NLP with CountVectorizer)
        (
            "weather",
            CountVectorizer(stop_words="english", binary=True),
            "weather",
        ),
        # Categorical features: OneHotEncoder
        (
            "direction",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ["direction"],
        ),
        (
            "country",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ["country"],
        ),
        (
            "locality",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ["locality"],
        ),
        (
            "type",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ["type"],
        ),
        (
            "season",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ["season"],
        ),
        (
            "round",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ["round"],
        ),
        (
            "qual_position",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ["qual_position"],
        ),
        (
            "grid",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ["grid"],
        ),
        (
            "race_name",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ["race_name"],
        ),
        # Numerical features: StandardScaler
        # Using with_mean=False for sparse data compatibility
        (
            "q_mean",
            StandardScaler(with_mean=False),
            ["q_mean"],
        ),
        (
            "q_best",
            StandardScaler(with_mean=False),
            ["q_best"],
        ),
        (
            "q_worst",
            StandardScaler(with_mean=False),
            ["q_worst"],
        ),
        (
            "length",
            StandardScaler(with_mean=False),
            ["length"],
        ),
        (
            "ageDuringRace",
            StandardScaler(with_mean=False),
            ["ageDuringRace"],
        ),
    ]

    return ColumnTransformer(transformers, remainder="drop")


def get_feature_names(transformer: ColumnTransformer) -> list[str]:
    """
    Get feature names from a fitted ColumnTransformer.

    Args:
        transformer: Fitted ColumnTransformer

    Returns:
        List of feature names
    """
    try:
        return list(transformer.get_feature_names_out())
    except AttributeError:
        # Fallback for older sklearn versions
        feature_names = []
        for name, trans, cols in transformer.transformers_:
            if name == "remainder":
                continue
            if hasattr(trans, "get_feature_names_out"):
                names = trans.get_feature_names_out(cols)
                feature_names.extend([f"{name}_{n}" for n in names])
            else:
                feature_names.extend([f"{name}_{c}" for c in cols])
        return feature_names

"""Quick EDA on dspy_hackathon/dataset/train.csv.

Run from the repo root:
    python dspy_hackathon/eda_on_train_dataset.py

Or use programmatically:
    from dspy_hackathon.eda_on_train_dataset import EDAOnTrainDataset
    eda = EDAOnTrainDataset.from_csv("dspy_hackathon/dataset/train.csv")
    cats = eda.get_classification_categories()  # -> DataFrame
    eda.run_all()                                # -> prints full report
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd


CSV_PATH = Path("dspy_hackathon/dataset/train.csv")
TARGET_COL = "target"
TEXT_COL = "description"


class EDAOnTrainDataset:
    """Exploratory data analysis for the PubMed-style classification CSV.

    The dataset has at minimum a text column (`description`) and a single-label
    target column (`target`). Each public method prints one report section;
    `run_all()` prints them in the canonical order.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str = TARGET_COL,
        text_col: str = TEXT_COL,
    ) -> None:
        self.df = df
        self.target_col = target_col
        self.text_col = text_col

    @classmethod
    def from_csv(
        cls,
        csv_path: Path | str = CSV_PATH,
        target_col: str = TARGET_COL,
        text_col: str = TEXT_COL,
    ) -> "EDAOnTrainDataset":
        """Load a CSV from disk and wrap it in a `EDAOnTrainDataset` instance."""
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run download_pubmed_dataset.py first."
            )
        return cls(pd.read_csv(path), target_col=target_col, text_col=text_col)

    @staticmethod
    def _banner(title: str) -> None:
        bar = "=" * len(title)
        print(f"\n{bar}\n{title}\n{bar}")

    def shape_and_dtypes(self) -> None:
        self._banner("1. Shape & dtypes")
        df = self.df
        print(f"rows: {len(df):,}")
        print(f"cols: {list(df.columns)}")
        print(df.dtypes.to_string())
        print(f"memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    def missingness(self) -> None:
        self._banner("2. Missingness")
        df = self.df
        nulls = df.isna().sum()
        print(nulls.to_string())
        blanks = df.select_dtypes(include="object").apply(
            lambda s: s.str.strip().eq("").sum()
        )
        print("\nempty strings (after strip):")
        print(blanks.to_string())

    def duplicates(self) -> None:
        self._banner("3. Duplicates")
        df = self.df
        dup_full = df.duplicated().sum()
        dup_desc = df.duplicated(subset=[self.text_col]).sum()
        print(f"fully duplicated rows: {dup_full:,}")
        print(f"duplicate `{self.text_col}` (any label): {dup_desc:,}")
        cross_label = df.groupby(self.text_col)[self.target_col].nunique().gt(1).sum()
        print(f"descriptions appearing under >1 target: {cross_label:,}")

    def get_classification_categories(self) -> pd.DataFrame:
        """Return the distinct classification labels with counts and proportions.

        The dataset is a single-label text classification task; this is the
        label set a downstream classifier is expected to predict over. Rows
        with a missing target are dropped before counting. The returned frame
        is sorted by frequency (descending) so the most common class is first.
        """
        if self.target_col not in self.df.columns:
            raise KeyError(
                f"Column '{self.target_col}' not found in dataframe; "
                f"got {list(self.df.columns)}"
            )

        series = self.df[self.target_col].dropna().astype(str).str.strip()
        counts = series.value_counts()
        total = int(counts.sum())
        pct = (counts / total * 100).round(2) if total else counts.astype(float)
        return pd.DataFrame(
            {"category": counts.index, "count": counts.values, "pct": pct.values}
        )

    def report_classification_categories(self) -> None:
        self._banner("4. Classification categories")
        categories = self.get_classification_categories()
        print(f"distinct categories: {len(categories)}")
        print(f"category set: {categories['category'].tolist()}")
        print("\nfrequency table (sorted desc):")
        print(categories.to_string(index=False))
        if len(categories) >= 2:
            ratio = categories["count"].max() / categories["count"].min()
            print(f"\nimbalance ratio (max/min): {ratio:.2f}")

    def description_length_chars(self) -> None:
        self._banner("5. Description length (chars)")
        char_len = self.df[self.text_col].str.len()
        print(char_len.describe(percentiles=[0.5, 0.9, 0.95, 0.99]).round(1).to_string())

    def description_length_tokens(self) -> None:
        self._banner("6. Description length (whitespace tokens)")
        word_len = self.df[self.text_col].str.split().str.len()
        print(word_len.describe(percentiles=[0.5, 0.9, 0.95, 0.99]).round(1).to_string())

    def length_per_target(self) -> None:
        self._banner("7. Length per target (words)")
        word_len = self.df[self.text_col].str.split().str.len()
        by_target_words = (
            self.df.assign(_w=word_len)
            .groupby(self.target_col)["_w"]
            .agg(["count", "mean", "median", "std", "min", "max"])
            .round(2)
        )
        print(by_target_words.to_string())

    def raw_vs_cleaned(self) -> None:
        self._banner("8. raw vs cleaned column equality")
        df = self.df
        if "description_cln" not in df.columns:
            print("(no `description_cln` column present; skipping)")
            return
        same = (df[self.text_col].fillna("") == df["description_cln"].fillna("")).sum()
        diff = len(df) - same
        print(f"identical rows: {same:,} ({same / len(df):.1%})")
        print(f"differing rows: {diff:,}")
        if diff:
            sample = df.loc[
                df[self.text_col] != df["description_cln"],
                [self.text_col, "description_cln"],
            ].head(3)
            for i, row in sample.iterrows():
                print(f"\n  example {i}:")
                print(f"    raw : {row[self.text_col][:140]!r}")
                print(f"    cln : {row['description_cln'][:140]!r}")

    def top_first_words_per_target(self, top_k: int = 15) -> None:
        self._banner(f"9. Top-{top_k} first words per target")
        first = self.df[self.text_col].str.split().str[0].str.lower()
        for tgt in sorted(self.df[self.target_col].dropna().unique()):
            top = first[self.df[self.target_col] == tgt].value_counts().head(top_k)
            print(f"\n[{tgt}]")
            for word, n in top.items():
                print(f"  {word:<20s} {n:>7,}")

    def corpus_vocab_sample(self, sample_size: int = 50_000, top_k: int = 20) -> None:
        self._banner("10. Roughest signal: corpus vocab")
        descriptions = self.df[self.text_col].dropna()
        sample = descriptions.sample(n=min(sample_size, len(descriptions)), random_state=0)
        tokens: Counter[str] = Counter()
        for s in sample:
            tokens.update(t.lower() for t in s.split() if t.isalpha())
        print(f"sampled docs: {len(sample):,}")
        print(f"unique alpha tokens (sample): {len(tokens):,}")
        print(f"top-{top_k} alpha tokens (sample):")
        for tok, n in tokens.most_common(top_k):
            print(f"  {tok:<20s} {n:>7,}")

    def run_all(self) -> None:
        """Print every report section in canonical order."""
        self.shape_and_dtypes()
        self.missingness()
        self.duplicates()
        self.report_classification_categories()
        self.description_length_chars()
        self.description_length_tokens()
        self.length_per_target()
        self.raw_vs_cleaned()
        self.top_first_words_per_target()
        self.corpus_vocab_sample()


def main() -> int:
    try:
        eda = EDAOnTrainDataset.from_csv(CSV_PATH)
    except FileNotFoundError as exc:
        print(exc)
        return 1
    eda.run_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())

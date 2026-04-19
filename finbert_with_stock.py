import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification


BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "data" / "stock_data"
HF_CACHE = BASE_DIR / ".hf-cache"
LOCAL_MODEL_DIR = BASE_DIR / "models" / "finbert_classifier"
MODEL_NAME = "ProsusAI/finbert"
DATE_LOW = pd.Timestamp("2021-01-04")
DATE_HIGH = pd.Timestamp("2022-09-20")

os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FinBERT sentiment scoring for one stock's headline files."
    )
    parser.add_argument(
        "--ticker",
        default="AMT",
        help="Ticker symbol to process, for example AMT.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional explicit output directory. Defaults to data/stock_data/<ticker>_Simon.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for FinBERT inference.",
    )
    return parser.parse_args()


def build_paths(ticker: str, explicit_output_dir: str | None) -> tuple[Path, Path, Path, Path]:
    input_dir = DATA_ROOT / ticker
    output_dir = Path(explicit_output_dir) if explicit_output_dir else DATA_ROOT / f"{ticker.lower()}_Simon"
    text_polarity_path = input_dir / "textPolarity.csv"
    stock_path = input_dir / f"stocks_ts_{ticker}_2021-1-4_2022-9-20.csv"

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not text_polarity_path.exists():
        raise FileNotFoundError(f"Missing textPolarity.csv: {text_polarity_path}")
    if not stock_path.exists():
        raise FileNotFoundError(f"Missing stock csv: {stock_path}")

    return input_dir, output_dir, text_polarity_path, stock_path


def load_model() -> tuple[AutoTokenizer, TFAutoModelForSequenceClassification]:
    if LOCAL_MODEL_DIR.exists():
        model_path = LOCAL_MODEL_DIR
    else:
        snapshot_ref = HF_CACHE / "hub" / "models--ProsusAI--finbert" / "refs" / "main"
        if snapshot_ref.exists():
            snapshot_hash = snapshot_ref.read_text(encoding="utf-8").strip()
            model_path = HF_CACHE / "hub" / "models--ProsusAI--finbert" / "snapshots" / snapshot_hash
        else:
            model_path = Path(MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = TFAutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    return tokenizer, model


def score_titles(
    titles: list[str],
    tokenizer: AutoTokenizer,
    model: TFAutoModelForSequenceClassification,
    batch_size: int,
) -> np.ndarray:
    batches = []
    for start in range(0, len(titles), max(batch_size, 1)):
        stop = start + max(batch_size, 1)
        batch_titles = titles[start:stop]
        batch_inputs = tokenizer(
            batch_titles,
            padding=True,
            truncation=True,
            return_tensors="tf",
        )
        batch_logits = model(batch_inputs, training=False).logits
        batches.append(tf.nn.softmax(batch_logits, axis=-1).numpy())
    return np.concatenate(batches, axis=0)


def read_headline_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    unnamed_columns = [column for column in frame.columns if column.startswith("Unnamed:")]
    if unnamed_columns:
        frame = frame.drop(columns=unnamed_columns)
    if "Titles" not in frame.columns:
        raise ValueError(f"{path} does not contain a Titles column.")
    frame["Titles"] = frame["Titles"].fillna("None").astype(str)
    return frame


def score_headline_files(
    input_dir: Path,
    tokenizer: AutoTokenizer,
    model: TFAutoModelForSequenceClassification,
    batch_size: int,
) -> list[pd.DataFrame]:
    scored_frames: list[pd.DataFrame] = []
    headline_files = sorted(
        input_dir.glob("headlinesIter*.csv"),
        key=lambda path: int(path.stem.replace("headlinesIter", "")),
    )

    for file_path in headline_files:
        frame = read_headline_file(file_path)
        frame["filename"] = file_path.name

        if frame.empty:
            scored_frames.append(frame)
            continue

        probabilities = score_titles(frame["Titles"].tolist(), tokenizer, model, batch_size)
        frame["Positive_BERT"] = probabilities[:, 0]
        frame["Negative_BERT"] = probabilities[:, 1]
        frame["Neutral_BERT"] = probabilities[:, 2]
        frame["avg_pos"] = frame["Positive_BERT"].mean()
        frame["avg_neg"] = frame["Negative_BERT"].mean()
        frame["avg_neu"] = frame["Neutral_BERT"].mean()
        scored_frames.append(frame)

    return scored_frames


def build_daily_outputs(
    scored_frames: list[pd.DataFrame],
    text_polarity: pd.DataFrame,
    stock_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows = pd.concat(scored_frames, axis=0, ignore_index=True)

    influential_rows = []
    missing_files = []
    for frame in scored_frames:
        filename = frame["filename"].iat[0] if "filename" in frame.columns and not frame.empty else None
        if frame.empty:
            if filename is None:
                missing_files.append(pd.DataFrame({"filename": [None]}))
            else:
                missing_files.append(pd.DataFrame({"filename": [filename]}))
            continue

        if frame["Positive_BERT"].sum() > frame["Negative_BERT"].sum():
            row = frame.loc[[frame["Positive_BERT"].idxmax()]]
        else:
            row = frame.loc[[frame["Negative_BERT"].idxmax()]]
        influential_rows.append(row)

    single_headlines = pd.concat(influential_rows, axis=0, ignore_index=True)

    missing_data = (
        pd.concat(missing_files, axis=0, ignore_index=True)
        if missing_files
        else pd.DataFrame(columns=["filename"])
    )
    if not missing_data.empty:
        missing_data["label_sin"] = np.nan
        missing_data["Sentiment_sin_PosorNeg"] = np.nan
        missing_data["Titles"] = "NaN"

    headnews = all_rows[
        [
            "filename",
            "Titles",
            "Positive_BERT",
            "Negative_BERT",
            "Neutral_BERT",
            "avg_pos",
            "avg_neg",
            "avg_neu",
        ]
    ].copy()
    headnews["Sentiment_mul"] = headnews[["Positive_BERT", "Negative_BERT", "Neutral_BERT"]].idxmax(axis=1)
    headnews["Sentiment_sin"] = headnews[["avg_pos", "avg_neg", "avg_neu"]].idxmax(axis=1)
    headnews["Sentiment_mul_PosorNeg"] = headnews[["Positive_BERT", "Negative_BERT"]].idxmax(axis=1)
    headnews["Sentiment_sin_PosorNeg"] = headnews[["avg_pos", "avg_neg"]].idxmax(axis=1)
    headnews["label_mul"] = pd.factorize(headnews["Sentiment_mul"])[0]
    headnews["label_sin"] = pd.factorize(headnews["Sentiment_sin"])[0]
    headnews["label_mul"] = np.where(headnews["label_mul"] == 2, -1, headnews["label_mul"])
    headnews["label_sin"] = np.where(headnews["label_sin"] == 2, -1, headnews["label_sin"])

    headnews_single = headnews[["filename", "label_sin", "Sentiment_sin_PosorNeg"]].drop_duplicates()
    if not missing_data.empty:
        headnews_single = pd.concat([headnews_single, missing_data[["filename", "label_sin", "Sentiment_sin_PosorNeg"]]])

    headnews_single["sort"] = headnews_single["filename"].str.extract(r"(\d+)", expand=False).astype(int)
    headnews_single = headnews_single.sort_values("sort").drop(columns=["sort"]).reset_index(drop=True)

    text_aligned = text_polarity.reset_index(drop=True)
    newsheadline_single_day = pd.concat([headnews_single, text_aligned], axis=1)
    newsheadline_single_day = newsheadline_single_day[
        ["Date", "Highest Category", "Polarity", "PosOrNeg", "label_sin", "Sentiment_sin_PosorNeg", "filename"]
    ].copy()
    newsheadline_single_day["Date"] = pd.to_datetime(newsheadline_single_day["Date"])

    stock_frame = stock_frame.copy()
    stock_frame["date"] = pd.to_datetime(stock_frame["date"], format="%m/%d/%y")
    stock_frame["Diff"] = (stock_frame["NextDayClose"] - stock_frame["close"] > 0).astype(int)
    stock_frame = stock_frame[["date", "symbol", "close", "rsi", "26-Day EMA", "NextDayClose", "Diff"]]
    stock_frame = stock_frame[(stock_frame["date"] > DATE_LOW) & (stock_frame["date"] < DATE_HIGH + pd.Timedelta(days=1))]

    headnews_a = headnews[["Titles", "filename"]].copy()
    if not missing_data.empty:
        headnews_a = pd.concat([headnews_a, missing_data[["Titles", "filename"]]])

    headnews_b = newsheadline_single_day[["Date", "filename"]].rename(columns={"Date": "date"})
    headnews_c = pd.merge(headnews_a, headnews_b, on="filename", how="inner")
    multi_data = pd.merge(stock_frame, headnews_c, on="date", how="inner")

    single_headlines = single_headlines[["filename", "Titles"]].copy()
    if not missing_data.empty:
        single_headlines = pd.concat([single_headlines, missing_data[["filename", "Titles"]]])
    single_headlines["sort"] = single_headlines["filename"].str.extract(r"(\d+)", expand=False).astype(int)
    single_headlines = single_headlines.sort_values("sort").drop(columns=["sort"]).reset_index(drop=True)

    single_newsheadline_day = pd.concat([single_headlines, text_aligned], axis=1)
    single_newsheadline_day = single_newsheadline_day[["Date", "Titles", "filename"]].rename(columns={"Date": "date"})
    single_newsheadline_day["date"] = pd.to_datetime(single_newsheadline_day["date"])
    single_newsheadline_day = pd.merge(stock_frame, single_newsheadline_day, on="date", how="inner")

    return single_newsheadline_day, multi_data


def save_outputs(
    scored_frames: list[pd.DataFrame],
    single_newsheadline_day: pd.DataFrame,
    multi_data: pd.DataFrame,
    output_dir: Path,
    ticker: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scored_dir = output_dir / "finbert_scored_headlines"
    scored_dir.mkdir(parents=True, exist_ok=True)

    for frame in scored_frames:
        if "filename" not in frame.columns or frame.empty:
            continue
        file_name = frame["filename"].iat[0]
        frame.drop(columns=["filename"]).to_csv(scored_dir / file_name, index=False)

    single_newsheadline_day.to_csv(
        output_dir / f"Single_newsheadline_day_{ticker.lower()}.csv",
        index=False,
    )
    multi_data.to_csv(
        output_dir / f"Multi_data_{ticker.lower()}.csv",
        index=False,
    )


def prepare_text_polarity(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    first_column = frame.columns[0]
    if first_column != "Date":
        frame = frame.rename(columns={first_column: "Date"})
    return frame


def main() -> None:
    args = parse_args()
    ticker = args.ticker.upper()
    input_dir, output_dir, text_polarity_path, stock_path = build_paths(ticker, args.output_dir)

    text_polarity = prepare_text_polarity(text_polarity_path)
    stock_frame = pd.read_csv(stock_path, encoding="utf-8-sig")

    tokenizer, model = load_model()
    scored_frames = score_headline_files(input_dir, tokenizer, model, args.batch_size)
    single_newsheadline_day, multi_data = build_daily_outputs(scored_frames, text_polarity, stock_frame)
    save_outputs(scored_frames, single_newsheadline_day, multi_data, output_dir, ticker)

    print(f"Ticker: {ticker}")
    print(f"Headline files scored: {len(scored_frames)}")
    print(f"Single_newsheadline_day rows: {len(single_newsheadline_day)}")
    print(f"Multi_data rows: {len(multi_data)}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()

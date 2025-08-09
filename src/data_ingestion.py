from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import logging, yaml

# ---------------- Logging ----------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(fmt)

fh = logging.FileHandler(LOG_DIR / "data_ingestion.log")
fh.setLevel(logging.DEBUG)
fh.setFormatter(fmt)

# avoid duplicate handlers when re-running in notebooks
if not logger.handlers:
    logger.addHandler(ch)
    logger.addHandler(fh)

# ---------------- Helpers ----------------
def load_params(params_path: Path) -> dict:
    try:
        with params_path.open("r", encoding="utf-8") as f:
            params = yaml.safe_load(f)
        logger.debug("Parameters retrieved from %s", params_path)
        return params
    except FileNotFoundError:
        logger.error("File not found: %s", params_path)
        raise
    except yaml.YAMLError as e:
        logger.error("YAML error: %s", e)
        raise

def load_data(data_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        logger.debug("Data loaded from %s", data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the CSV file: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred while loading the data: %s", e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])
        df = df.rename(columns={"v1": "target", "v2": "text"})
        logger.debug("Data preprocessing completed")
        return df
    except KeyError as e:
        logger.error("Missing column in the dataframe: %s", e)
        raise

def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, data_root: Path) -> None:
    try:
        raw_dir = data_root / "raw"         # <-- DVC expects "data/raw" (no "./")
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / "train.csv").write_text(train_df.to_csv(index=False), encoding="utf-8")
        (raw_dir / "test.csv").write_text(test_df.to_csv(index=False), encoding="utf-8")
        logger.debug("Train and test data saved to %s", raw_dir)
    except Exception as e:
        logger.error("Unexpected error occurred while saving the data: %s", e)
        raise

# ---------------- Main ----------------
def main():
    try:
        params = load_params(Path("params.yaml"))
        test_size = params["data_ingestion"]["test_size"]

        data_url = "https://raw.githubusercontent.com/vikashishere/Datasets/main/spam.csv"
        df = load_data(data_url)
        final_df = preprocess_data(df)

        train_df, test_df = train_test_split(final_df, test_size=test_size, random_state=2)
        save_data(train_df, test_df, data_root=Path("data"))
    except Exception as e:
        logger.error("Failed to complete the data ingestion process: %s", e)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

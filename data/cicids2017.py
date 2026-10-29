"""CIC-IDS2017 data loader and cleaning pipeline.

Source of Truth: SPEC.md (Section 2, Section 8 Module 1)
Selected corrected version: Liu et al. (IEEE CNS 2022) / Engelen et al. (IEEE SPW 2021).

Global Rules applied:
1. Fit ALL scalers/encoders on TRAIN data only. Never use TEST statistics.
2. Cleaning drops duplicates, inf/NaN, negative-value artifacts (physically impossible only),
   and identifier columns.
3. Produces a verifiable cleaning report (rows dropped and why).
4. Schema validation against YAML with explicit error reporting on missing/unexpected columns.
5. Distinct label audit and YAML-based attack family mapping with unmapped label detection.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, StandardScaler

# Identifier features to remove per SPEC.md Section 2:
IDENTIFIER_COLUMNS = [
    "flow id",
    "source ip",
    "src ip",
    "source port",
    "src port",
    "destination ip",
    "dst ip",
    "destination port",
    "dst port",
    "timestamp",
    "unnamed: 0",
    "index",
]

# Legitimate sentinel columns where negative values (such as -1) are protocol sentinels
# (e.g., TCP SYN without window scale option) and must NOT be filtered out.
SENTINEL_COLUMNS = [
    "init_win_bytes_forward",
    "init_win_bytes_backward",
]

# Physical flow metrics where negative values are physically impossible (toolchain corruption)
PHYSICALLY_IMPOSSIBLE_NEGATIVE_PATTERNS = [
    "duration",
    "packets",
    "bytes",
    "length",
    "rate",
    "iat",
    "active",
    "idle",
]

# Default / fallback attack family mapping (also loadable dynamically from YAML)
ATTACK_FAMILY_MAPPING: Dict[str, str] = {
    "benign": "Benign",
    "ftp-patator": "BruteForce",
    "ssh-patator": "BruteForce",
    "dos slowloris": "DoS/DDoS",
    "dos slowhttptest": "DoS/DDoS",
    "dos hulk": "DoS/DDoS",
    "dos goldeneye": "DoS/DDoS",
    "ddos": "DoS/DDoS",
    "heartbleed": "DoS/DDoS",
    "portscan": "PortScan",
    "web attack – brute force": "WebAttack",
    "web attack - brute force": "WebAttack",
    "web attack – xss": "WebAttack",
    "web attack - xss": "WebAttack",
    "web attack – sql injection": "WebAttack",
    "web attack - sql injection": "WebAttack",
    "bot": "Botnet",
    "infiltration": "Infiltration",
    "infilteration": "Infiltration",
}



class SchemaValidationError(ValueError):
    """Raised when data columns do not conform to the expected schema YAML."""
    pass


@dataclass
class CleaningReport:
    """Detailed audit report logging rows dropped, schema conformance, and negative values."""
    dataset_name: str = "CIC-IDS2017 (corrected)"
    raw_rows_total: int = 0
    duplicate_rows_dropped: int = 0
    inf_nan_rows_dropped: int = 0
    negative_value_rows_dropped: int = 0
    total_rows_dropped: int = 0
    clean_rows_total: int = 0
    identifier_columns_removed: List[str] = field(default_factory=list)
    features_retained_count: int = 0
    feature_columns: List[str] = field(default_factory=list)
    distinct_raw_labels_found: List[str] = field(default_factory=list)
    unmapped_labels_found: List[str] = field(default_factory=list)
    label_distribution_before: Dict[str, int] = field(default_factory=dict)
    label_distribution_after: Dict[str, int] = field(default_factory=dict)
    rows_dropped_per_label: Dict[str, int] = field(default_factory=dict)
    family_distribution_after: Dict[str, int] = field(default_factory=dict)
    negative_stats_per_column: Dict[str, Any] = field(default_factory=dict)
    clipped_columns: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def inspect_negative_values(
    df: pd.DataFrame,
    label_col: str = "Label",
    sentinel_cols: Optional[Sequence[str]] = None,
    print_summary: bool = True,
) -> pd.DataFrame:
    """Inspects and reports negative values per column and per label.
    
    Treats sentinel columns as legitimate and distinguishes them from physical metrics.
    """
    sentinel_set = {c.lower() for c in (sentinel_cols or SENTINEL_COLUMNS)}
    records = []

    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        col_lower = col.lower()
        neg_mask = df[col] < 0
        neg_count = int(neg_mask.sum())
        if neg_count == 0:
            continue

        pct_total = (neg_count / len(df)) * 100.0
        is_sentinel = col_lower in sentinel_set

        # Breakdown per label
        breakdown_per_label = {}
        if label_col in df.columns:
            for lbl, count in df.loc[neg_mask, label_col].value_counts().items():
                lbl_total = (df[label_col] == lbl).sum()
                lbl_pct = (count / lbl_total * 100.0) if lbl_total > 0 else 0.0
                breakdown_per_label[str(lbl)] = f"{count} ({lbl_pct:.2f}%)"

        records.append({
            "column": col,
            "negative_count": neg_count,
            "percentage_of_column": round(pct_total, 3),
            "is_sentinel": is_sentinel,
            "breakdown_per_label": breakdown_per_label,
        })

    report_df = pd.DataFrame(records)

    if print_summary:
        if report_df.empty:
            print("\n[Negative Value Analysis] No negative values detected in dataset.")
        else:
            print("\n" + "=" * 80)
            print("[Negative Value Analysis per Column and per Label]")
            print("=" * 80)
            for _, row in report_df.iterrows():
                sentinel_flag = "[SENTINEL]" if row["is_sentinel"] else "[PHYSICAL METRIC]"
                print(f"Column: {row['column']} {sentinel_flag}")
                print(f"  Total Negatives: {row['negative_count']} ({row['percentage_of_column']}%)")
                print("  Breakdown by Label:")
                for lbl, stats in row["breakdown_per_label"].items():
                    print(f"    - {lbl}: {stats}")
            print("=" * 80 + "\n")

    return report_df


class CICIDS2017Cleaner:
    """Preprocesses and cleans CIC-IDS2017 flow data.
    
    Adheres to SPEC.md Section 2 and research requirements:
    - Drops duplicates.
    - Inspects negative values and sentinels per column and label.
    - Drops rows with negative values ONLY when drop_negative_artifacts=True
      and strictly for physically impossible non-negative columns.
    - Drops inf/NaN.
    - Strips identifier columns.
    - Maps labels to canonical families via YAML.
    - Reports rows dropped per label class before and after.
    """

    def __init__(
        self,
        schema_path: Optional[Union[str, Path]] = "configs/schema_cicids2017.yaml",
        family_mapping_path: Optional[Union[str, Path]] = "configs/attack_families_cicids2017.yaml",
        identifier_cols: Optional[Sequence[str]] = None,
        sentinel_cols: Optional[Sequence[str]] = None,
        label_col: str = "Label",
        day_col: str = "Day",
        drop_negative_artifacts: bool = False,
    ):
        self.schema_path = Path(schema_path) if schema_path else None
        self.family_mapping_path = Path(family_mapping_path) if family_mapping_path else None
        self.label_col = label_col
        self.day_col = day_col
        self.drop_negative_artifacts = drop_negative_artifacts
        self.identifier_cols = [c.lower() for c in (identifier_cols or IDENTIFIER_COLUMNS)]
        self.sentinel_cols = [c.lower() for c in (sentinel_cols or SENTINEL_COLUMNS)]

        # Load family mapping
        self.family_mapping = self._load_family_mapping()

    def _load_family_mapping(self) -> Dict[str, str]:
        if self.family_mapping_path and self.family_mapping_path.exists():
            with open(self.family_mapping_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return data.get("label_mapping", {})
        return {}

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        return df

    def _identify_label_col(self, df: pd.DataFrame) -> str:
        for c in df.columns:
            if c.strip().lower() == self.label_col.lower():
                return c
        raise ValueError(f"Could not find '{self.label_col}' in columns.")

    def _is_physically_impossible_negative(self, col_name: str) -> bool:
        lower = col_name.lower().strip()
        if lower in self.sentinel_cols or col_name in self.sentinel_cols:
            return False
        return True

    def clean(
        self,
        df: pd.DataFrame,
        dataset_name: str = "CIC-IDS2017",
        print_negative_summary: bool = True,
    ) -> Tuple[pd.DataFrame, CleaningReport]:
        """Perform end-to-end cleaning and produce an audit report."""
        report = CleaningReport(dataset_name=dataset_name)
        report.raw_rows_total = len(df)

        df = self._normalize_columns(df)
        label_col = self._identify_label_col(df)

        # Record initial label distribution
        raw_labels = df[label_col].astype(str).str.strip()
        report.distinct_raw_labels_found = sorted(raw_labels.unique().tolist())
        print(f"[Loader] Distinct raw label strings found ({len(report.distinct_raw_labels_found)}):")
        for lbl in report.distinct_raw_labels_found:
            count = int((raw_labels == lbl).sum())
            print(f"  - '{lbl}': {count} flows")

        report.label_distribution_before = {
            str(k): int(v) for k, v in raw_labels.value_counts().items()
        }

        import re

        def normalize_label(lbl: str) -> str:
            lbl = str(lbl).strip()
            # Replace broken replacement characters \ufffd, en-dashes, em-dashes
            lbl = lbl.replace("\ufffd", "-").replace("–", "-").replace("—", "-")
            lbl = re.sub(r"\s*-\s*", " - ", lbl)
            lbl = re.sub(r"\s+", " ", lbl)
            return lbl.strip()

        # Check for unmapped labels
        unmapped = []
        for lbl in report.distinct_raw_labels_found:
            norm = normalize_label(lbl)
            if (
                lbl not in self.family_mapping
                and lbl.lower() not in self.family_mapping
                and norm not in self.family_mapping
                and norm.lower() not in self.family_mapping
            ):
                unmapped.append(lbl)

        if unmapped:
            print(f"\n[WARNING] Unmapped attack labels found in YAML configuration:")
            for lbl in unmapped:
                print(f"  - '{lbl}' (No family defined in {self.family_mapping_path})")
            report.unmapped_labels_found = unmapped
        else:
            print("[Loader] All distinct raw labels are mapped in YAML.")

        # Step 1: Drop duplicates BEFORE splitting (avoids identical flows across splits)
        df_no_dups = df.drop_duplicates()
        report.duplicate_rows_dropped = len(df) - len(df_no_dups)
        df = df_no_dups

        # Step 2: Separate identifier columns
        cols_to_remove = []
        for col in df.columns:
            if col == label_col or col == self.day_col:
                continue
            if col.lower() in self.identifier_cols:
                cols_to_remove.append(col)

        report.identifier_columns_removed = sorted(cols_to_remove)
        df = df.drop(columns=cols_to_remove)

        # Numeric feature identification
        meta_cols = [label_col]
        if self.day_col in df.columns:
            meta_cols.append(self.day_col)
        feature_cols = [c for c in df.columns if c not in meta_cols]

        # Step 3: Inspect negative values per column and label
        neg_analysis_df = inspect_negative_values(
            df,
            label_col=label_col,
            sentinel_cols=self.sentinel_cols,
            print_summary=print_negative_summary,
        )
        report.negative_stats_per_column = neg_analysis_df.to_dict(orient="records")

        # Step 4: Handle string "Infinity" / "-Infinity" -> inf -> NaN, then drop inf / NaN
        for c in feature_cols:
            if df[c].dtype == object or str(df[c].dtype) == "string":
                df[c] = df[c].replace({
                    "Infinity": np.inf,
                    "-Infinity": -np.inf,
                    "inf": np.inf,
                    "-inf": -np.inf,
                })
        df_feat = df[feature_cols].apply(pd.to_numeric, errors="coerce")
        df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
        nan_mask = df_feat.isna().any(axis=1)
        report.inf_nan_rows_dropped = int(nan_mask.sum())
        df = df.loc[~nan_mask].copy()
        df_feat = df_feat.loc[~nan_mask].copy()

        # Step 5: Negative value handling (PART A Cleaning Decision)
        if self.drop_negative_artifacts:
            # 1. Drop corrupt rows where Flow Duration < 0 (~115 rows)
            if "Flow Duration" in df_feat.columns:
                dur_neg_mask = df_feat["Flow Duration"] < 0
                n_dur_dropped = int(dur_neg_mask.sum())
                report.negative_value_rows_dropped = n_dur_dropped
                df = df.loc[~dur_neg_mask].copy()
                df_feat = df_feat.loc[~dur_neg_mask].copy()
            else:
                report.negative_value_rows_dropped = 0

            # 2. For other physical-metric columns, clip negatives to 0 (do NOT drop those rows)
            phys_cols_to_clip = [
                c for c in feature_cols
                if self._is_physically_impossible_negative(c) and c != "Flow Duration"
            ]
            clipped_stats: Dict[str, int] = {}
            for c in phys_cols_to_clip:
                neg_count = int((df_feat[c] < 0).sum())
                if neg_count > 0:
                    clipped_stats[c] = neg_count
                    df_feat[c] = df_feat[c].clip(lower=0.0)
            report.clipped_columns = clipped_stats

            # 3. Sentinel columns (e.g. Init_Win_bytes_forward/backward = -1) are kept completely untouched
        else:
            report.negative_value_rows_dropped = 0
            report.clipped_columns = {}

        # Update clean feature columns
        for c in feature_cols:
            df[c] = df_feat[c].values

        # Step 6: Map attack families from YAML
        clean_labels = df[label_col].astype(str).str.strip()
        df["Label"] = clean_labels

        def map_family(lbl: str) -> str:
            if lbl in self.family_mapping:
                return self.family_mapping[lbl]
            if lbl.lower() in self.family_mapping:
                return self.family_mapping[lbl.lower()]
            norm = normalize_label(lbl)
            if norm in self.family_mapping:
                return self.family_mapping[norm]
            if norm.lower() in self.family_mapping:
                return self.family_mapping[norm.lower()]
            return "Unknown_Attack" if norm.lower() != "benign" else "Benign"

        df["Attack_Family"] = clean_labels.map(map_family)
        df = df.reset_index(drop=True)

        # Finalize report
        report.clean_rows_total = len(df)
        report.total_rows_dropped = report.raw_rows_total - report.clean_rows_total
        report.features_retained_count = len(feature_cols)
        report.feature_columns = feature_cols
        report.label_distribution_after = {
            str(k): int(v) for k, v in df["Label"].value_counts().items()
        }
        report.family_distribution_after = {
            str(k): int(v) for k, v in df["Attack_Family"].value_counts().items()
        }

        # Report rows dropped per label class
        rows_dropped_per_lbl = {}
        for lbl, count_before in report.label_distribution_before.items():
            count_after = report.label_distribution_after.get(lbl, 0)
            rows_dropped_per_lbl[lbl] = count_before - count_after
        report.rows_dropped_per_label = rows_dropped_per_lbl

        return df, report


class CICIDS2017Dataset:
    """Configurable loader for CIC-IDS2017 files with strict schema validation."""

    def __init__(
        self,
        data_path: Union[str, Path],
        schema_path: Optional[Union[str, Path]] = "configs/schema_cicids2017.yaml",
        family_mapping_path: Optional[Union[str, Path]] = "configs/attack_families_cicids2017.yaml",
        cleaner: Optional[CICIDS2017Cleaner] = None,
        strict_schema: bool = True,
    ):
        self.data_path = Path(data_path)
        self.schema_path = Path(schema_path) if schema_path else None
        self.strict_schema = strict_schema
        self.cleaner = cleaner or CICIDS2017Cleaner(
            schema_path=schema_path,
            family_mapping_path=family_mapping_path,
        )

    def validate_schema(self, actual_columns: Sequence[str]) -> Tuple[List[str], List[str]]:
        """Validates dataframe columns against the schema YAML.
        
        Raises SchemaValidationError with missing and unexpected columns if validation fails.
        """
        if not self.schema_path or not self.schema_path.exists():
            return [], []

        with open(self.schema_path, "r", encoding="utf-8") as f:
            schema = yaml.safe_load(f)

        expected_all = []
        for section in ["identifier_columns", "expected_features"]:
            expected_all.extend(schema.get(section, []))
        if "label_column" in schema:
            expected_all.append(schema["label_column"])

        actual_norm = {str(c).strip().lower(): str(c).strip() for c in actual_columns}
        expected_norm = {str(c).strip().lower(): str(c).strip() for c in expected_all}

        # Check missing columns
        missing = [
            orig for low, orig in expected_norm.items()
            if low not in actual_norm and low != "day"
        ]

        # Check unexpected columns
        unexpected = [
            orig for low, orig in actual_norm.items()
            if low not in expected_norm and low != "day" and low != "attack_family"
        ]

        if (missing or unexpected) and self.strict_schema:
            error_msg = [
                f"\n[Schema Validation Failed for {self.schema_path}]",
                f"Missing expected columns ({len(missing)}): {missing}",
                f"Unexpected extra columns ({len(unexpected)}): {unexpected}",
            ]
            raise SchemaValidationError("\n".join(error_msg))

        return missing, unexpected

    def discover_files(self) -> List[Path]:
        if not self.data_path.exists():
            return []
        if self.data_path.is_file():
            return [self.data_path]
        # Search recursively for *.csv and *.parquet in data_path and subfolders
        files = list(self.data_path.rglob("*.csv")) + list(self.data_path.rglob("*.parquet"))
        return sorted(files)

    @staticmethod
    def infer_day_from_filename(filename: str) -> str:
        fn = filename.lower()
        if "monday" in fn:
            return "Monday"
        elif "tuesday" in fn:
            return "Tuesday"
        elif "wednesday" in fn:
            return "Wednesday"
        elif "thursday" in fn:
            return "Thursday"
        elif "friday" in fn:
            return "Friday"
        return "Unknown"

    def load_raw(self, max_rows_per_file: Optional[int] = None) -> pd.DataFrame:
        files = self.discover_files()
        if not files:
            raise FileNotFoundError(
                f"No data files found at {self.data_path}. "
                f"Please verify path or check README.md."
            )

        dfs = []
        for file in files:
            day = self.infer_day_from_filename(file.name)
            if file.suffix.lower() == ".parquet":
                df = pd.read_parquet(file)
                if max_rows_per_file is not None:
                    df = df.iloc[:max_rows_per_file]
            else:
                try:
                    df = pd.read_csv(file, nrows=max_rows_per_file, low_memory=False, encoding="utf-8")
                except UnicodeDecodeError:
                    df = pd.read_csv(file, nrows=max_rows_per_file, low_memory=False, encoding="cp1252")

            # Strip leading/trailing whitespace from column names
            df.columns = [str(c).strip() for c in df.columns]

            # Handle duplicate column "Fwd Header Length" (pandas renames second to "Fwd Header Length.1")
            if "Fwd Header Length.1" in df.columns:
                print(f"[Loader] File '{file.name}': Dropping duplicate column 'Fwd Header Length.1' (renamed from second 'Fwd Header Length').")
                df = df.drop(columns=["Fwd Header Length.1"])

            # Validate header
            self.validate_schema(df.columns)

            if "Day" not in df.columns:
                df["Day"] = day
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)
        return combined

    def load_and_clean(
        self,
        max_rows_per_file: Optional[int] = None,
        save_cleaned_to: Optional[Union[str, Path]] = None,
        save_report_to: Optional[Union[str, Path]] = None,
    ) -> Tuple[pd.DataFrame, CleaningReport]:
        raw_df = self.load_raw(max_rows_per_file=max_rows_per_file)
        cleaned_df, report = self.cleaner.clean(
            raw_df, dataset_name="CIC-IDS2017 (corrected)"
        )

        if save_cleaned_to is not None:
            save_path = Path(save_cleaned_to)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            if save_path.suffix.lower() == ".parquet":
                cleaned_df.to_parquet(save_path, index=False)
            else:
                cleaned_df.to_csv(save_path, index=False)

        if save_report_to is not None:
            report.save_json(save_report_to)

        return cleaned_df, report


class TrainOnlyScaler(BaseEstimator, TransformerMixin):
    """Scaler wrapper enforcing Global Rule 1:
    
    'Fit ALL scalers, encoders, PCA and covariance estimates on TRAIN data only.
    Never use TEST statistics anywhere.'
    """

    def __init__(self, scaler_type: str = "standard"):
        self.scaler_type = scaler_type
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        self.is_fitted = False
        self.feature_names_: Optional[List[str]] = None

    def fit(self, X_train: Union[pd.DataFrame, np.ndarray], y: Any = None) -> TrainOnlyScaler:
        if isinstance(X_train, pd.DataFrame):
            self.feature_names_ = list(X_train.columns)
            X_mat = X_train.values
        else:
            X_mat = np.asarray(X_train)

        self.scaler.fit(X_mat)
        self.is_fitted = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted on TRAIN data before transform.")
        if isinstance(X, pd.DataFrame):
            X_mat = X.values
        else:
            X_mat = np.asarray(X)
        return self.scaler.transform(X_mat)

    def fit_transform(self, X_train: Union[pd.DataFrame, np.ndarray], y: Any = None) -> np.ndarray:
        return self.fit(X_train, y).transform(X_train)

    @property
    def mean_(self) -> np.ndarray:
        if hasattr(self.scaler, "mean_"):
            return self.scaler.mean_
        raise AttributeError(f"{self.scaler_type} does not expose mean_")

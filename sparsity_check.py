import pandas as pd
import sys


def analyze_label_sparsity(filepath, property_columns):

    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}", file=sys.stderr)
        return
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        return

    # Check if all specified property columns exist in the DataFrame
    missing_cols = [col for col in property_columns if col not in df.columns]
    if missing_cols:
        print(f"Error: The following columns were not found in the CSV: {missing_cols}")
        print(f"Available columns are: {list(df.columns)}")
        return

    # 1. Count non-nulls for each row in the specified columns
    # .notna() counts non-empty/non-NaN cells as True (1)
    # .sum(axis=1) sums these counts across the row
    filled_count_per_row = df[property_columns].notna().sum(axis=1)

    # 2. Get the distribution of these counts
    # .value_counts() groups by the count (e.g., 0, 1, 2) and counts occurrences
    count_distribution = filled_count_per_row.value_counts().sort_index()

    # 3. Get the same distribution, but as percentages
    percentage_distribution = (
        filled_count_per_row.value_counts(normalize=True).sort_index() * 100
    )

    print("--- 📊 Label Sparsity Report ---")
    print(f"Total rows analyzed: {len(df)}\n")
    print("Properties filled per row:")

    # Combine the two series for a nice printout
    report = pd.DataFrame(
        {"Row Count": count_distribution, "Percentage": percentage_distribution}
    )

    # Ensure all counts from 0 to 5 are present for a full report
    report = report.reindex(range(6), fill_value=0)

    print(report.to_string(formatters={"Percentage": "{:,.2f}%".format}))
    print("\n" + "-" * 35)

    # --- Justification ---
    print("\n--- 💡 Justification for SSL ---")
    try:
        # Get count of rows with 0 labels
        unlabeled_percent = report.loc[0, "Percentage"]
        unlabeled_count = report.loc[0, "Row Count"]

        # Get count of rows with 1-4 labels
        partially_labeled_percent = report.loc[1:4, "Percentage"].sum()
        partially_labeled_count = report.loc[1:4, "Row Count"].sum()

        # Get count of all rows with *at least one* missing label
        total_sparse_count = unlabeled_count + partially_labeled_count
        total_sparse_percent = unlabeled_percent + partially_labeled_percent

        print(
            f"You have {total_sparse_count:.0f} rows ({total_sparse_percent:.2f}%) "
            f"with at least one missing label."
        )
        print(
            f"  • {unlabeled_count:.0f} rows ({unlabeled_percent:.2f}%) are **completely unlabeled**."
        )
        print(
            f"  • {partially_labeled_count:.0f} rows ({partially_labeled_percent:.2f}%) are **partially labeled**."
        )

        print("\nThis sparsity strongly justifies Self-Supervised Learning (SSL).")
        print(
            f"SSL can pre-train on all {len(df)} molecules using their structure alone,"
        )
        print(
            "allowing the model to learn meaningful representations before being fine-tuned"
        )
        print("on this sparse, partially-labeled dataset.")

    except KeyError:
        print("No rows with 0-4 labels found. Your dataset might be fully labeled.")
    except Exception as e:
        print(f"An error occurred generating the justification: {e}")


if __name__ == "__main__":

    # === 1. CONFIGURE THIS ===
    # Set the path to your training file.
    # This path should be relative to where you run this script from.
    TRAIN_FILE_PATH = "data/raw/train.csv"

    # === 2. CONFIGURE THIS ===
    # Set the *exact* names of your 5 property columns from the CSV header.
    # You can find these in your `conf/data/default.yaml` file, under `label_names`.
    PROPERTY_COLUMN_NAMES = [
        "Tg",
        "FFV",
        "Tc",
        "Density",
        "Rg",
    ]

    # === Run Analysis ===
    analyze_label_sparsity(TRAIN_FILE_PATH, PROPERTY_COLUMN_NAMES)

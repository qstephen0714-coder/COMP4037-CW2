from pathlib import Path
import pandas as pd

INPUT_FILE = Path("original_dataset.xlsx")
SHEET_NAME = "Sheet1"
OUTPUT_WIDE = Path("clean_wide.xlsx")
OUTPUT_LONG = Path("long_data.xlsx")


def find_header_row(raw_df: pd.DataFrame) -> int:
    # Find the row containing the main table header
    for i in range(len(raw_df)):
        row_values = raw_df.iloc[i].astype(str).tolist()
        row_text = " | ".join(row_values)
        if "All  diagnoses" in row_text or "All diagnoses" in row_text:
            return i
    raise ValueError("No header row found. Please check the Excel file format.")


def main() -> None:
    raw = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME, header=None)
    header_row = find_header_row(raw)
    print(f"Header row found at Excel row {header_row + 1}.")

    df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME, header=header_row)

    cols = list(df.columns)
    cols[0] = "Diagnosis Code"
    cols[1] = "Diagnosis Description"
    df.columns = cols

    # Remove empty and invalid rows
    df = df.dropna(how="all")
    df = df[df["Diagnosis Code"].notna()]
    df = df[df["Diagnosis Code"].astype(str).str.match(r"^[A-Z][0-9]{2}$", na=False)]

    # Detect age columns
    age_cols = [c for c in df.columns if isinstance(c, str) and c.strip().startswith("Age")]

    # Detect total admissions column
    all_diag_col = None
    for c in df.columns:
        if isinstance(c, str) and "All" in c and "diagnose" in c:
            all_diag_col = c
            break

    if all_diag_col is None:
        raise ValueError("The 'All diagnoses' column was not found.")

    keep_cols = ["Diagnosis Code", "Diagnosis Description", all_diag_col] + age_cols
    clean_wide = df[keep_cols].copy()
    clean_wide = clean_wide.rename(columns={all_diag_col: "All diagnoses"})

    num_cols = ["All diagnoses"] + age_cols
    for col in num_cols:
        clean_wide[col] = pd.to_numeric(clean_wide[col], errors="coerce")

    clean_wide = clean_wide[clean_wide["All diagnoses"].notna()]
    clean_wide.to_excel(OUTPUT_WIDE, index=False)
    print(f"Saved cleaned wide-format data to: {OUTPUT_WIDE}")

    # Reshape from wide to long
    long_df = clean_wide.melt(
        id_vars=["Diagnosis Code", "Diagnosis Description", "All diagnoses"],
        value_vars=age_cols,
        var_name="Age Group",
        value_name="Admissions"
    )

    long_df["Admissions"] = pd.to_numeric(long_df["Admissions"], errors="coerce").fillna(0)
    long_df["Age Share"] = long_df["Admissions"] / long_df["All diagnoses"]

    long_df.to_excel(OUTPUT_LONG, index=False)
    print(f"Saved cleaned long-format data to: {OUTPUT_LONG}")


if __name__ == "__main__":
    main()
from pathlib import Path

import pandas as pd


INPUT_FILE = Path("student_productivity_distraction_dataset_20000.csv")
OUTPUT_FILE = Path("student_productivity_distraction_dataset_20000.csv")
REQUIRED_COLUMNS = ["study_hours_per_day", "sleep_hours", "phone_usage_hours"]


def main() -> None:
	df = pd.read_csv(INPUT_FILE)

	missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
	if missing_cols:
		raise ValueError(f"Missing required columns: {missing_cols}")

	df["main_activity_time"] = (
		df["study_hours_per_day"] + df["sleep_hours"] + df["phone_usage_hours"]
	)

	df.to_csv(OUTPUT_FILE, index=False)
	print(f"Saved {len(df)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
	main()

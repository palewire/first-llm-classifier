"""Download and process campaign finance payee data from biglocalnews.org."""

import os
import sys
from pathlib import Path

import bln
import pandas as pd

bln.pandas.register(pd)

FILE_LIST = ["Form460ScheduleEItem.csv", "Form460ScheduleESubItem.csv"]
PROJECT_ID = "UHJvamVjdDo2MDVjNzdiYS0wODI4LTRlOTEtOGM3OC03ZjA4NGI2ZDEwZWE="

# Output directory: the Sphinx project's _static folder so files are served
# at https://palewi.re/docs/first-llm-classifier/_static/<filename>
OUTPUT_DIR = Path(__file__).parent / "docs" / "_static"


def get_payees(file_name: str) -> None:
    """Download a file from biglocalnews.org and write out a distinct list of business payees."""
    # Get the table from biglocalnews.org via its API.
    print(f"Downloading {file_name}")
    df = pd.read_bln(
        PROJECT_ID,
        file_name,
        os.environ["BLN_API_KEY"],
        dtype=str,
    )
    print(f"- {len(df)} records")

    # Cut out any records that have a first name. They will be people and not businesses. We don't want them.
    nopeople_df = df[pd.isnull(df.payee_firstname)].copy()

    # Get a distinct list of payees.
    distinct_payees = nopeople_df.payee_lastname.str.upper().unique()

    # Convert that back into a DataFrame.
    payee_df = pd.DataFrame(distinct_payees, columns=["payee"]).sort_values("payee")
    print(f"- {len(payee_df)} distinct payees")

    # Write it out into the Sphinx _static directory.
    output_path = OUTPUT_DIR / file_name
    payee_df.to_csv(output_path, index=False)
    print(f"- Written to {output_path}")


def main() -> None:
    """Run the download for all files."""
    if not os.getenv("BLN_API_KEY"):
        sys.exit("Error: BLN_API_KEY environment variable is not set.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for file_name in FILE_LIST:
        get_payees(file_name)


if __name__ == "__main__":
    main()

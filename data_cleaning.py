import os
from mysklearn.mypytable import MyPyTable

def clean_recidivism_data_NA():
    """
    Input data file: ../data/3-Year_Recidivism_for_Offenders_Released_from_Prison_in_Iowa.csv
    Output data file: ../data/cleaned-recidivism-data-NA.csv

    1. Removes the last five columns (not needed)
    2. Swaps the columns "Return to Prison" and "Days to Return" so that the class label is the last column
    3. Replaces empty cells with the string NA
    """
    filename = os.path.join("data", "3-Year_Recidivism_for_Offenders_Released_from_Prison_in_Iowa.csv")
    table = MyPyTable()
    table.load_from_file(filename)

    table.drop_columns(table.column_names[12:])

    table.swap_columns(table.column_names[10], table.column_names[11])

    for column_name in table.column_names:
        table.replace_missing_values_with_NA(column_name)
    write_filename = os.path.join("data", "cleaned-recidivism-data-NA.csv")
    table.save_to_file(write_filename)


if __name__ == "__main__":
    clean_recidivism_data_NA()
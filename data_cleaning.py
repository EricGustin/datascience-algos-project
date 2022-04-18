import os
from mysklearn.mypytable import MyPyTable

def clean_recidivism_data_NA():
    """
    Input data file: ../data/3-Year_Recidivism_for_Offenders_Released_from_Prison_in_Iowa.csv
    Output data file: ../data/cleaned-recidivism-data-NA.csv

    1. Removes the last six columns (not needed)
    3. Replaces empty cells with the string NA
    """
    filename = os.path.join("data", "3-Year_Recidivism_for_Offenders_Released_from_Prison_in_Iowa.csv")
    table = MyPyTable()
    table.load_from_file(filename)

    table.drop_columns(table.column_names[11:])

    for column_name in table.column_names:
        table.replace_missing_values_with_NA(column_name)
    
    yes = 0
    no = 0
    for row in table.data:
        if row[-1] == "Yes":
            yes += 1
        elif row[-1] == "No":
            no += 1
    print({yes / (yes + no) * 100})
    print({no / (no + yes) * 100})


    write_filename = os.path.join("data", "cleaned-recidivism-data-NA.csv")
    table.save_to_file(write_filename)


if __name__ == "__main__":
    clean_recidivism_data_NA()
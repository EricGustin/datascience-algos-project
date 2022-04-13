import copy
import csv
from collections import Counter
from tabulate import tabulate  # uncomment if you want to use the pretty_print() method

# install tabulate with: pip install tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.
    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.
        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure."""
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).
        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.data[0])

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.
        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.
        Returns:
            list of obj: 1D list of values in the column
        Notes:
            Raise ValueError on invalid col_identifier
        """
        col_index = -1
        # allows for negative index values since using negative
        # values to access an element in Python is valid
        if isinstance(col_identifier, int) and col_identifier < self.get_shape()[1]:
            col_index = col_identifier
        else:
            try:
                col_index = self.column_names.index(col_identifier)
            except ValueError as err:
                raise ValueError from err

        col = [row[col_index] for row in self.data]

        if include_missing_values:
            return col

        return [value for value in col if value not in ("NA", "N/A", "")]

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).
        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        num_rows, num_cols = self.get_shape()  # N is num rows, M is num cols
        for row_num in range(num_rows):
            for col_num in range(num_cols):
                try:
                    self.data[row_num][col_num] = float(self.data[row_num][col_num])
                except ValueError:
                    continue

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.
        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        unique_descending_rows = sorted(list(set(row_indexes_to_drop)), reverse=True)
        num_rows, _ = self.get_shape()  # N is num rows, M is num cols
        for row in unique_descending_rows:
            # only remove valid row numbers
            if 0 <= row < num_rows:
                self.data.pop(row)

    def drop_columns(self, col_identifiers):
        """
        Removes columns from the table data

        Parameters:
        -----------
        col_identifiers: list(str or int)
        """
        col_indices = []
        # allows for negative index values since using negative
        # values to access an element in Python is valid
        for col_identifier in col_identifiers:
            if isinstance(col_identifier, int) and col_identifier < self.get_shape()[1]:
                col_indices.append(col_identifier)
            else:
                try:
                    col_indices.append(self.column_names.index(col_identifier))
                except ValueError as err:
                    raise ValueError from err
        # remove rows from data
        for i, row in enumerate(self.data):
            self.data[i] = self.get_partial_row(row, col_indices)
        # remove elements from header
        self.column_names = self.get_partial_row(self.column_names, col_indices)

    def swap_columns(self, col_identifier_1, col_identifier_2):
        """
        Swaps two columns in table data

        Parameters:
        -----------
        col_identifier_1: str or int
        col_identifier_2 str or int
        """
        col_indices = []
        # allows for negative index values since using negative
        # values to access an element in Python is valid
        for col_identifier in (col_identifier_1, col_identifier_2):
            if isinstance(col_identifier, int) and col_identifier < self.get_shape()[1]:
                col_indices.append(col_identifier)
            else:
                try:
                    col_indices.append(self.column_names.index(col_identifier))
                except ValueError as err:
                    raise ValueError from err
        self.column_names[col_indices[0]], self.column_names[col_indices[1]] = self.column_names[col_indices[1]], self.column_names[col_indices[0]]
        for i in range(len(self.data)):
            self.data[i][col_indices[0]], self.data[i][col_indices[1]] = self.data[i][col_indices[1]], self.data[i][col_indices[0]]

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.
        Args:
            filename(str): relative path for the CSV file to open and load the contents of.
        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)
        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        with open(filename, "r") as file:
            reader = csv.reader(file)
            self.column_names = next(reader)
            self.data = list(reader)

        self.convert_to_numeric()

        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.
        Args:
            filename(str): relative path for the CSV file to save the contents to.
        Notes:
            Use the csv module.
        """
        num_rows, num_cols = self.get_shape()  # N is num rows, M is num cols
        with open(filename, "w") as file:
            # write the column names to file
            for col_num in range(num_cols - 1):
                file.write(self.column_names[col_num] + ",")
            file.write(self.column_names[col_num + 1] + "\n")
            # write the data to file
            for row_num in range(num_rows):
                for col_num in range(num_cols - 1):
                    file.write(str(self.data[row_num][col_num]) + ",")
                file.write(str(self.data[row_num][col_num + 1]) + "\n")

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.
        Args:
            key_column_names(list of str): column names to use as row keys.
        Returns
            list of int: list of indexes of duplicate rows found
        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        seen = set()
        duplicate_indexes = []
        columns = [self.get_column(name) for name in key_column_names]
        rows = list(zip(*columns))
        for i, row in enumerate(rows):
            if row not in seen:
                seen.add(row)
            else:
                duplicate_indexes.append(i)

        return duplicate_indexes

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA")."""
        num_rows, _ = self.get_shape()  # N is num rows, M is num cols
        for row_num in range(num_rows - 1, -1, -1):
            if "NA" in self.data[row_num]:
                self.data.pop(row_num)

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.
        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        column = self.get_column(col_name)
        column_without_na = self.get_column(col_name, include_missing_values=False)
        column_average = sum(column_without_na) / len(column_without_na)
        column_index = self.column_names.index(col_name)

        for i, value in enumerate(column):
            if value == "NA":
                self.data[i][column_index] = column_average

    def replace_missing_values_with_NA(self, col_name):
        """
        Wherever there is a missing value in the column, replace
        it with the string "NA"

        Parameters:
        -----------
        col_name(str): name of column to fill with the original average (of the column).
        """
        column = self.get_column(col_name)
        column_index = self.column_names.index(col_name)
        for i, value in enumerate(column):
            if value == "":
                self.data[i][column_index] = "NA"

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.
        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]
        """
        columns = [self.get_column(col_name) for col_name in col_names]
        stat_table = MyPyTable(
            column_names=["attribute", "min", "max", "mid", "avg", "median"]
        )
        if not columns[0]:
            return stat_table

        for i, column in enumerate(columns):
            mid = (min(column) + max(column)) / 2
            avg = sum(column) / len(column)
            median = 0
            sorted_column = sorted(column)
            if len(column) % 2:
                # compute median for odd lengthed column
                median = sorted_column[len(column) // 2]
            else:
                # compute median for even lengthed column
                median = (
                    sorted_column[len(column) // 2 - 1]
                    + sorted_column[len(column) // 2]
                ) / 2
            stat_table.data.append(
                [col_names[i], min(column), max(column), mid, avg, median]
            )

        return stat_table

    def get_joined_column_names(self, other_table):
        """
        Given self and another table, join their header
        Parameters:
        -----------
        other_table: MyPyTable
        Returns:
        --------
        list[str]
        """
        joined_column_names = self.column_names
        for column_name in other_table.column_names:
            if column_name not in joined_column_names:
                joined_column_names.append(column_name)

        return joined_column_names

    @staticmethod
    def is_inner_join_match(left_indexes, right_indexes, left_row, right_row):
        """
        Check if two rows from two different tables match on their key
        Parameters:
        -----------
        left_indexes: list[int] indexes for the left table's keys
        right_indexes: list[int] indexes for the right table's keys
        left_row: list[obj] a single row from the left table
        right_row: list[obj] a single row from the right table
        Returns:
        --------
        Bool
        """
        for key_indexes in zip(left_indexes, right_indexes):
            if left_row[key_indexes[0]] != right_row[key_indexes[1]]:
                return False
        return True

    @staticmethod
    def get_partial_row(row, exclude_indices):
        """
        This function takes in a row and returns as a list the elements at the given indexes
        Parameters:
        -----------
        row: list[obj] A row from a table
        indexes: list[int] the indexes for values in row to return
        Returns:
        --------
        list[obj]
        """
        row_without_key_cols = []
        for i, value in enumerate(row):
            if i not in exclude_indices:
                row_without_key_cols.append(value)
        return row_without_key_cols

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.
        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.
        Returns:
            MyPyTable: the inner joined table.
        """
        joined_column_names = self.get_joined_column_names(other_table)
        # indexes in the left table that correspond to the table's keys
        left_indexes = [
            self.column_names.index(column_name) for column_name in key_column_names
        ]
        # indexes in the right table that correspond to the table's keys
        right_indexes = [
            other_table.column_names.index(column_name)
            for column_name in key_column_names
        ]

        joined_data = []
        for left_row in self.data:
            for right_row in other_table.data:
                is_match = self.is_inner_join_match(
                    left_indexes, right_indexes, left_row, right_row
                )
                if is_match:
                    # add the joined rows to joined_data if there was a match
                    right_row_without_key_cols = self.get_partial_row(
                        right_row, right_indexes
                    )
                    joined_data.append(left_row + right_row_without_key_cols)

        return MyPyTable(joined_column_names, joined_data)

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.
        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.
        Returns:
            MyPyTable: the fully outer joined table.
        Notes:
            Pad the attributes with missing values with "NA".
        """
        # do an inner join first, and then do the outer join parts of the algorithm later
        joined_table = self.perform_inner_join(other_table, key_column_names)
        # indexes in the left table that correspond to the table's keys
        left_indexes = [
            self.column_names.index(column_name) for column_name in key_column_names
        ]
        # indexes in the right table that correspond to the table's keys
        right_indexes = [
            other_table.column_names.index(column_name)
            for column_name in key_column_names
        ]
        # padding to append to a left table row when there is not a match
        right_pad = ["NA"] * (other_table.get_shape()[1] - len(key_column_names))

        # fill right side with NA on left_rows that do not have a match
        for left_row in self.data:
            has_match = False
            for right_row in other_table.data:
                is_match = self.is_inner_join_match(
                    left_indexes, right_indexes, left_row, right_row
                )
                if is_match:
                    has_match = True
                    break
            if not has_match:
                joined_table.data.append(left_row + right_pad)

        # fill left side with NA on right_rows that do not have a match
        for right_row in other_table.data:
            # for each row in the right table, if there does not exist a match with
            # the left table, then add the row to joined_table
            has_match = False
            right_row_without_key_cols = self.get_partial_row(right_row, right_indexes)
            for left_row in self.data:
                is_match = self.is_inner_join_match(
                    left_indexes, right_indexes, left_row, right_row
                )
                if is_match:
                    has_match = True
                    break
            if not has_match:
                # the ordering of columns can be tricky. Start with a row filled
                # with NAs, and then build up the row with the correct values at
                # the correct indices
                joined_row = ["NA"] * len(joined_table.column_names)
                for i, right_index in enumerate(right_indexes):
                    joined_row[left_indexes[i]] = right_row[right_index]
                joined_row[
                    len(joined_row) - len(right_row_without_key_cols) :
                ] = right_row_without_key_cols

                joined_table.data.append(joined_row)

        return joined_table

    def get_column_frequency(self, col_identifier, include_missing_values=False):
        """Extracts a column from the table data and returns a dictionary of each
        value's frequency.
        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.
        Returns:
            dict - a value of a given attribute maps to the value's frequency
        Notes:
            Raise ValueError on invalid col_identifier
        """
        column = self.get_column(col_identifier, include_missing_values)
        return Counter(column)


if __name__ == "__main__":
    table = MyPyTable().load_from_file("./input_data/auto-mpg.txt")

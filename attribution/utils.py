"""
Helper functions and tools
"""
from functools import reduce
from pyspark.sql import DataFrame

def union_all(*data_frames):
  """
  Union 2 or more PySpark data frames. All data frames
  must contain the same columns.
  
  Args:
    data_frames (list)
  
  Returns:
    all_data (pyspark.sql.DataFrame):
  """

  all_data = reduce(
    DataFrame.unionAll,
    data_frames
  )

  return all_data

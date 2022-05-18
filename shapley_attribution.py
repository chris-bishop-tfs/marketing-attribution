# Databricks notebook source
# Import libraries
import datetime
import pandas as pd
import pyspark.sql.functions as sf
from pyspark.sql.window import Window
from pyspark.sql.types import BooleanType, DateType
from data_io import build_connection
from dateutil.relativedelta import relativedelta
from functools import reduce
from pyspark.sql import DataFrame

# Create touchpoints 

# Let's start with the transaction logs
# - recipient_id
# - event_time
# - player_id

# COMMAND ----------

# Let's pull in some raw data
campaign_table = (
  spark
  .createDataFrame(
    pd.read_csv('/dbfs/FileStore/tables/temp/campaign_table.csv')
  )
  .withColumnRenamed(
    'description',
    'campaign_type'
  )
  # Reduce description to A/B/C
  .withColumn(
    'campaign_type',
    sf.upper('campaign_type').substr(5, 1)
  )
  .withColumnRenamed(
    'campaign',
    'campaign_id'
  )
  .withColumnRenamed(
    'household_key',
    'household_id'
  )
).cache()

# We'll allow for lagging effects, such as a campaign having effects after the campaign ends
# The value here is arbitrary, the user can change this at their discretion
N_RESPONSE_DAYS = 90

# Start date is arbitrary, just wanted something more "human-readable"
START_DATE = datetime.date(2020, 1, 1)

# UDF to convert number of days to a calendar date
days_to_date = sf.udf(lambda x: START_DATE + relativedelta(days=x-1), DateType())

# Let's pull in some raw data
campaign_desc = (
  spark
  .createDataFrame(
    pd.read_csv('/dbfs/FileStore/tables/temp/campaign_desc.csv')
  )
  .withColumnRenamed(
    'description',
    'campaign_type'
  )
  # Reduce description to A/B/C
  .withColumn(
    'campaign_type',
    sf.upper('campaign_type').substr(5, 1)
  )
  .withColumnRenamed(
    'start_day',
    'window_start'
  )
  .withColumnRenamed(
    'end_day',
    'window_end'
  )
  # Add in response time
  .withColumn(
    'window_end',
    sf.col('window_end') + sf.lit(N_RESPONSE_DAYS)
  )
  .withColumn(
    'window_start',
    days_to_date(sf.col('window_start'))
  )
  .withColumn(
    'window_end',
    days_to_date(sf.col('window_end'))
  )
#   # Arbitrary start date for demonstrative purposes
#   .withColumn(
#     '_start_date',
#     sf.lit(datetime.date(2020, 1, 1))
#   )
#   .withColumn(
#     'transaction_day',
#     # EXPR required here, see https://issues.apache.org/jira/browse/SPARK-26393
#     sf.expr(
#       "date_add(_start_date, day - 1)"
#     )
#   )
  .withColumnRenamed(
    'campaign',
    'campaign_id'
  )
  .select(
    'campaign_id',
    'campaign_type',
    'window_start',
    'window_end'
  )
  .orderBy(
    'campaign_id'
  )
).cache()

transaction_data = (
  spark
  .createDataFrame(
    pd.read_csv('/dbfs/FileStore/tables/temp/transaction_data.csv')
  )
  .withColumn(
    'day',
    sf.col('day').cast('integer')
  )
  .withColumnRenamed(
    'household_key',
    'household_id'
  )
  .withColumnRenamed(
    'sales_value',
    'revenue_usd'
  )
  .withColumn(
    'transaction_day',
    days_to_date(sf.col('day'))
  )
).cache()

# Only need daily-level granularity
transaction_summary = (
  transaction_data
  .groupBy(
    'household_id',
    'transaction_day'
  )
  .agg(
    sf.sum('revenue_usd').alias('revenue_usd')
  )
  .orderBy(
    'household_id',
    'transaction_day'
  )
).cache()

display(transaction_summary.limit(10))

# COMMAND ----------

# Let's see if we have overlapping campaigns
master_campaign = (
  campaign_table
  .join(
    campaign_desc.select('campaign_id', 'window_start', 'window_end'),
    how='left',
    on='campaign_id'
  )
  # We need to determine if there is OVERLAP between campaigns of the same type
  # If so, then consolidate into a single record. This will make the window
  # creation logic easier below.
  .withColumn(
    'window_end_lag',
    sf.lag('window_end').over(
      Window()
      .partitionBy(
        'household_id',
        'campaign_type'
      )
      .orderBy(
        'window_start'
      )
    )
  )
  # Does the previous campaign end AFTER the next campaign begins?
  .withColumn(
    'is_overlap',
    sf.col('window_end_lag') >= sf.col('window_start')
  )
  .withColumn(
    'campaign_type_lag',
    sf.lag('campaign_type').over(
      Window()
      .partitionBy(
        'household_id'
      )
      .orderBy(
        'window_start'
      )
    )
  )
  .withColumn(
    'is_diff_type',
    sf.col('campaign_type_lag') != sf.col('campaign_type')
  )
  .fillna(
    False,
    subset=['is_overlap', 'is_diff_type']
  )
  .withColumn(
    'is_new_window',
    (
      (~sf.col('is_overlap')) |
      (
        sf.col('is_overlap') &
        sf.col('is_diff_type')
      )
    )
  )
  # Add in a campaign_record_id
  .withColumn(
    'campaign_record_id',
    sf.sum((sf.col('is_new_window')).cast('integer')).over(
      Window()
      .orderBy(
        'household_id',
        'window_start'
      )
      .rowsBetween(
        Window.unboundedPreceding,
        0
      )
    )
  )
  .orderBy(
    'household_id',
    'window_start'
  )
).cache()

# Consolidate campaign_records so we have a single record per time slot/campaign type
# This makes window creation simpler below.
master_campaign = (
  master_campaign
  .groupBy(
    'campaign_record_id',
    'household_id',
    'campaign_type'
  )
  .agg(
    sf.min('window_start').alias('window_start'),
    sf.max('window_end').alias('window_end')
  )
).cache()

# Spot check with a household_id we know has some overlapping campaign_types
display(master_campaign.filter(sf.col('household_id') == 1).orderBy('window_start'))

# COMMAND ----------

# Let's build our "treatment" windows for attribution purposes
window_map = (
  master_campaign.alias('a')
  .join(
    master_campaign.alias('b'),
    on=(
      (
        sf.col('b.window_start').between(sf.col('a.window_start'), sf.col('a.window_end'))
      ) &
      (
        sf.col('a.household_id') == sf.col('b.household_id')
      )
    )
  )
  .select(
    'a.household_id',
    sf.col('a.campaign_record_id').alias('window_id'),
    'a.window_start',
    'a.window_end',
    'b.campaign_record_id',
    'b.campaign_type'
  )
  # We'll assign a new 'player_id' column to conform to package requirements
  .withColumn(
    'player_id',
    sf.col('campaign_type')
  )
  .orderBy(
    'window_id',
    'window_start'
  )
).cache()

# Convert player columns into boolean values
player_ids = window_map.select('player_id').dropDuplicates().toPandas()['player_id'].tolist()

# Sort so I can stop ripping my hair out
player_ids.sort()

# Now that we have the windows mapped, we need to pivot into a wide data frame for attribution
attribution_window = (
  window_map
  .groupBy(
    'household_id',
    'window_id',
    'window_start',
    'window_end'
  )
  .pivot(
    'player_id'
  )
  .count()
  .fillna(
    0,
    subset=player_ids
  )
).cache()

# Convert player columns to boolean values
for p in player_ids:

  attribution_window = (
    attribution_window
    .withColumn(
      p,
      sf.col(p) > 0
    )
  ).cache()

display(attribution_window)

# COMMAND ----------

display(transaction_summary.groupBy('household_id').agg(sf.sum('revenue_usd')).describe())

# COMMAND ----------

# Now we need REVENUE per window
revenue_to_window = (
  attribution_window.select('household_id', 'window_id', 'window_start', 'window_end').alias('a')
  .join(
    transaction_summary.alias('b'),
    on=(
      sf.col('b.transaction_day').between(
        sf.col('a.window_start'),
        sf.col('a.window_end')
      ) &
      (
        sf.col('a.household_id') == sf.col('b.household_id')
      )
    ),
    how='left'
  )
  # Sum up revenue per window
  .groupBy(
    # Household_id here mostly for spot checking
    'a.household_id',
    'window_id'
  )
  .agg(
    sf.sum('revenue_usd').alias('revenue_usd')
  )
  # Truncate to dollars so I don't have to stare at floating point numbers
  .withColumn(
    'revenue_usd',
    sf.col('revenue_usd').cast('long')
  )
).cache()

display(revenue_to_window)

# COMMAND ----------

# Finally, build master attribution data
master_attribution = (
  attribution_window
  .join(
    revenue_to_window,
    on=['household_id', 'window_id'],
    how='left'
  )
  .fillna(
    0,
    'revenue_usd'
  )
).cache()

display(master_attribution.filter(sf.col('revenue_usd') == 0))
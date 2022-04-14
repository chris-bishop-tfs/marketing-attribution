# Databricks notebook source
# Import libraries
import datetime
import pandas as pd
import pyspark.sql.functions as sf
from pyspark.sql.window import Window
from pyspark.sql.types import BooleanType, DateType
from data_io import build_connection
from dateutil.relativedelta import relativedelta
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

# COMMAND ----------

from pyspark.sql import SparkSession

SparkSession.getActiveSession()

# COMMAND ----------

# OK, now let's start developing against this master data set
import abc
from attr import attrs, attrib
from itertools import combinations, permutations
from pyspark.sql import SparkSession

@attrs
class Valuator(abc.ABC):
  """
  Valuators will perform journey evaluation
  """
  
  # Need to know player_ids
  player_ids = attrib(default=[])
  
  # Can assign journeys if we have them - otherwise, run build_store_journeys
  journeys = attrib(default=None)

  # Spark session to use in computations
  spark = attrib(default=SparkSession.getActiveSession())

  @abc.abstractmethod
  def build_journeys(self):
    """
    Build possible journey
    """
    
    pass

#   @abc.abstractclassmethod
#   def compute_value(self, data, id_col=None, val_col=None):
    
#     pass

# Store standalone functions so we can strip these out of we choose
# We need to get a list of all possible coalitions
def build_journeys(player_ids, max_journey_length=None):
  """
  Returns all possible journeys.
  
  Note: Bishop later changed this such that ORDER doesn't matter,
  and only the included touch points do. When we start adding in
  other fields, like divisions, etc. the permutations starts to
  explode (to the point of running out of memory)

  Args:
  """

  if max_journey_length is None:
      max_journey_length = len(player_ids)

  journeys = []

  # Build all possible arrival orders up to a certain size (max_)
  for i in range(1, max_journey_length + 1):
    journeys.extend(map(list, combinations(player_ids, i)))

  return list(journeys)

@attrs
class Shapley(Valuator):
  """
  Shapley marginal value
  """

  def build_journeys(self, journey_delimiter=None, *largs, **kwargs):
    """
    Create a PySpark dataframe that describes
    """

    # Set default delimiter
    if journey_delimiter is None:
      journey_delimiter = '|'

    journeys = build_journeys(self.player_ids, *largs, **kwargs)
    
    # Sort each journey
    # Improves readability
    for l in journeys:
      l.sort()

    # Convert journeys to a PySpark dataframe
    journeys_pd = (
      pd.DataFrame(dict(journey=journeys))
    )
    
    # Add journey_id
    journeys_pd['journey_id'] = list(range(len(journeys)))
    
    # Add journey name
    journeys_pd['journey_name'] = (
      journeys_pd
      .journey
      .apply(
        lambda x: journey_delimiter.join(x)
      )
    )

    journeys_df = (
      self
      .spark
      .createDataFrame(
        journeys_pd
      )
      .withColumn(
        'journey',
        sf.explode('journey')
      )
      .groupBy(
        'journey_id',
        'journey_name'
      )
      .pivot(
        'journey'
      )
      .count()
      # Fill missing values
      .fillna(
        0,
        subset=self.player_ids
      )
    )
    
    # Convert to boolean values
    for p in self.player_ids:
      journeys_df = (
        journeys_df
        .withColumn(
          p,
          sf.col(p) > 0
        )
      )

    # Add cardinality
    journeys_df = (
      journeys_df
      .withColumn(
        'cardinality',
        self._cardinality_expression
      )
      .select(
        'journey_id',
        'journey_name',
        'cardinality',
        *self.player_ids
      )
    )

    return journeys_df
  
  @property
  def _cardinality_expression(self):
    """
    Cardinality expression for player set
    """
    for i, p in enumerate(self.player_ids):
      
      _c = (sf.col(p).cast('integer'))
      
      if i == 0:
        cardinality_expression = _c
      else:
        cardinality_expression += _c

    return cardinality_expression

  def build_store_journeys(self, *largs, **kwargs):
    
    # Build journeys_df
    journeys_df = self.build_journeys(*largs, **kwargs)
    
    # Assign to attribute
    self.journeys = journeys_df.orderBy('journey_id')

#   def build_subsets(self):
    
#     journey_subsets = (
#       self
#       .journey
#       .select(
#         'journey_id',
        
#       )
#     )
# Random tests
v = Shapley(player_ids)
# v._cardinality_expression

v.build_store_journeys()

display(v.journeys)


# COMMAND ----------



# COMMAND ----------

def compute_conversion(data):
# def compute_value(data):
  """
  Compute the coalition-specific conversion rates.
  These will later be used to estimate the "worth" of
  each journey and, ultimately, the marginal value of
  each channel.
  
  Note that the *denominator* for each channel is the *total
  number of opportunites*. (Pool opportunities over channels).

  Args:
    data: pass in the equivalent of master_attribution
    group_by: coalition column name ('coalition_id')
  
  Returns:
    group_conversion: conversion rates per group (e.g., coalition)
  """

  # Conversion here is DIFFERENT than value in the Shapley sense
  # But we need conversions to get to worth
  coalition_conversion = (
    data
    # XXX
    # Bishop isn't 100% sure NULLs should be here
    # XXX Bishop is now sure that NULLs are here and
    # are expected - but they should be filetered out
    # elsewhere.
    .filter(
      sf.col('coalition_id').isNotNull()
    )
    .withColumn(
      '_email_order',
      sf.when(
        sf.col('is_order'),
        sf.col('email_uid')
      )
    )
    # We'll estimate conversion rates per coalition and per group_by
    .groupby(
      'coalition_id',
      *group_by
    )
    .agg(
#       sf.countDistinct('_email_order').alias('n_orders'),
#       sf.countDistinct('email_uid').alias('n_opportunities')
      sf.sum(sf.col('is_order').cast('integer')).alias('n_orders'),
      sf.count('*').alias('n_opportunities'),
      # Also want to aggregate revenue (what we care about most directly)
      sf.sum('revenue').alias('revenue')
    )
    .withColumn(
      'total_opportunities',
      sf.sum('n_opportunities').over(
        Window()
        .partitionBy(
          *group_by
        )
        .rowsBetween(
          Window.unboundedPreceding,
          Window.unboundedFollowing
        )
      )
    )
    .withColumn(
      'total_orders',
      sf.sum('n_orders').over(
        Window()
        .partitionBy(
          *group_by
        )
        .rowsBetween(
          Window.unboundedPreceding,
          Window.unboundedFollowing
        )
      )
    )
    .withColumn(
      'total_revenue',
      sf.sum('revenue').over(
        Window()
        .partitionBy(
          *group_by
        )
        .rowsBetween(
          Window.unboundedPreceding,
          Window.unboundedFollowing
        )
      )
    )
    # This is counter intuitive, but we want the proportion
    # of VALUE in each cohort.
    #
    # XXX The denominator here needs more thought. Why shouldn't
    # it be `total_opportunities`
    .withColumn(
      'p_conversion',
#       sf.col('n_orders') / sf.col('total_orders')
      sf.col('n_orders')/ sf.col('total_opportunities')
    )
    # Compute proportion of revenue as well
    .withColumn(
      'p_revenue',
      sf.col('revenue') / sf.col('total_revenue')
    )
#     .withColumn(
#       'p_conversion',
#       sf.col('n_orders') / sf.col('n_customers')
#     )
    # Need coalition information
    # master_coalition won't change, so we'll just use whatever is in scope here
#     .join(
#       master_coalition,
#       on='coalition_id',
#       how='left'
#     )
  ).cache()

# COMMAND ----------

campaign_overlap = (
  master_campaign.alias('a')
  # Self join
  .join(
    master_campaign.alias('b'),
    on=(
      (
        sf.col('b.window_start').between(
          sf.col('a.window_start'),
          sf.col('a.window_end')
        ) &
        (
          sf.col('a.household_id') == sf.col('b.household_id')
        )
      )
    ),
    how='left'
  )
  .select(
    'a.household_id',
    'a.campaign_record_id',
    'b.campaign_type',
    'b.window_start',
    'b.window_end'
  )
  .orderBy(
    'campaign_record_id',
    'window_start'
  )
  # Reduce
#   .filter(
#     sf.col('a.campaign_type') != sf.col('b.campaign_type')
#   )
).cache()

display(campaign_overlap)

# COMMAND ----------

campaign_overlap = (
  master_campaign
  .withColumn(
    'window_end_lag',
    sf.lag('window_end').over(
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
  # Does the previous campaign end AFTER the next campaign begins?
  .withColumn(
    'is_overlap',
    sf.col('window_end_lag') >= sf.col('window_start')
  )
  .withColumn(
    'is_diff_type',
    sf.col('campaign_type_lag') != sf.col('campaign_type')
  )
  # For single treatments, we'll see nulls here
  .fillna(
    False,
    subset=['is_overlap', 'is_diff_type']
  )
  .orderBy(
    'household_id',
    'window_start'
  )
).cache()

display(campaign_overlap.filter(sf.col('household_id') == 1))
# display(campaign_overlap.filter(sf.col('is_diff_type') & sf.col('is_overlap')))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC OK so it looks like have genuine windows during which multiple campaigns are active and have been applied to a single household. The question is now "what's the value of each campaign type"?

# COMMAND ----------

# We need to break up campaign treatment into attribution windows
attribution_window = (
  
#   campaign_overlap
#   .withColumn(
#     'is_new_window',
#     (
#       (~sf.col('is_overlap')) |
#       (sf.col('is_overlap') & sf.col('is_diff_type'))
#     )
#   )
  
  # Assign an attribution window ID
  .withColumn(
    'window_id',
    sf.sum(sf.col('is_new_window').cast('integer')).over(
      Window()
      .orderBy(
        'household_id',
        'window_start'
      )
    )
  )
  # First window will be NULL, assign 0
  .fillna(
    0,
    subset=['window_id']
  )
  # Add a player_id column here
  # We want to have an independent column in case we do something more complex later
  .withColumn(
    'player_id',
    sf.col('campaign_type')
  )
).cache()

# Need a window summary
attribution_window_summary = (
  attribution_window
  .groupBy(
    'window_id'
  )
  .agg(
    sf.min('window_start').alias('window_start_min'),
    sf.max('window_start').alias('window_start_max'),
    sf.min('window_end').alias('window_end_min'),
    sf.max('window_end').alias('window_end_max'),
    sf.countDistinct('player_id').alias('n_players')
  )
)
display(attribution_window_summary.filter(sf.col('n_players') > 1))

# COMMAND ----------

display(
  attribution_window
  .filter(sf.col('household_id') == 1)
  .limit(10)
)

# COMMAND ----------

# Pivot into wide table
attribution_window_wide = (
  attribution_window
  .groupBy(
    'household_id',
    'window_id'
  )
  .pivot(
    'campaign_type'
  )
  .count()
  .fillna(
    0,
    subset=['A', 'B', 'C']
  )
).cache()

# Convert 
display(attribution_window_wide)

# COMMAND ----------



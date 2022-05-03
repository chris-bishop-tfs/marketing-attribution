"""
Base classes required for attribution
"""
import abc
from numbers import Number
from attr import define
import pyspark.sql.functions as sf
from pyspark.sql import DataFrame
from .audience import build_audience, Audience
from .builder import BaseBuilder
from .journey import build_journey_set
from .utils import union_all

@define
class Valuator(abc.ABC):
  """
  """

  # Valuation is done based on an audience
  # Recall that an audience is a group of journeys
  # with variable assigned value
  # audience = attrib()
  audience: Audience

  @abc.abstractmethod
  def valuate_treatments(self) -> dict:
    """
    Return the value of each treatment
    """
    
    raise NotImplemented

  def _initialize_values(self, start_val: int=0) -> dict:
    """
    Initialize return value for each treatment
    """

    treatments = self.audience.treatments

    for i, t in enumerate(treatments):

      if i == 0:

        value = dict()
      
      value[t] = start_val
    
    return value


@define
class LastTouchValuator(Valuator):
  """
  Full value alloted to most recent impression
  """

  def valuate_treatments(self):
    """
    Evaluate all treatments
    """

    treatment_value = self._initialize_values()

    for m in self.audience.members:

      impressions = m.journey.impressions
      value = m.journey.value

      # Apply full value to last impression
      last_impression = impressions[-1]

      treatment_value[last_impression] += value
    
    return treatment_value


@define
class FirstTouchValuator(Valuator):
  """
  Full value alloted to first impression
  """

  def valuate_treatments(self):
    """
    Evaluate all treatments
    """

    treatment_value = self._initialize_values()

    for m in self.audience.members:

      impressions = m.journey.impressions
      value = m.journey.value

      # Apply full value to last impression
      first_impression = impressions[0]

      treatment_value[first_impression] += value
    
    return treatment_value

@define
class ShapleyValuator(object):
  """
  This valuator will break the standard valuator mold a bit
  """

  journey_summary:  DataFrame
  journey_map: DataFrame

  # Pass in treatment identifiers
  treatments: tuple

  def _compute_worth(
    self,
    journey_summary: DataFrame,
    journey_map: DataFrame,
    val_col: str='value'
  ) -> DataFrame:
    """
    Compute worth of journeys as a total of itself and
    all possible "sub journeys"

    Make this a method so we can leverage it elsewhere
    """

    journey_worth = (
      journey_summary.alias('a')
      .join(
        journey_map.alias('b'),
        on=(
          (sf.col('a.journey_id') == sf.col('b.subset_id'))
        ),
        how='right'
      )
      .groupBy(
        'b.journey_id'
      )
      .agg(
        sf.sum(val_col).alias(val_col)
      )
    )

    return journey_worth

  def _valuate_treatment(
    self,
    treatment: str,
    val_col: str='value'
  ) -> Number:
    """
    Valuate a single treatment type
    """
    
    # Assign worth to each journey
    journey_worth = (
      self
      ._compute_worth(
        self.journey_summary,
        self.journey_map
      )
       # XXX I hate that I'm doing this,
       # but this will improve subsequent operations
      .repartition(
        20,
        'journey_id'
      )
    )

    # Replace existing value with "worth" estimate
    journeys_enriched = (
      self
      .journey_summary
      .drop(
        val_col
      )
      .join(
        journey_worth,
        on='journey_id',
        how='left'
      )
    )

    # We need to find all journeys that include the player of interest
    treatment_journeys = (
      journeys_enriched
      # These are boolean values
      .filter(
        sf.col(treatment)
      )
      # The value column here will be the total value col
      .withColumn(
        f'{val_col}_total',
        sf.col(val_col)
      )
    )

    reference_journeys = (
      journeys_enriched
      .select(
        *self.treatments,
        sf.col('journey_id').alias('journey_id_ref'),
        sf.col(val_col).alias(f'{val_col}_ref')
      )
      # We need S - player_id sets
      .filter(
        ~sf.col(treatment)
      )
      # We'll be joining below, so drop the player column to avoid duplicate names
      .drop(treatment)
    )
    
    # Combine
    marginal_value = (
      treatment_journeys
      .join(
        reference_journeys,
        # We join based on the journey SUBSET
        on=(
          list(
            set(self.treatments) - {treatment}
          )
        ),
        how='left'
      )
      # If the reference value is missing, then it's 0
      .fillna(
        0,
        subset=f'{val_col}_ref'
      )
      .withColumn(
        f'{val_col}_margin',
        sf.col(f'{val_col}_total') - sf.col(f'{val_col}_ref')
      )
      # Cardinality of player set (|N|)
      .withColumn(
        '_n',
        sf.lit(len(self.treatments))
      )
      .withColumn(
        '_s',
        # Exclude the player under consideration
        # Recall that 'S' is S - player
        sf.col('cardinality') - sf.lit(1)
      )
      # Compute weight so all contributions sum to 1
      .withColumn(
        'weight',
        (
          sf.factorial(sf.col('_s')) * sf.factorial(sf.col('_n') - sf.col('_s') - 1)
        ) / (sf.factorial(sf.col('_n')))
      )
      .withColumn(
        f'{val_col}_margin_w',
        sf.col(f'{val_col}_margin') * sf.col('weight')
      )
      .agg(
        sf.sum(sf.col(f'{val_col}_margin_w')).alias(f'{val_col}_margin_w')
      )
    )

    # Extract the only value we care about
    treatment_value = marginal_value.toPandas()[f'{val_col}_margin_w'].iloc[0]

    return treatment_value

  def valuate_treatments(self):
    """
    Valuate all treatments
    """

    for i, t in enumerate(self.treatments):

      if i == 0:
        treatment_values = dict()
      
      _v = self._valuate_treatment(t)

      treatment_values[t] = _v

    return treatment_values


@define
class AudiencetoShapley(BaseBuilder):
  """
  We need to instantiate a Shapley
  """

  audience: Audience

  def build(self) -> ShapleyValuator:
    """
    Convert an audience to appropriate inputs for Shapley
    """

    journey_set = self._build_journey_set()
    jj_map = self._build_journey_map()

    audience_data = self.audience.to_pyspark()  
  
    # Convert to boolean values to mesh with journey data
    treatments = self.audience.treatments

    # Convert to boolean values
    attribution_data = audience_data

    for i, t in enumerate(treatments):

      attribution_data = (
        attribution_data
        .withColumn(
          t,
          sf.col(t).isNotNull()
        )
      )

    # Create the attribution set
    attribution_data = (
      self
      .append_journey_id(
        attribution_data
      )
    )

    journey_summary = (
      self
      ._summarize_journeys(
        attribution_data,
        journey_set
      )
    )

    # Instantiate valuator
    return ShapleyValuator(
      journey_summary,
      jj_map,
      treatments
    )

  def _summarize_journeys(
    self,
    attribution_data: DataFrame,
    journey_set: DataFrame
  ) -> DataFrame:
    """
    Get value by journey_id
    """

    # Journey value
    journey_value = (
      attribution_data
      .groupBy(
        'journey_id'
      )
      .agg(
        sf.sum('value').alias('value')
      )
    )

    # Left join to journey_set so we can get 0 value items
    journey_summary = (
      (
        journey_set
        .withColumnRenamed(
          'identifier',
          'journey_id'
        )
      )
      .join(
        journey_value,
        on='journey_id',
        how='left'
      )
      .fillna(
        0,
        subset=['value']
      )
    )

    return journey_summary

  @property
  def _cardinality_expression(self):
    """
    Journey cardinality expression
    """

    treatments = self.audience.treatments

    for i, t in enumerate(treatments):
      
      # Recall that missing treatments will be NULL
      _c = (sf.col(t).cast('integer'))
      
      if i == 0:
        cardinality_expression = _c
      else:
        cardinality_expression += _c

    return cardinality_expression

  def _build_journey_set(self) -> DataFrame:
    """
    Build out possible journey sets.
    """

    treatments = self.audience.treatments

    # All journey combinations
    possible_journeys = build_journey_set(treatments)

    # Massage into a data frame with identifiers, etc.
    #  We'll leverage an "audience" to do this
    for i, j in enumerate(possible_journeys):
      if i == 0:
        data = list()
      
      # Append a sequential identifier
      _data = dict(identifier=i, **j.to_dict())
      data.append(_data)

    journey_set = (
      build_audience(data)
      .to_pyspark()
      # "Value" here doesn't have any meaning
      # Remove it.
      .drop(
        'value'
      )
    )

    # Convert treatment columns to BOOLEAN values
    # In our case, we no longer care about order
    for t in treatments:
      journey_set = (
        journey_set
        .withColumn(
          t,
          sf.col(t).isNotNull()
        )
      )

    # Add cardinality
    journey_set = (
      journey_set
      .withColumn(
        'cardinality',
        self._cardinality_expression
      )
    )

    return journey_set

  def _build_journey_map(self, *largs, **kwargs) -> DataFrame:
    """
    Build journey-to-journey map for valuation purposes
    """

    # Journey sets
    journey_set = self._build_journey_set()

    # Repartition
    # XXX Fix hard-coded value
    journey_set = journey_set.repartition(20, 'identifier')

    # Map individual journeys 
    jj_map = (
      journey_set.select(
        'identifier',
        # Needed for subset creation
        'cardinality',
        'treatment_set'
      ).alias('a')
      .join(
        journey_set.select(
          'identifier',
          'cardinality',
          'treatment_set'
        ).alias('b'),
        # This will lead to some incorrect mapping, but we'll fix that below
        # through filtering
        on=(
          (sf.col('b.cardinality') < sf.col('a.cardinality')) |
          (sf.col('a.identifier') == sf.col('b.identifier'))
        ),
        how='left'
      )
      # Compute shared journey set
      .withColumn(
        '_common_treatment',
        sf.array_intersect(
          sf.col('a.treatment_set'),
          sf.col('b.treatment_set')
        )
      )
      # How many shared touched points?
      .withColumn(
        '_common_cardinality',
        sf.size(sf.col('_common_treatment'))
      )
      # Focus only on those with a shared journey subset
      .filter(
        sf.col('_common_cardinality') == sf.col('b.cardinality')
      )
      .select(
        sf.col('a.identifier').alias('journey_id'),
        sf.col('b.identifier').alias('subset_id')
      )
      # XXX Fix hard-coded value
      .repartition(
        20,
        'journey_id'
      )
    )

    return jj_map

  @property
  def _append_journeys_exp(self):
    """
    Expression used to append journey information.
    """

    treatments = self.audience.treatments

    for i, t in enumerate(treatments):
      
      _c = (sf.col(f'a.{t}') == sf.col(f'b.{t}'))

      if i == 0:
        append_expression = _c
      else:
        append_expression = append_expression & (_c)
    
    return append_expression

  def append_journey_id(
    self,
    data: DataFrame,
    *largs,
    **kwargs
  ) -> DataFrame:
    """
    Append journeys to external data set.

    External data set must have appropriately labeled 
    """

    journey_sets = self._build_journey_set()

    data_enriched = (
      data.alias('a')
      .join(
        journey_sets.alias('b'),
        on=self._append_journeys_exp,
        how='left'
      )
      .select(
        'a.*',
        sf.col('b.identifier').alias('journey_id')
      )
    )

    return data_enriched


class AudiencetoValuator(BaseBuilder):
  """
  """

  def build(
    self,
    audience: Audience,
    valuator_type: str,
    *largs,
    **kwargs
  ):
    """
    Build valuator from an audience
    """

    if valuator_type == 'first_touch':

      valuator = FirstTouchValuator(audience, *largs, **kwargs)
    
    elif valuator_type == 'last_touch':

      valuator = LastTouchValuator(audience, *largs, **kwargs)
    
    elif valuator_type == 'shapley':

      # Takes a little extra work to get the Shapley valuator
      valuator = AudiencetoShapley(audience).build()

    else:

      raise NotImplemented

    return valuator
    

class ValuatorBuilder(BaseBuilder):
  """
  Build valuators
  """

  def build(self, data, *largs, **kwargs):
    """
    Build based on type
    """

    # Build audience
    audience = build_audience(data)

    # Hard-coded for now
    builder_key = ('Audience', )

    # Retrieve correct builder
    builder = self.get_handler(builder_key)

    valuator = builder.build(audience, *largs, **kwargs)

    return valuator

valuator_builder = ValuatorBuilder()
valuator_builder.register(('Audience', ), AudiencetoValuator())

def build_valuator(data, valuator_type, *largs, **kwargs):

  return valuator_builder.build(data, valuator_type, *largs, **kwargs)

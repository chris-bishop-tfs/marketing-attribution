"""
Audience is comprised of one or more respondents.
"""
import abc
from attr import attrs, attrib
from pyspark.sql import SparkSession
import pyspark.sql.functions as sf
from pyspark.sql.window import Window

from .utils import union_all
from .builder import BaseBuilder
from .journey import build_journey


@attrs
class Member(abc.ABC):
  """
  Respondees include an identifier and a journey
  """

  identifier = attrib()

  @abc.abstractmethod
  def to_pandas(self):
    # Export individual members to pandas
    pass

  @abc.abstractmethod
  def to_pyspark(self):
    # Dump individual member to spark
    pass


@attrs
class Respondent(Member):

  journey = attrib()

  def to_pandas(self, *largs, **kwargs):
    
    # Dump Journey
    data = self.journey.to_pandas(*largs, **kwargs)

    _cols = data.columns

    # Add respondent id
    data = data.assign(identifier=self.identifier)

    # Reshuffle columns so ID is in first column
    data = data[
      [
        'identifier',
        *_cols
      ]
    ]

    return data

  def to_pyspark(self, spark=None, *largs, **kwargs):

    if spark is None:
      # Spark session to use in computations
      spark = SparkSession.builder.getOrCreate()

    # We'll cheat to start and convert to a pandas intermediate
    data = spark.createDataFrame(
      self.to_pandas(*largs, **kwargs)
    )

    return data
    

@attrs
class Cohort(abc.ABC):
  """
  We'll use different kinds of "cohorts" (collections),
  so create a base class and build on top of it.
  """

  # Tuple of members
  members = attrib()

  @abc.abstractmethod
  def to_pandas(self):
    # Export to pandas
    raise NotImplemented

  @abc.abstractmethod
  def to_pyspark(self, spark=None):
    # Export data to spark
    raise NotImplemented
  

class Audience(Cohort):
  """
  Cohort of respondents
  """

  def _get_treatments(self):
    """
    Extract treatment information from audience
    """

    treatments = set()

    for m in self.members:
      treatments = treatments | set(m.journey.impressions)
    
    treatments = tuple(treatments)

    return treatments

  @property
  def treatments(self):

    return self._get_treatments()

  @property
  def cardinality(self):
    # Treatment cardinality
    return len(self.treatments)

  def to_pandas(self, *largs, **kwargs):

    pass

  def to_pyspark(self, format: str='wide', *largs, **kwargs):
    """
    Stitch journeys into a single pyspark dataframe
    """

    # Convert all members to pyspark
    #  Note that we'll initially pull this back as a 
    #  long data frame by default. This ensures dataframe
    #  compatibility.
    audience = [
      m.to_pyspark(*largs, **kwargs) for m in self.members
    ]

    audience_all = union_all(*audience)

    if format == 'wide':

      audience_all = (
        audience_all
        # We also want to maintain the treatment SET
        # This is useful for set operations down the line
        .withColumn(
          'treatment_set',
          # XXX Should this be collect_set?
          sf.collect_list(sf.col('treatment')).over(
            Window()
            .partitionBy(
              'identifier'
            )
            .orderBy(
              'order'
            )
            .rowsBetween(
              Window.unboundedPreceding,
              Window.unboundedFollowing
            )
          )
        )
        .groupBy(
          'identifier',
          'treatment_set',
          'value'
        )
        .pivot(
          'treatment'
        )
        # Arbitrary aggregation, can swap out with
        # ... almost anything?
        .min('order')
        .orderBy(
          'identifier'
        )
      )

    return audience_all


class DicttoAudience(object):
  """
  Build an audience from a dictionary

  Keys are IDs, values are impressions
  """
  
  def build(self, data, *largs, **kwargs):

    # Gather respondents
    members = [
      Respondent(
        identifier=k,
        # Need to build a journey from impression list
        # Pass through things like 'uniform', value, etc.
        journey=build_journey(
          v,
          *largs,
          **kwargs
        )
      ) for k, v in data.items()
    ]

    audience = Audience(members)

    return audience


class JSONtoAudience(object):
  """
  """

  def build(self, data, *largs, **kwargs):
    """
    Each record requires:
      - value:
      - identifier: respondent ID
      - impressions: treatment order
    """

    # Every record
    members = [
      Respondent(
        identifier=r['identifier'],
        # Need to build a journey from impression list
        # Pass through things like 'uniform', value, etc.
        journey=build_journey(
          impressions=r['impressions'],
          value=r['value'],
          *largs,
          **kwargs
        )
      ) for r in data
    ]

    audience = Audience(members)

    return audience


class AudienceBuilder(BaseBuilder):
  """
  Base audience builder
  """

  def build(self, data, *largs, **kwargs):
    """
    We'll use some sort of useful data format here with
    which we'll build out an audience object
    """

    # We'll choose the correct builder based on the data type
    data_type = type(data).__name__
    builder_key = (data_type, )

    # Retrieve correct builder
    builder = self.get_handler(builder_key)

    audience = builder.build(data, *largs, **kwargs)

    return audience

audience_builder = AudienceBuilder()
audience_builder.register(('dict', ), DicttoAudience())
audience_builder.register(('list', ), JSONtoAudience())


def build_audience(data, *largs, **kwargs):
  """
  Highest-level build function
  """

  # There are cases in which build_audience might be called
  # with an Audience object. In this case, just pass the object
  # back.

  if isinstance(data, Audience):
    return data
  else:
    return audience_builder.build(data, *largs, **kwargs)
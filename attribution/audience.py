"""
Audience is comprised of one or more respondents.
"""
import abc
from attr import define, field, attrs, attrib
from pyspark.sql import SparkSession, Row
import pyspark.sql.functions as sf
from pyspark.sql.window import Window

from .utils import union_all
from .builder import BaseBuilder
from .journey import build_journey


@define
class Member(abc.ABC):
  """
  Base class used to create respondents and the like

  Args:
    identifier (str): member identifier
  """

  identifier: str

  @abc.abstractmethod
  def to_pandas(self):
    """
    Convert member to a pandas data frame
    """

    # Export individual members to pandas
    pass

  @abc.abstractmethod
  def to_pyspark(self):
    """
    Convert to pyspark dataframe
    """
    # Dump individual member to spark
    pass


@define
class Respondent(Member):
  """
  Respondents comprised of an identifier and a joureny.

  Journeys are perhaps best understood in the context of
  a marketing nurturing campaign. A journey is synonymous
  with the order of treament.

  Args:
    identifier (str): respondent ID
    journey (tuple): ordered impressions
  """

  journey: tuple

  def to_dict(self, *largs, **kwargs):
    """
    Return respondent as a dictionary
    """

    return dict(
      identifier=self.identifier,
      impressions=self.journey.impressions,
      value=self.journey.value
    )

  def to_row(self, *largs, **kwargs):
    """
    Convert to a row
    """
    return Row(
      **self.to_dict()
    )

  def to_pandas(self, *largs, **kwargs):
    """
    Convert individual respondent to a Pandas Dataframe
    """

    # Dump Journey
    data = self.journey.to_pandas(*largs, **kwargs)

    # We'll use this below to order the output
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
    """
    Convert journey to a pyspark dataframe.

    At time of writing, Bishop hacked this a bit and
    used a pandas DF intermediate. That's not a great
    approach moving forward, but will work to start.

    Args:
      spark (SparkSession): spark session
    """

    if spark is None:
      # Spark session to use in computations
      spark = SparkSession.builder.getOrCreate()

    # We'll cheat to start and convert to a pandas intermediate
    data = spark.createDataFrame(
      self.to_pandas(*largs, **kwargs)
    )

    return data


@define
class Cohort(abc.ABC):
  """
  Bishop needed a way to organize a collection of members.
  A "cohort" is the start of such collections.

  Classes inheriting Cohort will need to define a to_pandas
  and to_pyspark method.

  Args:
    members (tuple): tuple of Member class types
  """

  # A tuple of members
  members: tuple

  @abc.abstractmethod
  def to_pandas(self):
    # Export to pandas
    raise NotImplemented

  @abc.abstractmethod
  def to_pyspark(self, spark=None):
    # Export data to spark
    raise NotImplemented


@define
class Audience(Cohort):
  """
  An audience is a cohort of "treated" respondents.
  """

  def _get_treatments(self) -> tuple:
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

  def to_pyspark(
    self,
    format: str='wide',
    spark=None,
    *largs,
    **kwargs
  ):
    """
    Stitch journeys into a single pyspark dataframe

    Args:
      format (str): 'wide'|'long'.
    """

    # Convert all members to pyspark
    #  Note that we'll initially pull this back as a 
    #  long data frame by default. This ensures dataframe
    #  compatibility.
    # audience = [
    #   m.to_pyspark(*largs, **kwargs) for m in self.members
    # ]

    if spark is None:
      # Spark session to use in computations
      spark = SparkSession.builder.getOrCreate()

    row_data = []

    for m in self.members:
      row_data.append(m.to_row())

    # Explode into proper format
    audience_all = (
      spark
      .createDataFrame(
        row_data
      )
      .withColumnRenamed(
        'impressions',
        'treatment'
      )
      .select(
        'identifier',
        sf.posexplode('treatment'),
        'value'
      )
      .withColumnRenamed(
        'pos',
        'order'
      )
      .withColumnRenamed(
        'col',
        'treatment'
      )
    )

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

    # Repartition based on identifier
    audience_all = (
      audience_all
      .repartition(
        20,
        'identifier'
      )
    )

    return audience_all


class DicttoAudience(object):
  """
  Build an audience from a dictionary

  Keys are IDs, values are impressions
  """

  def build(
    self,
    data: dict,
    *largs,
    **kwargs
  ) -> Audience:

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
  Convert JSON to an audience.

  Each JSON entry contains:
    - identifier: respondent ID
    - impressions: tuple of impressions
    - value: value of those impressions
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
  We'll need to build audiences from various data types.
  This builder will identify the correct builder, based
  on the type of data provided, and return a properly
  instantiated Audience object.
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

# Instantiate an audience builder and add supported
# data formats/associated builders
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

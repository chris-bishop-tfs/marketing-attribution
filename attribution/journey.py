
"""
Each Respondent (Member) is exposed to a series of one
or more impressions. Together, the Respondent and their
associated impressions comprise a Journey.

Journeys will form one of the base objects used in
subsequent valuation.
"""
import abc
from itertools import combinations
from numbers import Number
from attr import define
import pandas as pd
from .builder import BaseBuilder


@define
class Journey(abc.ABC):
	"""
	Individual journey, including multiple touch points
	"""

	# Treatment order
	#  Tuples cause some issues with pyspark so stick with lists for now
	impressions: list

	# Total value of journey (e.g., sales made, conversions, etc.)
	value: Number

	@property
	def n_impressions(self):
		"""
		Total number of impressions served
		"""

		return len(self.impressions)

	@property
	def treatments(self):

		return list(set(self.impressions))

	@property
	def cardinality(self):

		return len(self.treatments)

	def to_dict(self):
		"""
		"""

		return dict(
			impressions=self.impressions,
			value=self.value
		)

	def to_pandas(self, format: str='long'):
		"""
		Export journey to a LameDas dataframe
		"""

		if format == 'long':
			data = pd.DataFrame(
				dict(
					treatment=self.impressions,
					order=range(self.n_impressions),
					value=self.value
				)
			)

		# Wide exports here cause problems down the road
		# since we don't know hich treatments are potentially
		# missing. Discourage using this directly.
		elif format == 'wide':
			
			# Start with the long format and massage into wide
			# format.
			#
			# Bishop hates this in so many different colors.
			data_long = self.to_pandas(format='long')

			# Assign a dummy column that we'll use as an index below
			data_long['_dummy'] = 0

			data = (
				data_long
				.pivot(
					index='_dummy',
					columns='treatment',
					values='order'
				)
				.reset_index(
					drop=True
				)
				.rename_axis(
					None,
					axis=1
				)
				# Assign journey_value
				.assign(
					journey_value=data_long.journey_value[0]
				)
			)

		else:

			raise NotImplemented

		return data


class JourneyBuilder(BaseBuilder):
	"""
	How we build journeys
	"""

	def __init__(self, *largs, **kwargs):
		super(BaseBuilder, self).__init__()

		# XXX handlers not initializing correctly in BaseBuilder
		# Duplicating here to clear error, but needs attention
		self.handlers = dict()

	def build(
		self,
		# Tuples cause issues with pyspark down the line,
		# so create list here
		impressions: list,
		value: int=0
	) -> Journey:
		"""
		Last Touch will be the default journey type assigned
		"""

		# This is hard-coded for now since we only have 1 kind
		# of journey. BUT, used builder pattern in case we extend this
		# later.
		journey_type = ('journey', )
		journey_class = self.get_handler(journey_type)

		journey = journey_class(impressions, value)

		return journey

# All attribution methods leverage the same base class (for now)
# This might change in the future, so we'll use a register to
# facilitate future changes
journey_builder = JourneyBuilder()
journey_builder.register(('journey', ), Journey)

def build_journey(
	impressions: list,
	value: Number,
	*largs,
	**kwargs
) -> Journey:
	"""
	Simplest journey build possible is just a list of impressions
	"""

	return journey_builder.build(
		impressions,
		value,
		*largs,
		**kwargs
	)

def build_journey_set(
	treatments: (list, tuple),
	value: Number=1,
	*largs,
	**kwargs
) -> list:
	"""
	We needed a way to create combinations of treatments.
	That's what we do here.

	Provide a list of possible treatments, this returns all
	possible journeys (ignoring order)

	Args:
		treatments (tuple): possible treatments
		value (Number): numeric value of treatments
			Note: at time of writing, this is mostly a placeholder
	
	Returns:
		journeys (list): a list of possible journeys (ignoring order)
	"""

	journeys = []

	n_treatments = len(treatments)

	# Build all possible arrival orders up to a certain size (max_)
	for i in range(1, n_treatments + 1):
		# Build a journey from all set combinations
		journeys.extend(
			map(
				lambda x: build_journey(list(x), value, *largs, **kwargs),
				combinations(treatments, i)
			)
		)

	return list(journeys)
	
## Overview

This is the first installment of an attribution package written and maintained by data scientists at Thermo Fisher Scientific. Questions should be directed to the primary authors: [Chris Bishop](chris.bishop@thermofisher.com) and [Myra Haider](myra.haider@thermofisher.com).

Perform (marketing) attribution using an extensible framework. While its primary use case is marketing attribution, it can be used to allocate arbitrary value metrics over an arbitrary set of "players".

Attribution methods include:

- Last Touch
- First Touch
- Shapley Marginal Contribution

## Installation

```
pip install git+https://github.com/chris-bishop-tfs/marketing-attribution.git
```

## Terminology

Terminology here loosely follows marketing conventions.

- `Treatment`: A single manipulation, such as a phone call or a digital touch point

- `Impression`: A treatment applied to a respondent

- `Journey`: A sequence of impressions served to a respondent

- `Respondent` A treated element (e.g., customer) and its associated impressions

- `Audience`: A collection of respondents

- `Valuator`: A method through which value is attributed to one or more `treatments`

## Creating an Demonstrative Data Set

Execute the `data\create_example_data.ipynb` notebook to create a demonstrative data set.

## Example Usage:

Please see `examples\valuator_examples.ipynb` for a working example of the attribution package.

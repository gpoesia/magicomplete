ZIPFILE=pragmatic-code-autocomplete.zip
zip -r $ZIPFILE \
  *.py \
  */*.py \
  experiments/accuracy-small.json \
  experiments/ambiguity-small.json \
  experiments/fine-tuning-small.json \
  experiments/user-study.json \
  results/user-study-data.json.2020-08-31T23-31-35.578975 \
  README \
  figures/README \
  ./lines-small-py.json \
  ./repos-python-small.json.gz \
  models/c19ea.model \
  models/8f854.model

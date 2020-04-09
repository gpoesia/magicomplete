ZIPFILE=conadcomplete.zip
rm -f $ZIPFILE
zip -r $ZIPFILE \
  *.py \
  interactions.json \
  dataset/large.json \
  models/__init__.py \
  figures/README \
  results/README \
  "models/UniformEncoder(0.70)_0.0005_pythonalphabet.model" \
  "models/UniformEncoder(0.70)_0.0005_pythondecoder.model" \
  "models/UniformEncoder(0.70)_0.0005_python.json"

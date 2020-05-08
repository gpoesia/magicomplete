ZIPFILE=conadcomplete.zip
# rm -f $ZIPFILE
zip -r $ZIPFILE \
  *.py \
  models/__init__.py \
  pack.sh \
  *.ipynb \
  figures/README \
  results/README

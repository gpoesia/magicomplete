ZIPFILE=conadcomplete.zip
rm -f $ZIPFILE
zip -r $ZIPFILE \
  *.py \
  models/__init__.py \
  figures/README \
  results/README

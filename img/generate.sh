#note:  requires imagemagick

pdfdir='../tex'
find $pdfdir -type f -name '*.pdf' -print0 |
  while IFS= read -r -d '' file 
  do 
    filename=`basename $file`
    convert -verbose -density 150 -quality 100 -alpha remove "${file}" "${filename%.*}.jpg"
  done
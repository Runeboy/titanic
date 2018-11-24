pdfdir='../tex'
find $pdfdir -type f -name '*.pdf' -print0 |
  while IFS= read -r -d '' file
    filename=`basename $file`
    do convert -verbose -density 500 -resize 700 "${file}" "${filename%.*}.png"
  done
import csv, json, sys, pandas as pd
#if you are not using utf-8 files, remove the next line
 #set the encode to utf8
#check if you pass the input file and output file
if sys.argv[1] is not None and sys.argv[2] is not None:

    fileInput = sys.argv[1]
    fileOutput = sys.argv[2]

    d= pd.read_json(fileInput)

    d.to_csv(fileOutput,encoding='utf-8')

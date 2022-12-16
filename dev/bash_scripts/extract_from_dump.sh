foreach fn (` cat $1 `) 

echo ${fn}
python extract_from_lexicon3.py ${fn} ${fn}.output F

end




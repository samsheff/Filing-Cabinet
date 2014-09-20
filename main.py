import nltk
import fileinput

for line in fileinput.input("./input/keywords.txt"):
	text = nltk.word_tokenize(line)

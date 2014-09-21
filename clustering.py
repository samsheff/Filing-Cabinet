import sys

import glob

import numpy
from nltk.cluster import KMeansClusterer, GAAClusterer, euclidean_distance
import nltk.corpus
from nltk import decorators
import nltk.stem

import codecs
 
stemmer_func = nltk.stem.snowball.EnglishStemmer().stem
stopwords = set(nltk.corpus.stopwords.words('english'))
 
@decorators.memoize
def normalize_word(word):
    return stemmer_func(word.lower())
 
def get_words(titles):
    words = set()
    for title in job_titles:
        for word in title.split():
            words.add(normalize_word(word))
    return list(words)
 
@decorators.memoize
def vectorspaced(title):
    title_components = [normalize_word(word) for word in title.split()]
    
    return numpy.array([
        word in title_components and not word in stopwords
        for word in words], numpy.short)
 
if __name__ == '__main__':
 
    if len(sys.argv) == 2:
        filename = sys.argv[1]
 
    with open(filename) as title_file:
 
        print "Reading Files"
        job_titles = [unicode(line.strip(), "utf-8") for line in title_file.readlines()]
 
        print "Parsing Words"
        words = get_words(job_titles)
 
        print "Creating Cluster Instance"
        cluster = KMeansClusterer(10, euclidean_distance, 5)
        #cluster = GAAClusterer(20)
        
        print "Clustering"
        cluster.cluster([vectorspaced(title) for title in job_titles if title])
 
        # NOTE: This is inefficient, cluster.classify should really just be
        # called when you are classifying previously unseen examples!
        print "Classifying"
        classified_examples = [
                cluster.classify(vectorspaced(title)) for title in job_titles
           ]
        print "Saving results"
        for cluster_id, title in sorted(zip(classified_examples, job_titles)):
            filename = "results/"+ str(cluster_id) + ".txt"
            list = codecs.open(filename, "a", "utf-8")
            list.write(title + "\n")

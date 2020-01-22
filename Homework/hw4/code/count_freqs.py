import sys
from collections import defaultdict
import math

"""
Count n-gram frequencies in a CoNLL NER data file and write counts to
stdout. 
"""

def simple_conll_corpus_iterator(corpus_file):
    """
    Get an iterator object over the corpus file. The elements of the
    iterator contain (word, ne_tag) tuples. Blank lines, indicating
    sentence boundaries return (None, None).
    """
    l = corpus_file.readline()
    while l:
        line = l.strip()
        if line: # Nonempty line
            # Extract information from line.
            # Each line has the format
            # word pos_tag phrase_tag ne_tag
            fields = line.split(" ")
            ne_tag = fields[-1]
            #phrase_tag = fields[-2] #Unused
            #pos_tag = fields[-3] #Unused
            word = " ".join(fields[:-1])
            yield (word, ne_tag)
        else: # Empty line
            yield (None, None)                        
        l = corpus_file.readline()
        
def simple_conll_test_corpus_iterator(corpus_file):
    """
    Get an iterator object over the test corpus file. The elements of the
    iterator contain words. Blank lines, indicating sentence boundaries
    return None.
    """
    l = corpus_file.readline()
    while l:
        line = l.strip()
        if line:
            yield line
        else:
            yield None
        l = corpus_file.readline()

def sentence_iterator(corpus_iterator):
    """
    Return an iterator object that yields one sentence at a time.
    Sentences are represented as lists of (word, ne_tag) tuples.
    """
    current_sentence = [] #Buffer for the current sentence
    for l in corpus_iterator:        
            if l==(None, None):
                if current_sentence:  #Reached the end of a sentence
                    yield current_sentence
                    current_sentence = [] #Reset buffer
                else: # Got empty input stream
                    sys.stderr.write("WARNING: Got empty input file/stream.\n")
                    raise StopIteration
            else:
                current_sentence.append(l) #Add token to the buffer

    if current_sentence: # If the last line was blank, we're done
        yield current_sentence  #Otherwise when there is no more token
                                # in the stream return the last sentence.

def test_sentence_iterator(corpus_iterator):
    """
    Return an iterator object that yields one sentence at a time.
    Sentences are represented as lists of words.
    """
    current_sentence = []
    for l in corpus_iterator:
        if l == None:
            if current_sentence:
                yield current_sentence
                current_sentence = []
            else:
                sys.stderr.write("WARNING: Got empty input file/stream.\n")
                raise StopIteration
        else:
            current_sentence.append(l)
        
    if current_sentence:
        yield current_sentence
                                
def get_ngrams(sent_iterator, n):
    """
    Get a generator that returns n-grams over the entire corpus,
    respecting sentence boundaries and inserting boundary tokens.
    Sent_iterator is a generator object whose elements are lists
    of tokens.
    """
    for sent in sent_iterator:
         #Add boundary symbols to the sentence
         w_boundary = (n-1) * [(None, "*")]
         w_boundary.extend(sent)
         w_boundary.append((None, "STOP"))
         #Then extract n-grams
         ngrams = (tuple(w_boundary[i:i+n]) for i in xrange(len(w_boundary)-n+1))
         for n_gram in ngrams: #Return one n-gram at a time
            yield n_gram        


class Hmm(object):
    """
    Stores counts for n-grams and emissions. 
    """

    def __init__(self, n=3):
        assert n>=2, "Expecting n>=2."
        self.n = n
        self.emission_counts = defaultdict(int)
        self.ngram_counts = [defaultdict(int) for i in xrange(self.n)]
        self.all_states = set()
        # Parameters for replacing low frequency words
        self.word_counts = defaultdict(int)
        self.low_freq = 5
        self.rare_symbol = "_RARE_"

    def train(self, corpus_file):
        """
        Count n-gram frequencies and emission probabilities from a corpus file.
        P1: + count word frequencies
        """
        ngram_iterator = \
            get_ngrams(sentence_iterator(simple_conll_corpus_iterator(corpus_file)), self.n)

        for ngram in ngram_iterator:
            #Sanity check: n-gram we get from the corpus stream needs to have the right length
            assert len(ngram) == self.n, "ngram in stream is %i, expected %i" % (len(ngram, self.n))

            tagsonly = tuple([ne_tag for word, ne_tag in ngram]) #retrieve only the tags            
            for i in xrange(2, self.n+1): #Count NE-tag 2-grams..n-grams
                self.ngram_counts[i-1][tagsonly[-i:]] += 1
            
            if ngram[-1][0] is not None: # If this is not the last word in a sentence
                self.ngram_counts[0][tagsonly[-1:]] += 1 # count 1-gram
                self.word_counts[ngram[-1][0]] += 1 # and word
                self.emission_counts[ngram[-1]] += 1 # and emission frequencies
                self.all_states.add(ngram[-1][1]) # track all possible tags

            # Need to count a single n-1-gram of sentence start symbols per sentence
            if ngram[-2][0] is None: # this is the first n-gram in a sentence
                self.ngram_counts[self.n - 2][tuple((self.n - 1) * ["*"])] += 1

    def write_counts(self, output, printngrams=[1,2,3]):
        """
        Writes counts to the output file object.
        Format:

        """
        # Initialize
        for ne_tag in self.all_states:
            self.emission_counts[(self.rare_symbol, ne_tag)] = 0
            
        # Find words with low frequencies
        low_freq_words = set()
        low_freq_words_count_sum = 0
        for word in self.word_counts:
            if self.word_counts[word] < self.low_freq:
                low_freq_words.add(word)
                low_freq_words_count_sum += self.word_counts[word]
 
        # Replace words and emissions with low frequencies
        self.word_counts[self.rare_symbol] = low_freq_words_count_sum
        for word in low_freq_words:
            del self.word_counts[word]
            for ne_tag in self.all_states:
                if (word, ne_tag) in self.emission_counts:
                    self.emission_counts[(self.rare_symbol, ne_tag)] += self.emission_counts[(word, ne_tag)]
                    del self.emission_counts[(word, ne_tag)]
     
        # First write counts for emissions
        for word, ne_tag in self.emission_counts:   
            output.write("%i WORDTAG %s %s\n" % (self.emission_counts[(word, ne_tag)], ne_tag, word))

        # Then write counts for all ngrams
        for n in printngrams:            
            for ngram in self.ngram_counts[n-1]:
                ngramstr = " ".join(ngram)
                output.write("%i %i-GRAM %s\n" %(self.ngram_counts[n-1][ngram], n, ngramstr))
                
        # And write counts for all words
        for word in self.word_counts:
            output.write("%i WORDCOUNT %s\n" % (self.word_counts[word], word))
        
    def read_counts(self, corpusfile):

        self.n = 3
        self.emission_counts = defaultdict(int)
        self.ngram_counts = [defaultdict(int) for i in xrange(self.n)]
        self.all_states = set()

        for line in corpusfile:
            parts = line.strip().split(" ")
            count = float(parts[0])
            if parts[1] == "WORDTAG":
                ne_tag = parts[2]
                word = parts[3]
                self.emission_counts[(word, ne_tag)] = count
                self.all_states.add(ne_tag)
            elif parts[1].endswith("GRAM"):
                n = int(parts[1].replace("-GRAM",""))
                ngram = tuple(parts[2:])
                self.ngram_counts[n-1][ngram] = count
            else:
                word = parts[-1]
                self.word_counts[word] = count
                

    def emission_probability(self, word, tag):
        if (word, tag) in self.emission_counts:
            return self.emission_counts[(word, tag)] / self.ngram_counts[0][(tag,)]
        elif word in self.word_counts:
            return 0
        elif tag in self.all_states:
            return self.emission_counts[(self.rare_symbol, tag)] / self.ngram_counts[0][(tag,)]
        else:
            return -1
            
    def ml_probability(self, tag, bigram):
        trigram = bigram + (tag, )
        if bigram in self.ngram_counts[1] and trigram in self.ngram_counts[2]:
            return self.ngram_counts[2][trigram] / self.ngram_counts[1][bigram]
        elif tag in self.all_states.union(["STOP"]) and bigram[0] in self.all_states.union(["*"]) and bigram[1] in self.all_states.union(["*"]):
            return 0
        else:
            return -1
            
def usage():
    print """
    python count_freqs.py [input_file] > [output_file]
        Read in a named entity tagged training input file and produce counts.
    """

if __name__ == "__main__":

    if len(sys.argv)!=2: # Expect exactly one argument: the training data file
        usage()
        sys.exit(2)

    try:
        input = file(sys.argv[1],"r")
    except IOError:
        sys.stderr.write("ERROR: Cannot read inputfile %s.\n" % arg)
        sys.exit(1)
    
    # Initialize a trigram counter
    counter = Hmm(3)
    # Collect counts
    counter.train(input)
    # Write the counts
    counter.write_counts(sys.stdout)
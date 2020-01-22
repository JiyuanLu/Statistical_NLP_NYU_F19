import sys
from collections import defaultdict
import math

import count_freqs
from count_freqs import Hmm

class SimpleTagger(object):
    def __init__(self, hmm):
        self.hmm = hmm
    
    def tag(self, test_file, output_file):
        for line in test_file:
            word = line.strip()
            if word == "":
                output_file.write("\n")
                continue
            max_prob = 0
            max_tag = None
            for tag in self.hmm.all_states:
                emission_prob = self.hmm.emission_probability(word, tag)
                if emission_prob > max_prob:
                    max_prob = emission_prob
                    max_tag = tag
            if max_prob > 0:
                max_prob = math.log(max_prob, 2)
            output_file.write("%s %s %f\n" % (word, max_tag, max_prob))
    
def usage():
    print """
    python simple_tagger.py [counts_file] [test_file] > [output_file]
    """
    
if __name__ == "__main__":
    if len(sys.argv) != 3:
        usage()
        sys.exit(2)
        
    try:
        counts_file = file(sys.argv[1], "r")
        test_file = file(sys.argv[2], "r")
    except IOError:
        sys.stderr.write("ERROR: Cannot read inputfile %s.\n" % arg)
        sys.exit(1)
    
    # Initialize a trigram counter
    counter = Hmm(3)
    # Read counts
    counter.read_counts(counts_file)
    # Initialize a simple tagger
    tagger = SimpleTagger(counter)
    # Tag the data
    tagger.tag(test_file, sys.stdout)
    
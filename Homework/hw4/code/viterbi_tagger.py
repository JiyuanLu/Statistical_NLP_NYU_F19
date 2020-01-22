import sys
from collections import defaultdict
import math

import count_freqs
from count_freqs import Hmm, simple_conll_test_corpus_iterator, test_sentence_iterator

class ViterbiTagger(object):
    def __init__(self, hmm):
        self.hmm = hmm
        
    def tag(self, test_file, output_file):
        all_states = self.hmm.all_states
        sent_iterator = test_sentence_iterator(simple_conll_test_corpus_iterator(test_file))
        
        for sent in sent_iterator:
            probs = defaultdict(int)
            tags = defaultdict(str)
            
            probs[(-1, "*", "*")] = 1
            n = len(sent)
            
            # compute all probabilities
            for i in range(n):
                for v in all_states:
                    for u in all_states if i > 0 else ["*"]:
                        max_prob = 0
                        max_tag = None
                        for w in all_states if i > 1 else ["*"]:
                            prob = probs[(i-1, w, u)] * self.hmm.ml_probability(v, (w, u)) * self.hmm.emission_probability(sent[i], v)
                            if prob > max_prob:
                                max_prob = prob
                                max_tag = w
                        
                        probs[(i, u, v)] = max_prob
                        tags[(i, u, v)] = max_tag
              
            # find the maximum probability and the corresponding tag sequence
            max_sent_prob = 0
            max_tags = [None] * n
            for u in all_states if n > 1 else ["*"]:
                for v in all_states:
                    prob = probs[(n - 1, u, v)] * self.hmm.ml_probability("STOP", (u, v))
                    if prob > max_sent_prob:
                        max_sent_prob = prob
                        max_tags[n-2] = u
                        max_tags[n-1] = v
            
            # continue finding maximum tag sequence
            for i in range(n-3, -1, -1):
                max_tags[i] = tags[(i+2, max_tags[i+1], max_tags[i+2])]   
            max_tags.append("*")
            
            # write results
            for i in range(n):
                word = sent[i]
                tag = max_tags[i]
                prob = probs[(i, max_tags[i-1], max_tags[i])]
                log_prob = 0
                if prob > 0:
                    log_prob = math.log(probs[(i, max_tags[i-1], max_tags[i])], 2)
                output_file.write("%s %s %f\n" % (word, tag, log_prob))
            output_file.write("\n")
            
def usage():
    print """
    python viterbi_tagger.py [counts_file] [test_file] > [output_file]
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
    tagger = ViterbiTagger(counter)
    # Tag the data
    tagger.tag(test_file, sys.stdout)
        
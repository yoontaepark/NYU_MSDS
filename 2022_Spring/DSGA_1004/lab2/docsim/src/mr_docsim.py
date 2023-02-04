#! /usr/bin/env python

import re
import string
import pathlib

from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.compat import jobconf_from_env


WORD_RE = re.compile(r"[\S]+")


class MRDocSim(MRJob):
    """
    A class to count word frequency in an input file.
    """

    def mapper_get_words(self, _, line):
        """

        Parameters:
            -: None
                A value parsed from input and by default it is None because the input is just raw text.
                We do not need to use this parameter.
            line: str
                each single line a file with newline stripped

            Yields:
                (key, value) pairs
        """

        # This part extracts the name of the current document being processed
        current_file = jobconf_from_env("mapreduce.map.input.file")

        # Use this doc_name as the identifier of the document
        doc_name = pathlib.Path(current_file).stem

        for word in WORD_RE.findall(line):
            # strip any punctuation
            word = word.strip(string.punctuation)

            # enforce lowercase
            word = word.lower()

            # TODO: start implementing here!
            # creating ((doc_name, word), 1) pair
            yield (doc_name, word), 1

        # create a placeholder(i.e. punctuation @) for every doc with value = 0
        for _ in range(1):
            yield (doc_name, '@'), 0


    # creating word, (doc_name, sum(counts)) pair
    def reducer_1(self, word, counts):
        yield word[1], (word[0], sum(counts))

    # by word key, append (doc_name, sum(counts)) pairs
    def reducer_2(self, word, counts):
        yield word, list(counts)

    # iterate each appended list of (doc_name, value) pair
    # we need len(pair) * len(pair) combinations, so iterate twice
    # so, we now have (doc_i, doc_j, word), ((counts for doc_i and word), (counts for doc_j and word))
    def mapper_3(self, word, counts):
        for i in range(len(counts)):
            for j in range(len(counts)):
                yield (counts[i][0], counts[j][0], word), (counts[i][1], counts[j][1])

    # for given list of counts, we want min value, so min(counts) to get the value
    def reducer_3(self, word, counts):
        yield word, min(list(counts)[0])

    # we don't need word to be yielded, so only yield (doc_i, doc_j), counts
    def mapper_4(self, word, counts):
        yield (word[0], word[1]), counts

    # intermediate result: (doc_i, doc_j), sum(counts)
    def reducer_4(self, word, counts):
        yield word, sum(counts)            
    
    def steps(self):
        return [
            MRStep(mapper=self.mapper_get_words,
                   reducer=self.reducer_1),
            MRStep(reducer=self.reducer_2),
            MRStep(mapper=self.mapper_3,
                   reducer=self.reducer_3),
            MRStep(mapper=self.mapper_4,
                   reducer=self.reducer_4)
            ]


# this '__name__' == '__main__' clause is required: without it, `mrjob` will
# fail. The reason for this is because `mrjob` imports this exact same file
# several times to run the map-reduce job, and if we didn't have this
# if-clause, we'd be recursively requesting new map-reduce jobs.
if __name__ == "__main__":
    # this is how we call a Map-Reduce job in `mrjob`:
    MRDocSim.run()

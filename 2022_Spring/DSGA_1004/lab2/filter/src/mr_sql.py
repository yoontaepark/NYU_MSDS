#! /usr/bin/env python

from mrjob.job import MRJob

class MRMoviesByGenreCount(MRJob):
    """
    Find the distinct number of movies in each genre.
    """

    def mapper(self, _, line):
        for each_line in line.split('\n'):
            name, genre = each_line.split(',')
            
            if genre != 'Horror': 
                yield genre, name


    def reducer(self, key, values):        
        sum_names = len(set(values))

        if sum_names >= 100:
            yield key, sum_names
        

# don't forget the '__name__' == '__main__' clause!
if __name__ == '__main__':
    MRMoviesByGenreCount.run()

""" Part 1 of Homework 0. This is an ungraded assignment that helps you get comfortable with writing Python."""
import string
class TokenIndexer:

    def __init__(self, dictionary):
        self.dictionary = dictionary

    def token_to_index(self, token: str) -> int:
        """Map a word to the integer index defined in self.dictionary.
        
        Returns:
            The index assigned to token in self.dictionary. If token is not defined in self.dictionary, then
            return -1.
        """
        ### TODO: Your code here!
        # if given word is in dictionary, return its value
        # else, return -1 
        if token in self.dictionary: return self.dictionary.get(token, -1)
        else: return -1
 
        ### End of your code.

    # def tokens_to_indices(self, tokens: str) -> list[int]:
    def tokens_to_indices(self, tokens: str):
        """Take a string representing a sentence, tokenize it, and return a list of token indices.

        Hints:
            1.  You should first convert tokens to a list of lower-case words. This involves removing punctuation
                and lower-casing each word. You can use the built-in Python function string.lower().
            2. Next, you should use a list comprehension to apply self.token_to_index to each word.
        
        Returns:
            List of token indices, for each word. If a word is undefined, then return -1.
        """
        ### TODO: Your code here!
        # lowercase
        tokens = tokens.lower()

        # iterate and remove punctuation

        tokens = [token.replace(token, '') for token in tokens if token in string.punctuation]

        # for token in tokens:
        #     if token in string.punctuation:
        #         tokens = tokens.replace(token, '')
    
        # split strings by ' '
        token_list = tokens.split(' ')

        # create an empty list 
        ans_list = []

        # iterate by each word and conver to interger
        for token in token_list:
            ans_list.append(self.token_to_index(token))

        # return the answer list
        return ans_list
        
        ### End of your code.


if __name__ == "__main__":
    dictionary = {
        "hello": 0,
        "world": 1,
        "welcome": 2,
        "to": 3,
        "nlp": 4,
    }
    indexer = TokenIndexer(dictionary)
    
    print("Testing Part 1A...")
    assert indexer.token_to_index("hello") == 0
    assert indexer.token_to_index("goodbye") == -1
    print("  => Part 1A tests passed!")

    print("Testing Part1B...")
    assert indexer.tokens_to_indices("Hello world, welcome to NLP!") == [0, 1, 2, 3, 4]
    assert indexer.tokens_to_indices("Hello and welcome, student!") == [0, -1, 2, -1]
    print("  => Part 1B tests passed!")
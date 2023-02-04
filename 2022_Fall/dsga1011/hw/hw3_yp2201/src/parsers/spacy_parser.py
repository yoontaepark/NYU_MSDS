from src.parsers.parser import Parser
from src.dependency_parse import DependencyParse

# add lib
import spacy

class SpacyParser(Parser):

    def __init__(self, model_name: str):
        # TODO: Your code here!
        self.model = spacy.load(model_name)
    

    def parse(self, sentence: str, tokens: list) -> DependencyParse:
        """Use the specified spaCy model to parse the sentence.py.

        The function should return the parse in the Dependency format.

        You should refer to the spaCy documentation for how to use the model for dependency parsing.
        """
        # Your code here!
        # https://spacy.io/api/doc
        # https://stackoverflow.com/questions/48169545/does-spacy-take-as-input-a-list-of-tokens
                
        # creating a space variable 
        spaces = []
        for i, token in enumerate(tokens):
            if i == len(tokens)-1:
                spaces.append(False)
                break
            else:           
                if sentence[sentence.index(token[-1])+1] == ' ':
                    spaces.append(True)
                else:
                    spaces.append(False)
                    
                sentence = sentence[sentence.index(token[-1])+1:]                    
        
        # creating a spacy doc
        doc = spacy.tokens.Doc(self.model.vocab, words=tokens, spaces=spaces)
        
        # run the standard pipeline for given doc
        # now doc contains features of: tok2vec, tagger, parser, attribute_ruler, ner
        for _, proc in self.model.pipeline:
            doc = proc(doc)
        
        # creating a dependency parser based on the documentation: https://spacy.io/api/doc
        # text: use predefined sentence 
        # tokens: use predefined tokens
        # head: will be generated looking at .head function in spacy token 
        # deprel: using .dep_ function in spacy token. 
        # alos, lowercasing the result as ROOT is capitalized and also there are some possibilities that others can be capitalized 
        res_dict = {
            "text": sentence,
            "tokens": tokens, 
            "head": [],
            "deprel": [t.dep_.lower() for t in doc],
        }
        
        # generating heads
        for token in doc:
            # if token index and token head index is same, then this token is a root. (head = 0)
            # else, find index of token head, and add 1 as root is already assigned (as 0)
            if token.i == token.head.i:
                res_dict['head'].append(str(0))
            else: 
                res_dict['head'].append(str(token.head.i + 1))
        
        # return dependency parser
        return DependencyParse.from_huggingface_dict(res_dict)
""" Provides utility methods used in scripts."""

from src.cord19q.tokenizer import Tokenizer
from gensim.parsing.preprocessing import strip_tags, strip_punctuation, \
    strip_multiple_whitespaces, strip_numeric, \
    remove_stopwords, strip_short, strip_non_alphanum

CUSTOM_FILTERS = [
    lambda x: x.lower(), strip_tags, strip_punctuation,
    strip_multiple_whitespaces, strip_numeric, strip_non_alphanum,
    remove_stopwords, strip_short
]


class Validate:
    @staticmethod
    def query_validate(query, embeddings):
        if is_proper(query):
            try:
                # validate if all tokens are found
                tokens = Tokenizer.tokenize(query)
                embeddings.lookup(tokens)
            except Exception as e:
                return False, 'One or more words in your query not found in the vocabulary'
        else:
            return False, 'Query is not valid. Query must be non-zero length and contain fewer than 200 alphanumeric \
                                                    characters (with the exception of ? and -)'
        return True, None


# checks if a query is non zero length, < 100 chars and only contains alphanumeric characters
def is_proper(query):
    if len(query) == 0 or len(query) > 200: return False
    return all(x.isalnum() or x.isspace() or x is '-' or x is '?' for x in query)

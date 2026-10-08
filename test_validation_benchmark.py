import unittest
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from benchmark_text import tokenize, lexical_predict, LEXICAL_CUES

class TextBenchmarkTests(unittest.TestCase):
    def test_negation_numbers_contractions_and_punctuation_survive(self):
        tokens=tokenize("I can't pay 1,234.56; not refunded! 15 U.S.C. 1666 _")
        for token in ["can't",'not','1,234.56',';','!','15','1666','.','_']:
            self.assertIn(token,tokens)

    def test_validation_only_word_does_not_enter_vocabulary_or_idf(self):
        v=TfidfVectorizer(tokenizer=tokenize,token_pattern=None,ngram_range=(1,2))
        v.fit(['not charged interest','card was closed'])
        vocabulary=dict(v.vocabulary_); idf=v.idf_.copy()
        transformed=v.transform(['validationexclusive not interest'])
        self.assertNotIn('validationexclusive',v.vocabulary_)
        self.assertEqual(vocabulary,v.vocabulary_)
        self.assertTrue(np.array_equal(idf,v.idf_))
        self.assertGreater(transformed.nnz,0)

    def test_fixed_phrase_baseline_and_training_only_fallback(self):
        labels=list(LEXICAL_CUES)
        counts={label:1 for label in labels}
        counts['Fees or interest']=100
        pred,evidence=lexical_predict(['financial hardship','no matching cues xyz'],labels,counts)
        self.assertEqual(pred,['Struggling to pay your bill','Fees or interest'])
        self.assertFalse(evidence[0]['majority_fallback'])
        self.assertTrue(evidence[1]['majority_fallback'])

if __name__=='__main__': unittest.main()

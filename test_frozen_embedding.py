import unittest
import numpy as np
from frozen_embedding import partition_tokens,aggregate_chunks

class CompleteNarrativeTests(unittest.TestCase):
    def test_boundary_and_very_long_inputs_have_exactly_once_coverage(self):
        for length in [1,254,255,508,509,10000]:
            ids=list(range(length)); chunks=partition_tokens(ids,254)
            self.assertEqual([t for chunk in chunks for t in chunk],ids)
            self.assertTrue(all(len(chunk)<=254 for chunk in chunks))
            self.assertEqual(len(chunks),(length+253)//254)

    def test_aggregation_weights_short_tail_by_its_token_count(self):
        vec=np.array([[1,0],[0,1]],dtype=np.float32)
        result=aggregate_chunks(vec,[254,1])
        self.assertAlmostEqual(float(np.linalg.norm(result)),1,places=6)
        self.assertGreater(result[0],.99)
        self.assertLess(result[1],.01)

    def test_single_chunk_vector_is_preserved(self):
        v=np.array([[.6,.8]],dtype=np.float32)
        self.assertTrue(np.allclose(aggregate_chunks(v,[20]),v[0]))

if __name__=='__main__':unittest.main()

import unittest
import torch
from finetune_model import balanced_weights,weighted_loss_sum

class TrainingLossTests(unittest.TestCase):
    def test_weighted_training_mass_equal_across_classes(self):
        counts=torch.tensor([100,10,5])
        weights=balanced_weights(counts)
        self.assertTrue(torch.allclose(weights*counts,torch.full((3,),115/3)))
        self.assertAlmostEqual(float((weights*counts).sum()/counts.sum()),1,places=6)

    def test_accumulated_gradients_match_weighted_full_batch_mean(self):
        torch.manual_seed(4)
        logits=torch.randn(5,3,requires_grad=True)
        targets=torch.tensor([0,1,2,0,2]);weights=balanced_weights([100,10,5])
        (weighted_loss_sum(logits,targets,weights)/5).backward()
        expected=logits.grad.clone();logits.grad.zero_()
        (weighted_loss_sum(logits[:2],targets[:2],weights)/5).backward()
        (weighted_loss_sum(logits[2:],targets[2:],weights)/5).backward()
        self.assertTrue(torch.allclose(logits.grad,expected,atol=1e-7))

if __name__=='__main__':unittest.main()

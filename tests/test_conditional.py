# import pytest
# import torch
# from torchGMM import TimeDependentConditional

# class TestConditionalGMM:

#     @pytest.mark.parametrize("dim", [1, 3])
#     def test_conditional_log_prob(self, dim):
#         x0 = torch.randn(10, dim)
#         cond = TimeDependentConditional(x0)
#         x = torch.randn(11, dim)

#         log_prob = cond(x, t=0.0)
#         assert log_prob.shape == (10, 11)

#     @pytest.mark.parametrize("dim", [1, 3])
#     def test_conditional_score(self, dim):
#         x0 = torch.randn(10, dim)
#         cond = TimeDependentConditional(mu=x0)
#         x = torch.randn(11, dim)

#         score = cond_gmm.score(x, t=0.0)
#         assert score.shape == (10, 11, dim)

#     @pytest.mark.parametrize('t', [0.0, 0.1, 0.8, 1.0])
#     @pytest.mark.parametrize("dim", [1, 3])
#     def test_conditional_energy(self, dim, t):
#         x0 = torch.randn(10,dim)
#         cond = TimeDependentConditional(x0)
#         x = torch.randn(11, dim)
#         energy = cond_gmm.energy(x, t=t)

#     @pytest.mark.parametrize('t', [0.0, 0.1, 0.8, 1.0])
#     @pytest.mark.parametrize("dim", [1, 3])
#     def test_conditional_energy(self, dim, t):
#         x0 = torch.randn(10,dim)
#         cond_gmm = TimeDependentConditional(x0)
#         x = torch.randn(11, dim)
#         energy = cond_gmm.energy(x, t=t)
    
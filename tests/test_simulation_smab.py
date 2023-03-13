# # MIT License
# #
# # Copyright (c) 2022 Playtika Ltd.
# #
# # Permission is hereby granted, free of charge, to any person obtaining a copy
# # of this software and associated documentation files (the "Software"), to deal
# # in the Software without restriction, including without limitation the rights
# # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# # copies of the Software, and to permit persons to whom the Software is
# # furnished to do so, subject to the following conditions:
# #
# # The above copyright notice and this permission notice shall be included in all
# # copies or substantial portions of the Software.
# #
# # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# # SOFTWARE.

# import pandas as pd
# from pandas._testing import assert_frame_equal

# from pybandits.smab import Smab
# from pybandits.simulation_smab import SimulationSmab


# def test_init():
#     random_seed = 1
#     smab = Smab(action_ids=["action A", "action B", "action C"], random_seed=random_seed)
#     SimulationSmab(smab=smab)
#     SimulationSmab(
#         smab=smab,
#         n_updates=20,
#         batch_size=2000,
#         probs_reward={"action A": 0.6, "action B": 0.0, "action C": 1.0},
#         save=True,
#         path="folder/",
#         random_seed=1,
#         verbose=True,
#     )


# def test_run():
#     random_seed = 1
#     smab = Smab(action_ids=["action A", "action B", "action C"], random_seed=random_seed)
#     sim = SimulationSmab(smab=smab, n_updates=5, batch_size=6, random_seed=random_seed)
#     sim.run()

#     X = pd.DataFrame(
#         [
#             ["action B", 1.0],
#             ["action B", 0.0],
#             ["action A", 1.0],
#             ["action A", 0.0],
#             ["action C", 1.0],
#             ["action C", 0.0],
#             ["action C", 1.0],
#             ["action A", 1.0],
#             ["action A", 0.0],
#             ["action A", 0.0],
#             ["action B", 1.0],
#             ["action B", 0.0],
#             ["action C", 1.0],
#             ["action C", 0.0],
#             ["action B", 1.0],
#             ["action C", 0.0],
#             ["action B", 0.0],
#             ["action C", 1.0],
#             ["action C", 1.0],
#             ["action A", 1.0],
#             ["action C", 0.0],
#             ["action A", 0.0],
#             ["action B", 1.0],
#             ["action B", 0.0],
#             ["action C", 1.0],
#             ["action C", 0.0],
#             ["action B", 1.0],
#             ["action B", 0.0],
#             ["action C", 0.0],
#             ["action C", 1.0],
#         ],
#         columns=["action", "reward"],
#     )
#     assert_frame_equal(sim.results, X)


# def test_functions_get():
#     random_seed = 1
#     smab = Smab(action_ids=["action A", "action B", "action C"], random_seed=random_seed)
#     sim = SimulationSmab(smab=smab, n_updates=5, batch_size=6, random_seed=random_seed)
#     sim.run()

#     summary_action = {"action A": 7, "action B": 10, "action C": 13}
#     summary_reward = {"action A": 0.42857142857142855, "action B": 0.5, "action C": 0.5384615384615384}

#     _, _ = sim.get_cumulative_proportions()["action"], sim.get_cumulative_proportions()["reward"]

#     assert sim.get_count_selected_actions() == summary_action
#     assert sim.get_proportion_positive_reward() == summary_reward

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

# import numpy as np
# import pandas as pd
# from numpy.testing import assert_equal
# from pandas._testing import assert_frame_equal

# from pybandits.cmab import Cmab
# from pybandits.simulation_cmab import SimulationCmab


# def test_init():
#     """Test init SimulationCmab."""

#     random_seed = 1
#     batch_size = 100
#     n_groups = 3
#     n_updates = 10
#     n_jobs = 1
#     actions_ids = ['action A', 'action B', 'action C']
#     n_features = 5
#     X = np.random.rand(batch_size*n_updates, n_features)

#     group = np.random.randint(n_groups, size=batch_size*n_updates)
#     cmab = Cmab(n_features=n_features, actions_ids=actions_ids, n_jobs=n_jobs, random_seed=random_seed)
#     prob_rewards = pd.DataFrame([[0.05, 0.80, 0.05],
#                                  [0.80, 0.05, 0.05],
#                                  [0.80, 0.05, 0.80]], columns=actions_ids, index=range(n_groups))
#     verbose = True
#     save = True
#     path = 'tests/simulation/'

#     # init with default params
#     sim = SimulationCmab(cmab=cmab, X=X)

#     # init with custom params
#     X = pd.DataFrame(np.random.rand(batch_size * n_updates, n_features))
#     sim = SimulationCmab(cmab=cmab, X=X, group=group, batch_size=batch_size, n_updates=n_updates,
#                          prob_rewards=prob_rewards, save=save, path=path, random_seed=random_seed, verbose=verbose)

#     assert sim._X.shape == (sim._batch_size * sim._n_updates, sim._cmab._n_features), 'X shape mismatch'
#     assert sim.results['group'].shape == (sim._batch_size * sim._n_updates,), ' group shape mismatch'
#     assert sim.results['group'].isin(range(sim._n_groups)).all(), 'group array should contain only values in ' \
#                                                                   '' + str(range(n_groups))
#     assert sim._rewards.shape == (sim._batch_size * sim._n_updates, len(actions_ids)), 'reward shape mismatch'
#     assert sim.results.shape == (sim._batch_size * sim._n_updates, sim._cmab._n_features), 'result shape mismatch'

#     assert_frame_equal(sim._X, X)


# def test_run():
#     """ Test simulation with cmab model. """

#     random_seed = 2
#     batch_size = 10
#     features_ids = ['feat_1', 'feat_2']
#     n_groups = 2
#     n_updates = 2
#     actions_ids = ['action A', 'action B']
#     prob_rewards = pd.DataFrame([[0.05, 0.80],
#                                  [0.80, 0.05]], columns=actions_ids, index=range(n_groups))
#     cmab = Cmab(n_features=len(features_ids), actions_ids=actions_ids, random_seed=random_seed)
#     df = pd.DataFrame([[34.07772868659151, -28.948390811625714, 'action B', 0.0, 1, 0.05, 0.8, 0.75, 0.75],
#                       [47.602172988242444, -11.585294068594154, 'action A', 1.0, 1, 0.8, 0.8, 0.0, 0.75],
#                       [58.74075304505904, -99.71942656529076, 'action A', 0.0, 1, 0.8, 0.8, 0.0, 0.75],
#                       [41.462039348288144, -117.66517424958462, 'action A', 0.0, 0, 0.05, 0.8, 0.75, 1.5],
#                       [56.18746540687566, -122.02451865370041, 'action A', 0.0, 0, 0.05, 0.8, 0.75, 2.25],
#                       [29.982587761836534, -114.24870860989691, 'action B', 0.0, 1, 0.05, 0.8, 0.75, 3.0],
#                       [56.085236090749326, -1.974650230141235, 'action A', 1.0, 1, 0.8, 0.8, 0.0, 3.0],
#                       [32.42925525229372, -106.92841840939255, 'action A', 0.0, 0, 0.05, 0.8, 0.75, 3.75],
#                       [31.00949679739198, -42.8284308455658, 'action A', 0.0, 0, 0.05, 0.8, 0.75, 4.5],
#                       [35.47919944987376, -88.52700687134366, 'action B', 1.0, 0, 0.8, 0.8, 0.0, 4.5],
#                       [62.6929015285017, -148.44461692725085, 'action B', 1.0, 0, 0.8, 0.8, 0.0, 4.5],
#                       [39.95202041753999, 49.56228281374906, 'action B', 0.0, 1, 0.05, 0.8, 0.75, 5.25],
#                       [47.779661185138565, 23.932111189164278, 'action A', 1.0, 1, 0.8, 0.8, 0.0, 5.25],
#                       [31.077061872610724, 88.69384882684793, 'action A', 1.0, 1, 0.8, 0.8, 0.0, 5.25],
#                       [56.119706130978265, -138.1918694119457, 'action B', 1.0, 0, 0.8, 0.8, 0.0, 5.25],
#                       [37.9189897898034, 136.88209858829075, 'action A', 1.0, 1, 0.8, 0.8, 0.0, 5.25],
#                       [35.18935782070607, -83.10216873048782, 'action B', 0.0, 0, 0.8, 0.8, 0.0, 5.25],
#                       [2.11516522092686, -0.702259810984084, 'action A', 0.0, 0, 0.05, 0.8, 0.75, 6.0],
#                       [25.560807764772438, -2.8576525901465555, 'action A', 0.0, 0, 0.05, 0.8, 0.75, 6.75],
#                       [40.465956204463076, 144.48135008944516, 'action A', 1.0, 1, 0.8, 0.8, 0.0, 6.75]],
#                       columns=features_ids+['action', 'reward', 'group', 'selected_prob_reward', 'max_prob_reward',
#                                             'regret', 'cum_regret'])
#     X = df[features_ids]
#     group = df['group']
#     verbose = True
#     save = False
#     path = 'tests/simulation/'

#     # init simulation
#     sim = SimulationCmab(cmab=cmab, X=X, group=group, batch_size=batch_size, n_updates=n_updates,
#                          prob_rewards=prob_rewards, save=save, path=path, random_seed=random_seed, verbose=verbose)

#     # start simulation
#     sim.run()
#     assert_frame_equal(sim.results, df[['action', 'reward', 'group', 'selected_prob_reward', 'max_prob_reward',
#                                         'regret', 'cum_regret']])

#     # test functions get
#     d = {'group 0': {'action A': 6, 'action B': 4},
#          'group 1': {'action A': 7, 'action B': 3}}
#     assert_equal(sim.get_count_selected_actions(), d)

#     d = {'group 0': {'action A': np.nan, 'action B': 0.75},
#          'group 1': {'action A': 0.8571428571428571, 'action B': np.nan}}
#     assert_equal(sim.get_proportion_positive_reward(), d)

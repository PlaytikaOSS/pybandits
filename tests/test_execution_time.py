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

# import time

# import numpy as np

# from pybandits.core.cmab import Cmab

# verbose = True


# def run_cmab_predict_streaming(n_actions, n_features, n_samples, n_iterations, n_jobs, sampling, params_sample,
#                                verbose=False):
#     """
#     This function executes the following steps:
#         - initialize cmab with input params
#         - simulate first batch of users with actions and rewards
#         - update cmab with first batch
#         - predict 1 sample at time (i.e. in streaming) with sampling (sampling=True) or without (sampling=False)
#         - return the mean and std of the prediction time.
#     """

#     # params
#     size_first_batch = 1000
#     actions_ids = ['action' + str(i + 1) for i in range(n_actions)]

#     # init model
#     cmab = Cmab(n_features=n_features, actions_ids=actions_ids, n_jobs=n_jobs, params_sample=params_sample)

#     # simulate first batch
#     X = 2 * np.random.random_sample((size_first_batch, n_features)) - 1  # float in the interval (-1, 1)
#     actions, _ = cmab.predict(X)
#     rewards = np.random.randint(2, size=size_first_batch)

#     # update
#     start = time.time()
#     cmab.update(X=X, actions=actions, rewards=rewards)
#     end = time.time()
#     t = end - start
#     if verbose:
#         print('\nUpdate with n_actions = {}, n_features = {}, size_first_batch = {}. Time = {:.6f} sec.'
#               .format(n_actions, n_features, size_first_batch, t))

#     # predict 1 sample at time
#     t = []
#     for i in range(n_iterations):
#         x = 2 * np.random.random_sample((n_samples, n_features)) - 1  # floats in the interval (-1, 1)
#         if sampling:
#             start = time.time()
#             _, _ = cmab.predict(x)
#             end = time.time()
#         else:
#             start = time.time()
#             _, _ = cmab.fast_predict(x)
#             end = time.time()
#         t.append(end-start)
#     mu_t, simga_t = np.mean(t), np.std(t)

#     if verbose:
#         print('Predict of n_actions={}, n_features={}, n_samples={}, n_iterations={}, sampling={}. '
#               '\nmean execution time = {:.6f} sec, std execution time = {:.6f} sec '
#               .format(n_actions, n_features, n_samples, n_iterations, sampling, mu_t, simga_t))

#     return mu_t, simga_t


# def test_cmab_time_predict_before_update():
#     """ Test cmab.predict() in steaming before the first update(). """
#     # input
#     n_iteration = 10000
#     n_actions = 1000
#     n_samples = 1
#     n_features = 1000
#     actions_ids = ['action' + str(i + 1) for i in range(n_actions)]
#     params_sample = {'tune': 500, 'draws': 1000, 'chains': 2, 'init': 'adapt_diag', 'cores': 1, 'target_accept': 0.95,
#                      'progressbar': False}

#     # init model
#     cmab = Cmab(n_features=n_features, actions_ids=actions_ids, params_sample=params_sample)

#     # predict
#     t = []
#     for i in range(n_iteration):
#         x = 2 * np.random.random_sample((n_samples, n_features)) - 1  # float in the interval (-1, 1)
#         start = time.time()
#         _, _ = cmab.predict(x)
#         end = time.time()
#         t.append(end - start)
#     mu_t, simga_t = np.mean(t), np.std(t)

#     if verbose:
#         print('\nPredict before the first update of n_samples={}, n_actions={}, n_features={}, n_iteration={}'
#               '\nmean execution time = {:.6f} sec, std execution time = {:.6f} sec '
#               .format(n_samples, n_iteration, n_actions, n_features, mu_t, simga_t))


# # test with fast predict

# def test_cmab_time_predict_2_2_1_fp():
#     """ Test cmab.fast_predict() in steaming after the first update(). """
#     # input
#     n_actions = 2
#     n_features = 2
#     n_samples = 1
#     n_iterations = 10
#     n_jobs = n_actions
#     sampling = False
#     params_sample = {'tune': 500, 'draws': 1000, 'chains': 2, 'init': 'adapt_diag', 'cores': 1, 'target_accept': 0.95,
#                      'progressbar': False}

#     # run test
#     mu_t, simga_t = run_cmab_predict_streaming(n_actions=n_actions, n_features=n_features, n_samples=n_samples,
#                                                n_iterations=n_iterations, n_jobs=n_jobs, sampling=sampling,
#                                                params_sample=params_sample, verbose=verbose)


# def test_cmab_time_predict_2_2_10000_fp():
#     """ Test cmab.fast_predict() in steaming after the first update(). """
#     # input
#     n_actions = 2
#     n_features = 2
#     n_samples = 10000
#     n_iterations = 10
#     n_jobs = n_actions
#     sampling = False
#     params_sample = {'tune': 500, 'draws': 1000, 'chains': 2, 'init': 'adapt_diag', 'cores': 1, 'target_accept': 0.95,
#                      'progressbar': False}

#     # run test
#     mu_t, simga_t = run_cmab_predict_streaming(n_actions=n_actions, n_features=n_features, n_samples=n_samples,
#                                                n_iterations=n_iterations, n_jobs=n_jobs, sampling=sampling,
#                                                params_sample=params_sample, verbose=verbose)


# def test_cmab_time_predict_2_5_10000_fp():
#     """ Test cmab.fast_predict() in steaming after the first update(). """
#     # input
#     n_actions = 2
#     n_features = 5
#     n_samples = 10000
#     n_iterations = 10
#     n_jobs = n_actions
#     sampling = False
#     params_sample = {'tune': 500, 'draws': 1000, 'chains': 2, 'init': 'adapt_diag', 'cores': 1, 'target_accept': 0.95,
#                      'progressbar': False}

#     # run test
#     mu_t, simga_t = run_cmab_predict_streaming(n_actions=n_actions, n_features=n_features, n_samples=n_samples,
#                                                n_iterations=n_iterations, n_jobs=n_jobs, sampling=sampling,
#                                                params_sample=params_sample, verbose=verbose)


# def test_cmab_time_predict_2_100_10000_fp():
#     """ Test cmab.fast_predict() in steaming after the first update(). """
#     # input
#     n_actions = 2
#     n_features = 100
#     n_samples = 10000
#     n_iterations = 10
#     n_jobs = n_actions
#     sampling = False
#     params_sample = {'tune': 500, 'draws': 1000, 'chains': 2, 'init': 'adapt_diag', 'cores': 1, 'target_accept': 0.95,
#                      'progressbar': False}

#     # run test
#     mu_t, simga_t = run_cmab_predict_streaming(n_actions=n_actions, n_features=n_features, n_samples=n_samples,
#                                                n_iterations=n_iterations, n_jobs=n_jobs, sampling=sampling,
#                                                params_sample=params_sample, verbose=verbose)


# def test_cmab_time_predict_5_2_10000_fp():
#     """ Test cmab.fast_predict() in steaming after the first update(). """
#     # input
#     n_actions = 5
#     n_features = 2
#     n_samples = 10000
#     n_iterations = 10
#     n_jobs = n_actions
#     sampling = False
#     params_sample = {'tune': 500, 'draws': 1000, 'chains': 2, 'init': 'adapt_diag', 'cores': 1, 'target_accept': 0.95,
#                      'progressbar': False}

#     # run test
#     mu_t, simga_t = run_cmab_predict_streaming(n_actions=n_actions, n_features=n_features, n_samples=n_samples,
#                                                n_iterations=n_iterations, n_jobs=n_jobs, sampling=sampling,
#                                                params_sample=params_sample, verbose=verbose)


# def test_cmab_time_predict_20_2_1_fp():
#     """ Test cmab.fast_predict() in steaming after the first update(). """
#     # input
#     n_actions = 20
#     n_features = 2
#     n_samples = 1
#     n_iterations = 10
#     n_jobs = n_actions
#     sampling = False
#     params_sample = {'tune': 500, 'draws': 1000, 'chains': 2, 'init': 'adapt_diag', 'cores': 1, 'target_accept': 0.95,
#                      'progressbar': False}

#     # run test
#     mu_t, simga_t = run_cmab_predict_streaming(n_actions=n_actions, n_features=n_features, n_samples=n_samples,
#                                                n_iterations=n_iterations, n_jobs=n_jobs, sampling=sampling,
#                                                params_sample=params_sample, verbose=verbose)


# def test_cmab_time_predict_20_100_1_fp():
#     """ Test cmab.fast_predict() in steaming after the first update(). """
#     # input
#     n_actions = 20
#     n_features = 100
#     n_samples = 1
#     n_iterations = 10
#     n_jobs = n_actions
#     sampling = False
#     params_sample = {'tune': 500, 'draws': 1000, 'chains': 2, 'init': 'adapt_diag', 'cores': 1, 'target_accept': 0.95,
#                      'progressbar': False}

#     # run test
#     mu_t, simga_t = run_cmab_predict_streaming(n_actions=n_actions, n_features=n_features, n_samples=n_samples,
#                                                n_iterations=n_iterations, n_jobs=n_jobs, sampling=sampling,
#                                                params_sample=params_sample, verbose=verbose)


# # test with sampling

# def test_cmab_time_predict_2_2_1_w_s():
#     """ Test cmab.predict() in steaming after the first update(). """
#     # input
#     n_actions = 2
#     n_features = 2
#     n_samples = 1
#     n_iterations = 10
#     n_jobs = n_actions
#     sampling = True
#     params_sample = {'tune': 500, 'draws': 1000, 'chains': 2, 'init': 'adapt_diag', 'cores': 1, 'target_accept': 0.95,
#                      'progressbar': False}

#     # run test
#     mu_t, simga_t = run_cmab_predict_streaming(n_actions=n_actions, n_features=n_features, n_samples=n_samples,
#                                                n_iterations=n_iterations, n_jobs=n_jobs, sampling=sampling,
#                                                params_sample=params_sample, verbose=verbose)


# def test_cmab_time_predict_2_2_10000_w_s():
#     """ Test cmab.predict() in steaming after the first update(). """
#     # input
#     n_actions = 2
#     n_features = 2
#     n_samples = 10000
#     n_iterations = 10
#     n_jobs = n_actions
#     sampling = True
#     params_sample = {'tune': 500, 'draws': 1000, 'chains': 2, 'init': 'adapt_diag', 'cores': 1, 'target_accept': 0.95,
#                      'progressbar': False}

#     # run test
#     mu_t, simga_t = run_cmab_predict_streaming(n_actions=n_actions, n_features=n_features, n_samples=n_samples,
#                                                n_iterations=n_iterations, n_jobs=n_jobs, sampling=sampling,
#                                                params_sample=params_sample, verbose=verbose)


# def test_cmab_time_predict_2_5_10000_w_s():
#     """ Test cmab.predict() in steaming after the first update(). """
#     # input
#     n_actions = 2
#     n_features = 5
#     n_samples = 10000
#     n_iterations = 10
#     n_jobs = n_actions
#     sampling = True
#     params_sample = {'tune': 500, 'draws': 1000, 'chains': 2, 'init': 'adapt_diag', 'cores': 1, 'target_accept': 0.95,
#                      'progressbar': False}

#     # run test
#     mu_t, simga_t = run_cmab_predict_streaming(n_actions=n_actions, n_features=n_features, n_samples=n_samples,
#                                                n_iterations=n_iterations, n_jobs=n_jobs, sampling=sampling,
#                                                params_sample=params_sample, verbose=verbose)


# def test_cmab_time_predict_2_100_10000_w_s():
#     """ Test cmab.predict() in steaming after the first update(). """
#     # input
#     n_actions = 2
#     n_features = 100
#     n_samples = 10000
#     n_iterations = 10
#     n_jobs = n_actions
#     sampling = True
#     params_sample = {'tune': 500, 'draws': 1000, 'chains': 2, 'init': 'adapt_diag', 'cores': 1, 'target_accept': 0.95,
#                      'progressbar': False}

#     # run test
#     mu_t, simga_t = run_cmab_predict_streaming(n_actions=n_actions, n_features=n_features, n_samples=n_samples,
#                                                n_iterations=n_iterations, n_jobs=n_jobs, sampling=sampling,
#                                                params_sample=params_sample, verbose=verbose)

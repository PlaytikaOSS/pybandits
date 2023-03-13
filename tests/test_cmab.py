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

# import pytest

# import numpy as np
# import numpy.testing as npt
# from deepdiff import DeepDiff
# from sklearn.datasets import make_classification

# from pybandits.cmab import Cmab


# def test_init():
#     """Test init() function."""
#     # inputs
#     random_seed = 1
#     n_features = 3
#     n_jobs = 3

#     mu_alpha = {"action1": 1.0, "action2": 2.0}
#     mu_betas = {"action1": [1.0, 4.0, 7.1], "action2": [2.0, 11.2, 10.2]}
#     sigma_alpha = {"action1": 15.0, "action2": 20.0}
#     sigma_betas = {"action1": [1.0, 0.0, 100.0], "action2": [2.0, 11.2, 43.3]}
#     nu_alpha = {"action1": 2.0, "action2": 4.0}
#     nu_betas = {"action1": [1.0, 18.1, 0.0], "action2": [2.0, 21.1, 11.0]}

#     n_actions = 2
#     actions_ids = ["action" + str(i + 1) for i in range(n_actions)]
#     params_sample = {
#         "tune": 500,
#         "draws": 1000,
#         "chains": 2,
#         "init": "adapt_diag",
#         "cores": 1,
#         "target_accept": 0.95,
#         "progressbar": False,
#     }

#     # init model with default arguments
#     cmab = Cmab(n_features=n_features, actions_ids=actions_ids)

#     # init model with specified arguments
#     cmab = Cmab(
#         n_features=n_features,
#         actions_ids=actions_ids,
#         params_sample=params_sample,
#         n_jobs=n_jobs,
#         mu_alpha=mu_alpha,
#         mu_betas=mu_betas,
#         sigma_alpha=sigma_alpha,
#         sigma_betas=sigma_betas,
#         nu_alpha=nu_alpha,
#         nu_betas=nu_betas,
#         random_seed=random_seed,
#     )

#     assert cmab._n_features == n_features, "n_features was not correctly initialized"
#     assert cmab._actions_ids == actions_ids, "actions were not correctly initialized"
#     assert cmab._params_sample == params_sample, "CMAB parameters were not correctly initialized"
#     assert cmab._n_jobs == n_jobs, "n_jobs was not correctly initialized"
#     assert cmab._mu_alpha == mu_alpha, "mu_alpha was not correctly initialized"
#     assert cmab._mu_betas == mu_betas, "mu_betas was not correctly initialized"
#     assert cmab._sigma_alpha == sigma_alpha, "sigma_alpha was not correctly initialized"
#     assert cmab._sigma_betas == sigma_betas, "sigma_betas was not correctly initialized"
#     assert cmab._nu_alpha == nu_alpha, "nu_alpha was not correctly initialized"
#     assert cmab._nu_betas == nu_betas, "nu_betas was not correctly initialized"
#     assert cmab._random_seed == random_seed, "random_seed was not correctly initialized"
#     assert cmab._traces["action1"] is None, "_traces[action1] is not None"
#     assert cmab._traces["action2"] is None, "_traces[action1] is not None"


# def test_update_parallel():
#     n_actions = 2
#     update(n_actions=n_actions, n_jobs=n_actions)


# def test_update_sequential():
#     n_actions = 2
#     update(n_actions=n_actions, n_jobs=1)


# def update(n_actions, n_jobs):
#     """Test update() function."""
#     random_seed = 1
#     n_features = 2
#     n_samples = 10
#     actions_ids = ["action" + str(i + 1) for i in range(n_actions)]
#     params_sample = {
#         "tune": 500,
#         "draws": 1000,
#         "chains": 2,
#         "init": "adapt_diag",
#         "cores": 1,
#         "target_accept": 0.95,
#         "progressbar": False,
#         "return_inferencedata": False,
#     }
#     X, _ = make_classification(
#         n_samples=n_samples,
#         n_features=n_features,
#         n_informative=2,
#         n_redundant=0,
#         n_classes=2,
#         random_state=random_seed,
#     )

#     # approximation of floating number might give slightly different values depending on the machine executing
#     # the tests try to decrease this value if tests fail.
#     # decimals = 4

#     # run test
#     cmab = Cmab(n_features=n_features, actions_ids=actions_ids, n_jobs=n_jobs, random_seed=random_seed)

#     assert cmab._traces["action1"] is None, "_traces[action1] is not None"
#     assert cmab._traces["action2"] is None, "_traces[action1] is not None"

#     # update
#     rewards = [1, 0, 1, 1, 1, 0, 0, 1, 0, 1]
#     actions = [
#         "action1",
#         "action1",
#         "action2",
#         "action2",
#         "action1",
#         "action2",
#         "action1",
#         "action2",
#         "action2",
#         "action2",
#     ]

#     cmab.update(X=X, actions=actions, rewards=rewards)

#     assert len(cmab._traces["action1"]) == params_sample["draws"], "the number of draws in trace do not match"
#     assert (
#         len(cmab._traces["action1"]["alpha"]) == params_sample["draws"] * params_sample["chains"]
#     ), "the number of draws in the trace of action1-alpha do not match"
#     assert (
#         len(cmab._traces["action1"]["beta0"]) == params_sample["draws"] * params_sample["chains"]
#     ), "the number of draws in the trace of action1-beta0 do not match"
#     assert (
#         len(cmab._traces["action1"]["beta1"]) == params_sample["draws"] * params_sample["chains"]
#     ), "the number of draws in the trace of action1-beta1 do not match"
#     assert (
#         len(cmab._traces["action1"]["p"]) == params_sample["draws"] * params_sample["chains"]
#     ), "the number of draws in the trace of action1-p do not match"
#     assert (
#         len(cmab._traces["action1"]["linear_combination"]) == params_sample["draws"] * params_sample["chains"]
#     ), "the number of draws in the trace of action1-linear_combination do not match"

#     assert len(cmab._traces["action2"]) == params_sample["draws"], "the number of draws in trace do not match"
#     assert (
#         len(cmab._traces["action2"]["alpha"]) == params_sample["draws"] * params_sample["chains"]
#     ), "the number of draws in the trace of action2-alpha do not match"
#     assert (
#         len(cmab._traces["action2"]["beta0"]) == params_sample["draws"] * params_sample["chains"]
#     ), "the number of draws in the trace of action2-beta0 do not match"
#     assert (
#         len(cmab._traces["action2"]["beta1"]) == params_sample["draws"] * params_sample["chains"]
#     ), "the number of draws in the trace of action2-beta1 do not match"
#     assert (
#         len(cmab._traces["action2"]["p"]) == params_sample["draws"] * params_sample["chains"]
#     ), "the number of draws in the trace of action2-p do not match"
#     assert (
#         len(cmab._traces["action2"]["linear_combination"]) == params_sample["draws"] * params_sample["chains"]
#     ), "the number of draws in the trace of action2-linear_combination do not match"

#     # update
#     rewards = [0, 1, 0, 0, 1, 1, 1, 0, 0, 1]
#     actions = [
#         "action2",
#         "action2",
#         "action1",
#         "action2",
#         "action1",
#         "action1",
#         "action1",
#         "action2",
#         "action1",
#         "action1",
#     ]

#     cmab.update(X=X, actions=actions, rewards=rewards)

#     assert len(cmab._traces["action1"]) == params_sample["draws"], "the number of draws in trace do not match"
#     assert (
#         len(cmab._traces["action1"]["alpha"]) == params_sample["draws"] * params_sample["chains"]
#     ), "the number of draws in the trace of action1-alpha do not match"
#     assert (
#         len(cmab._traces["action1"]["beta0"]) == params_sample["draws"] * params_sample["chains"]
#     ), "the number of draws in the trace of action1-beta0 do not match"
#     assert (
#         len(cmab._traces["action1"]["beta1"]) == params_sample["draws"] * params_sample["chains"]
#     ), "the number of draws in the trace of action1-beta1 do not match"
#     assert (
#         len(cmab._traces["action1"]["p"]) == params_sample["draws"] * params_sample["chains"]
#     ), "the number of draws in the trace of action1-p do not match"
#     assert (
#         len(cmab._traces["action1"]["linear_combination"]) == params_sample["draws"] * params_sample["chains"]
#     ), "the number of draws in the trace of action1-linear_combination do not match"

#     assert len(cmab._traces["action2"]) == params_sample["draws"], "the number of draws in trace do not match"
#     assert (
#         len(cmab._traces["action2"]["alpha"]) == params_sample["draws"] * params_sample["chains"]
#     ), "the number of draws in the trace of action2-alpha do not match"
#     assert (
#         len(cmab._traces["action2"]["beta0"]) == params_sample["draws"] * params_sample["chains"]
#     ), "the number of draws in the trace of action2-beta0 do not match"
#     assert (
#         len(cmab._traces["action2"]["beta1"]) == params_sample["draws"] * params_sample["chains"]
#     ), "the number of draws in the trace of action2-beta1 do not match"
#     assert (
#         len(cmab._traces["action2"]["p"]) == params_sample["draws"] * params_sample["chains"]
#     ), "the number of draws in the trace of action2-p do not match"
#     assert (
#         len(cmab._traces["action2"]["linear_combination"]) == params_sample["draws"] * params_sample["chains"]
#     ), "the number of draws in the trace of action2-linear_combination do not match"


# def test_predict_after_init():
#     """Test predict after init. The cmab should randomly recommend actions with equal probability."""

#     random_seed = 1
#     n_features = 2
#     n_samples = 10000
#     n_actions = 5
#     actions_ids = ["action" + str(i + 1) for i in range(n_actions)]
#     params_sample = {
#         "tune": 500,
#         "draws": 1000,
#         "chains": 2,
#         "init": "adapt_diag",
#         "cores": 1,
#         "target_accept": 0.95,
#         "progressbar": False,
#     }

#     # run test
#     prop = []

#     for i in range(10):
#         cmab = Cmab(
#             n_features=n_features, actions_ids=actions_ids, params_sample=params_sample, random_seed=random_seed + i
#         )

#         X, _ = make_classification(
#             n_samples=n_samples,
#             n_features=n_features,
#             n_informative=2,
#             n_redundant=0,
#             n_classes=2,
#             random_state=random_seed,
#         )

#         # Recommend actions
#         pred, _ = cmab.predict(X)

#         action, count = np.unique(pred, return_counts=True)
#         prop.append(count / n_samples)

#     assert np.min(prop) == 0.1867, "The proportion of recommended actions should be ~ 0.20"
#     assert np.max(prop) == 0.2114, "The proportion of recommended actions should be ~ 0.20"


# def test_predict_after_update():
#     """Test predict() after update()."""
#     random_seed = 1
#     n_features = 2
#     n_samples = 20
#     n_actions = 2
#     actions_ids = ["action" + str(i + 1) for i in range(n_actions)]
#     params_sample = {
#         "tune": 500,
#         "draws": 1000,
#         "chains": 2,
#         "init": "adapt_diag",
#         "cores": 1,
#         "target_accept": 0.95,
#         "progressbar": False,
#     }
#     X, _ = make_classification(
#         n_samples=n_samples,
#         n_features=n_features,
#         n_informative=2,
#         n_redundant=0,
#         n_classes=2,
#         random_state=random_seed,
#     )

#     # Split X in 2
#     X1 = X[: int(n_samples * 1 / 2)]
#     X2 = X[int(n_samples * 1 / 2) : n_samples]

#     # init model
#     cmab = Cmab(n_features=n_features, actions_ids=actions_ids, params_sample=params_sample, random_seed=random_seed)

#     pred_actions_batch_1, probs_batch_1 = cmab.predict(X1)  # predict batch
#     pred_actions_row_1, probs_row_1 = cmab.predict(np.array([0.1, 1.5]).reshape(1, n_features))  # predict 1 row

#     assert set(pred_actions_batch_1).issubset({"action1", "action2"}), "predicted actions do not match"
#     assert np.array_equal(
#         probs_batch_1,
#         [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]],
#     )
#     assert set(pred_actions_row_1).issubset({"action1", "action2"}), "predicted actions do not match"
#     assert np.array_equal(probs_row_1, [[0.5], [0.5]])

#     rewards = [1, 0, 1, 1, 1, 0, 0, 1, 1, 1]

#     cmab.update(X=X1, actions=pred_actions_batch_1, rewards=rewards)

#     pred_actions_batch_2, probs_batch_2 = cmab.predict(X2)
#     pred_actions_row_2, probs_row_2 = cmab.predict(np.array([0.3, -0.2]).reshape(1, n_features))  # predict 1 row

#     assert set(pred_actions_batch_2).issubset({"action1", "action2"}), "predicted actions do not match"
#     assert len(probs_batch_2) == 2, "probs_batch_2 has not the correct shape"
#     assert len(probs_batch_2[0]) == 10, "probs has not the correct shape"
#     assert np.min(probs_batch_2) >= 0, "probs is lower than 0"
#     assert np.max(probs_batch_2) <= 1, "probs is higher than 1"
#     assert set(pred_actions_row_2).issubset({"action1", "action2"}), "predicted actions do not match"
#     assert len(probs_row_2) == 2, "probs_batch_2 has not the correct shape"
#     assert len(probs_row_2[0]) == 1, "probs has not the correct shape"
#     assert np.min(probs_row_2) >= 0, "probs is lower than 0"
#     assert np.max(probs_row_2) <= 1, "probs is higher than 1"


# def test_convergence():
#     """Test model convergence."""
#     random_seed = 1
#     n_features = 2
#     n_samples = 10000
#     n_actions = 2
#     actions_ids = ["action" + str(i + 1) for i in range(n_actions)]
#     params_sample = {
#         "tune": 500,
#         "draws": 1000,
#         "chains": 2,
#         "init": "adapt_diag",
#         "cores": 1,
#         "target_accept": 0.95,
#         "progressbar": False,
#     }

#     X, _ = make_classification(
#         n_samples=n_samples * 2,
#         n_features=n_features,
#         n_informative=2,
#         n_redundant=0,
#         n_classes=2,
#         random_state=random_seed,
#     )

#     # Split X in 2
#     X1 = X[:n_samples]
#     X2 = X[n_samples : 2 * n_samples]

#     # init model
#     cmab = Cmab(n_features=n_features, actions_ids=actions_ids, params_sample=params_sample, random_seed=random_seed)

#     # Recommend actions for X1
#     pred_actions_1, _ = cmab.predict(X1)

#     # Count recommended actions
#     action, count = np.unique(pred_actions_1, return_counts=True)

#     assert (
#         action[0] == "action1" and count[0] / n_samples > 0.49
#     ), "The proportion of recommended actions should be ~ 0.50. It must not be < 0.49"
#     assert (
#         action[1] == "action2" and count[1] / n_samples < 0.51
#     ), "The proportion of recommended actions should be ~ 0.50. It must not be > 0.51"

#     # Update model
#     rewards = [1 if a == "action2" else 0 for a in pred_actions_1]
#     cmab.update(X=X1, actions=pred_actions_1, rewards=rewards)

#     # Recommend actions for X2
#     pred_actions_2, _ = cmab.predict(X2)

#     # Count recommended actions
#     action, count = np.unique(pred_actions_2, return_counts=True)

#     assert (
#         action[0] == "action2" and count[0] / n_samples == 1.0
#     ), "The proportion of recommended actions should be equal to 1"


# def test_invalid_input_update():
#     """Test update() function with invalid input."""
#     random_seed = 1
#     n_features = 2
#     n_samples = 10
#     n_actions = 2
#     actions_ids = ["action" + str(i + 1) for i in range(n_actions)]
#     params_sample = {
#         "tune": 500,
#         "draws": 1000,
#         "chains": 2,
#         "init": "adapt_diag",
#         "cores": 1,
#         "target_accept": 0.95,
#         "progressbar": False,
#     }

#     X, _ = make_classification(
#         n_samples=n_samples,
#         n_features=n_features,
#         n_informative=2,
#         n_redundant=0,
#         n_classes=2,
#         random_state=random_seed,
#     )

#     # run test
#     cmab = Cmab(n_features=n_features, actions_ids=actions_ids, params_sample=params_sample, random_seed=random_seed)

#     # update
#     rewards = [1, 0, 1, 1, 1, 0, 0, 1, 0, 1]
#     rewards_invalid = [1, 0, 1, 1, 1, 0, 0, 1, 0, 2]
#     pred_actions = [
#         "action1",
#         "action1",
#         "action2",
#         "action2",
#         "action1",
#         "action2",
#         "action1",
#         "action2",
#         "action2",
#         "action2",
#     ]
#     pred_actions_invalid = [
#         "action1",
#         "action1",
#         "action1",
#         "action1",
#         "action1",
#         "action1",
#         "action1",
#         "action1",
#         "action1",
#         "action3",
#     ]

#     with pytest.raises(ValueError):  # wrong shape of pred_actions
#         cmab.update(X=X, actions=pred_actions[:5], rewards=rewards)
#     with pytest.raises(ValueError):  # wrong shape of rewards
#         cmab.update(X=X, actions=pred_actions, rewards=rewards[:5])
#     with pytest.raises(ValueError):  # wrong shape of len(X)
#         cmab.update(X=X[:5], actions=pred_actions, rewards=rewards)
#     with pytest.raises(ValueError):  # wrong shape of X features
#         cmab.update(X=X[:, 1], actions=pred_actions, rewards=rewards)
#     with pytest.raises(ValueError):  # invalid actions
#         cmab.update(X=X, actions=pred_actions_invalid, rewards=rewards)
#     with pytest.raises(ValueError):  # invalid rewards
#         cmab.update(X=X, actions=pred_actions, rewards=rewards_invalid)


# def test_fast_predict():
#     """Test fast_predict()."""
#     random_seed = 1
#     n_features = 2
#     n_samples = 20
#     n_actions = 2
#     actions_ids = ["action" + str(i + 1) for i in range(n_actions)]
#     params_sample = {
#         "tune": 500,
#         "draws": 1000,
#         "chains": 2,
#         "init": "adapt_diag",
#         "cores": 1,
#         "target_accept": 0.95,
#         "progressbar": False,
#     }
#     X, _ = make_classification(
#         n_samples=n_samples,
#         n_features=n_features,
#         n_informative=2,
#         n_redundant=0,
#         n_classes=2,
#         random_state=random_seed,
#     )

#     # Split X in 2
#     X1 = X[: int(n_samples * 1 / 2)]
#     X2 = X[int(n_samples * 1 / 2) : n_samples]

#     # init model
#     cmab = Cmab(n_features=n_features, actions_ids=actions_ids, params_sample=params_sample, random_seed=random_seed)

#     pred_actions_batch_1, probs_batch_1 = cmab.fast_predict(X1)
#     pred_actions_row_1, probs_row_1 = cmab.fast_predict(np.array([0.1, 1.5]).reshape(1, n_features))

#     assert np.array_equal(
#         pred_actions_batch_1,
#         ["action1", "action2", "action2", "action2", "action1", "action1", "action2", "action2", "action1",
#          "action1"],
#     ), "predicted actions do not match"
#     assert np.array_equal(
#         probs_batch_1,
#         [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]],
#     )
#     assert pred_actions_row_1 == ["action1"]
#     assert np.array_equal(probs_row_1, [[0.5], [0.5]])

#     rewards = [1, 0, 1, 1, 1, 0, 0, 1, 1, 1]

#     cmab.update(X=X1, actions=pred_actions_batch_1, rewards=rewards)

#     pred_actions_batch_2, probs_batch_2 = cmab.fast_predict(X2)
#     pred_actions_row_2, probs_row_2 = cmab.fast_predict(np.array([0.3, -0.2]).reshape(1, n_features))
#     assert np.array_equal(
#         pred_actions_batch_2,
#         ["action2", "action2", "action2", "action1", "action2", "action2", "action1", "action1", "action2",
#          "action2"],
#     )
#     assert len(probs_batch_2) == 2, "probs_batch_2 has not the correct shape"
#     assert len(probs_batch_2[0]) == 10, "probs has not the correct shape"
#     assert np.min(probs_batch_2) >= 0, "probs is lower than 0"
#     assert np.max(probs_batch_2) <= 1, "probs is higher than 1"
#     assert pred_actions_row_2 == ["action2"]
#     assert len(probs_row_2) == 2, "probs_batch_2 has not the correct shape"
#     assert len(probs_row_2[0]) == 1, "probs has not the correct shape"
#     assert np.min(probs_row_2) >= 0, "probs is lower than 0"
#     assert np.max(probs_row_2) <= 1, "probs is higher than 1"


# def test_update_predict_not_all_actions():
#     """Test update() function when not all actions are present in the batch of samples."""
#     random_seed = 1
#     n_features = 2
#     n_samples = 10
#     n_actions = 3
#     actions_ids = ["action" + str(i + 1) for i in range(n_actions)]

#     X, _ = make_classification(
#         n_samples=n_samples,
#         n_features=n_features,
#         n_informative=2,
#         n_redundant=0,
#         n_classes=2,
#         random_state=random_seed,
#     )

#     # init cmab
#     cmab = Cmab(n_features=n_features, actions_ids=actions_ids, random_seed=random_seed)
#     cmab.fast_predict(X)

#     rewards = [1, 0, 1, 1, 1, 0, 0, 1, 0, 1]
#     actions = [
#         "action1",
#         "action1",
#         "action2",
#         "action2",
#         "action1",
#         "action2",
#         "action1",
#         "action2",
#         "action2",
#         "action2",
#     ]

#     # first update with ['action1', 'action2'] and without ['action 3']
#     cmab.update(X=X, actions=actions, rewards=rewards)
#     cmab.fast_predict(X)

#     assert cmab._mu_alpha["action1"] != 0
#     assert cmab._mu_alpha["action2"] != 0
#     assert cmab._mu_alpha["action3"] == 0
#     assert cmab._sigma_alpha["action1"] != 10
#     assert cmab._sigma_alpha["action2"] != 10
#     assert cmab._sigma_alpha["action3"] == 10
#     assert cmab._mu_betas["action1"] != [0, 0]
#     assert cmab._mu_betas["action2"] != [0, 0]
#     assert cmab._mu_betas["action3"] == [0, 0]
#     assert cmab._sigma_betas["action1"] != [10, 10]
#     assert cmab._sigma_betas["action2"] != [10, 10]
#     assert cmab._sigma_betas["action3"] == [10, 10]

#     # assert mean, std traces
#     mu_alpha = cmab._mu_alpha.copy()
#     sigma_alpha = cmab._sigma_alpha.copy()
#     mu_betas = cmab._mu_betas.copy()
#     sigma_betas = cmab._sigma_betas.copy()

#     t1 = cmab._traces["action1"][0]
#     t2 = cmab._traces["action2"][0]

#     rewards = [0, 1, 0, 0, 1, 1, 1, 0, 0, 1]
#     actions = ["action2"] * n_samples

#     # second update with ['action2'] and without ['action1, 'action 3']
#     cmab.update(X=X, actions=actions, rewards=rewards)
#     cmab.fast_predict(X)

#     # assert mean, std traces
#     npt.assert_almost_equal(cmab._mu_alpha["action1"], mu_alpha["action1"])
#     assert cmab._mu_alpha["action2"] != mu_alpha["action2"]
#     assert cmab._mu_alpha["action3"] == 0
#     npt.assert_almost_equal(cmab._sigma_alpha["action1"], sigma_alpha["action1"])
#     assert cmab._sigma_alpha["action2"] != sigma_alpha["action2"]
#     assert cmab._sigma_alpha["action3"] == 10
#     npt.assert_almost_equal(cmab._mu_betas["action1"], mu_betas["action1"])
#     assert cmab._mu_betas["action2"] != mu_betas["action2"]
#     assert cmab._mu_betas["action3"] == [0, 0]
#     npt.assert_almost_equal(cmab._sigma_betas["action1"], sigma_betas["action1"])
#     assert cmab._sigma_betas["action2"] != sigma_betas["action2"]
#     assert cmab._sigma_betas["action3"] == [10, 10]

#     # assert traces
#     assert DeepDiff(cmab._traces["action1"][0], t1) == {}, "Trace for action 1 must not change."
#     assert DeepDiff(cmab._traces["action2"][0], t2) != {}, "Trace for actions 2 must change."
#     assert cmab._traces["action3"] is None, "Trace for actions 3 must not change."

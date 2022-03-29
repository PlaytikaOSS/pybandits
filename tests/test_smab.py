# MIT License
#
# Copyright (c) 2022 Playtika Ltd.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import random

import pytest

from pybandits.core.smab import Smab


def test_init_model():
    """
    Initialize several models with different parameters and make sure the internals of the model (actions,
    successes and failures) are properly initialized.
    """
    ts = ['1', '2', '3']
    rs = {'1': 234, '2': 32, '3': 90}
    ps = {'1': 24, '2': 3, '3': 32}
    parameters = [
        {'action_ids': ts, 'success_priors': None, 'failure_priors': None},
        {'action_ids': ts, 'success_priors': rs, 'failure_priors': ps},
        {'action_ids': ts}
    ]

    for params in parameters:
        smab = Smab(**params)
        assert (smab._actions_ids == ts), 'Treatments are not initialized correctly.'
        if 'success_priors' in params and params['success_priors'] is not None:
            assert (smab._success_counters == rs and smab._failure_counters == ps)
        for t in ts:
            assert (t in smab._success_counters and t in smab._failure_counters)
            if 'success_priors' in params and params['success_priors'] is not None:
                assert (smab._success_counters[t] == rs[t] and smab._failure_counters[t] == ps[t]), \
                    'unmatching prior successes and failures.'
            else:
                assert (smab._success_counters[t] == 1 and smab._failure_counters[t] == 1), \
                    'default successes and failures must be 1.'


def test_success_not_specified():
    """
    Either both successes and failures should be specified, or neither. Expecting IOError.
    """
    params1 = {'action_ids': ['1', '2', '3'],
               'success_priors': None,
               'failure_priors': {'1': 12, '2': 14, '3': 15}}
    with pytest.raises(ValueError):
        _ = Smab(**params1)


def test_bad_success_key():
    """
    Bad input: wrong key in successes. Expecting IOError
    """
    params2 = {'action_ids': ['1', '2', '3'],
               'success_priors': {'1': 12, '2': 14, 'wrong_key': 15},
               'failure_priors': {'1': 12, '2': 14, '3': 15}}
    with pytest.raises(ValueError):
        _ = Smab(**params2)


def test_bad_prior_parameters():
    """
    Prior parameters should be > 0. Expecting IOError
    """
    params3 = {'action_ids': ['1', '2', '3'],
               'success_priors': {'1': 12, '2': -1, '3': 14},
               'failure_priors': {'1': 12, '2': 14, '3': 0}}
    with pytest.raises(ValueError):
        _ = Smab(**params3)


def test_model_prediction_in_actions():
    """
    Make sure the prediction is one of the model's actions
    """
    params = {'action_ids': ['1', '2', '3']}
    n_samples = 100
    smab = Smab(**params)
    best_actions, probs = smab.predict(n_samples=n_samples)
    assert (all(action in params['action_ids'] for action in best_actions)), \
        'predicted value must be one of the actions.'
    assert (type(best_actions) is list and all(isinstance(x, str) for x in best_actions)), \
        'Predicted action must be a string'
    assert (type(probs) is list and all(isinstance(x, dict) for x in probs)), 'Predicted probs must be dict'
    assert (len(best_actions) == len(probs) == n_samples)


def test_model_feedback_with_priors():
    """
    Create a smab with prior parameters and update it many times. Make sure that the internal counters of the model are
    updated as they should.
    """
    actions = ['1', '2', '3', '4', '5']
    params = {'action_ids': actions,
              'success_priors': dict((t, random.randint(1, 100)) for t in actions),
              'failure_priors': dict((t, random.randint(1, 100)) for t in actions)}

    smab = Smab(**params)
    counters = {}
    for t in actions:
        counters[t] = [params['success_priors'][t], params['failure_priors'][t]]
    for i in range(1000):
        t = random.choice(actions)
        n_successes = random.randint(1, 500)
        n_failures = random.randint(1, 500)
        counters[t][0] += n_successes
        counters[t][1] += n_failures
        smab.update(action_id=t, n_successes=n_successes, n_failures=n_failures)
        for action in actions:
            assert (counters[action] == [smab._success_counters[action], smab._failure_counters[action]])


def test_model_feedback_without_priors():
    """
    Create a smab with no prior parameters and update it many times. Make sure that the internal counters of the
    model are updated as they should.
    """
    actions = ['1', '2', '3', '4', '5']
    params = {'action_ids': actions}

    smab = Smab(**params)
    counters = {}
    for t in actions:
        counters[t] = [1, 1]
    for i in range(1000):
        t = random.choice(actions)
        n_successes = random.randint(1, 500)
        n_failures = random.randint(1, 500)
        counters[t][0] += n_successes
        counters[t][1] += n_failures
        smab.update(action_id=t, n_successes=n_successes, n_failures=n_failures)
        for action in actions:
            assert (counters[action] == [smab._success_counters[action], smab._failure_counters[action]])


def test_random_seed():
    """
    Make sure that when a random seed is specified, the outputs of the ML model are deterministic.
    Create a reference model with fixed seed, than iteratively create 100 similar models with the same seed. Apply
    the same combination of updates and predictions to the test models and ensure that their predictions are always
    the same as the reference model's predictions.
    """
    actions = ['1', '2', '3', '4', '5']
    params = {'action_ids': actions,
              'random_seed': 42}

    reference_model = Smab(**params)
    reference_preds = []
    for t in actions:
        reference_model.update(t, 2, 3)
        reference_preds = reference_preds + [reference_model.predict(n_samples=10)[0]]

    for i in range(100):
        smab = Smab(**params)
        preds = []
        for t in actions:
            smab.update(t, 2, 3)
            preds = preds + [smab.predict(n_samples=10)[0]]
        assert (preds == reference_preds)


def test_update_bad_input():
    actions = ['1', '2', '3', '4', '5']
    params = {'action_ids': actions,
              'success_priors': dict((t, random.randint(1, 100)) for t in actions),
              'failure_priors': dict((t, random.randint(1, 100)) for t in actions)}
    smab = Smab(**params)

    with pytest.raises(ValueError):
        smab.update(random.choice(actions), -1, 6)
    with pytest.raises(ValueError):
        smab.update(random.choice(actions), 4, -1)
    with pytest.raises(TypeError):
        smab.update(random.choice(actions), 1.5, 2)
    with pytest.raises(TypeError):
        smab.update(random.choice(actions), 2, 1.5)
    with pytest.raises(ValueError):
        smab.update('wrong action', 1, 1)
    with pytest.raises(TypeError):
        smab.update(1, 1, 1)


def test_batch_update():
    batch = [
        {'action_id': '1', 'n_successes': 1, 'n_failures': 2},
        {'action_id': '2', 'n_successes': 3, 'n_failures': 4},
        {'action_id': '3', 'n_successes': 5, 'n_failures': 6},
        {'action_id': '1', 'n_successes': 7, 'n_failures': 8},
    ]
    smab = Smab(action_ids=['1', '2', '3'])
    smab.batch_update(batch)

    assert(smab._success_counters['1'] == 9 and smab._failure_counters['1'] == 11)
    assert (smab._success_counters['2'] == 4 and smab._failure_counters['2'] == 5)
    assert (smab._success_counters['3'] == 6 and smab._failure_counters['3'] == 7)


def test_forbidden_actions():
    """
    Test predict function with forbidden actions.
    """
    actions = ['1', '2', '3', '4', '5']
    params = {'action_ids': actions,
              'success_priors': dict((t, 1) for t in actions),
              'failure_priors': dict((t, 1) for t in actions)}
    smab = Smab(**params)

    assert set(smab.predict(n_samples=1000, forbidden_actions=['2', '3', '4', '5'])[0]) == {'1'}
    assert set(smab.predict(n_samples=1000, forbidden_actions=['1', '3'])[0]) == {'2', '4', '5'}
    assert set(smab.predict(n_samples=1000, forbidden_actions=['1'])[0]) == {'2', '3', '4', '5'}
    assert set(smab.predict(n_samples=1000, forbidden_actions=[])[0]) == {'1', '2', '3', '4', '5'}

    with pytest.raises(TypeError):  # not a list
        assert set(smab.predict(n_samples=1000, forbidden_actions=1)[0])
    with pytest.raises(ValueError):  # invalid action_ids
        assert set(smab.predict(n_samples=1000, forbidden_actions=['1', '100', 'a', 5])[0])
    with pytest.raises(ValueError):  # duplicates
        assert set(smab.predict(n_samples=1000, forbidden_actions=['1', '1', '2'])[0])
    with pytest.raises(ValueError):  # all actions forbidden
        assert set(smab.predict(n_samples=1000, forbidden_actions=['1', '2', '3', '4', '5'])[0])
    with pytest.raises(ValueError):  # all actions forbidden (unordered)
        assert set(smab.predict(n_samples=1000, forbidden_actions=['5', '4', '2', '3', '1'])[0])


def test_return_probs():
    """
    Test predict function with forbidden actions.
    """
    actions = ['action A', 'action B', 'action C', 'action D', 'action E']
    params = {'action_ids': actions, 'random_seed': 42}
    smab = Smab(**params)

    best_action, probs = smab.predict()

    assert(best_action[0] == 'action A'), 'The best action returned do not match the expected value'
    assert(probs[0] == {'action A': 0.9757708986968694,
                        'action B': 0.5601155051094101,
                        'action C': 0.5415020530650179,
                        'action D': 0.9607666225987184,
                        'action E': 0.9476908718028403}), 'The dict of probs returned do not match the expected value'

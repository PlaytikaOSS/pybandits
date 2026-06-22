# API documentation

## Architecture

- `pybandits` contains the Multi-Armed Bandits (MAB) algorithms.

  > - {mod}`pybandits.smab`
  > - {mod}`pybandits.cmab`

- `pybandits` provides a configuration framework for the MAB algorithms.

  > - {mod}`pybandits.model`
  > - {mod}`pybandits.quantitative_model`
  > - {mod}`pybandits.strategy`
  > - {mod}`pybandits.actions_manager`

- `pybandits` provides a simulation environment framework.

  > - {mod}`pybandits.smab_simulator`
  > - {mod}`pybandits.cmab_simulator`

- `pybandits` provides an OPE framework.
  > - {mod}`pybandits.offline_policy_evaluator`
  > - {mod}`pybandits.offline_policy_estimator`

- `pybandits` provides utilities for cost-control bandit tuning and transfer learning.
  > - {mod}`pybandits.subsidy_factor`
  > - {mod}`pybandits.transfer`

## API

Please visit the full [API](fullapi.md) for details.

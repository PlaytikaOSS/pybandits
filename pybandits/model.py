# MIT License
#
# Copyright (c) 2023 Playtika Ltd.
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


from random import betavariate
from typing import List, Tuple

from numpy import array, c_, exp, insert, mean, multiply, ones, sqrt, std
from numpy.typing import ArrayLike
from pydantic import (
    Field,
    NonNegativeFloat,
    PositiveInt,
    confloat,
    model_validator,
    validate_call,
)
from pymc import Bernoulli, Data, Deterministic, sample
from pymc import Model as PymcModel
from pymc import StudentT as PymcStudentT
from pymc.math import sigmoid
from pytensor.tensor import dot
from scipy.stats import t

from pybandits.base import BinaryReward, Model, Probability, PyBanditsBaseModel


class BaseBeta(Model):
    """
    Beta Distribution model for Bernoulli multi-armed bandits.

    Parameters
    ----------
    n_successes: PositiveInt = 1
        Counter of the number of successes.
    n_failures: PositiveInt = 1
        Counter of the number of failures.
    """

    n_successes: PositiveInt = 1
    n_failures: PositiveInt = 1

    @model_validator(mode="before")
    @classmethod
    def both_or_neither_counters_are_defined(cls, values):
        if hasattr(values, "n_successes") != hasattr(values, "n_failures"):
            raise ValueError("Either both or neither n_successes and n_failures should be specified.")
        return values

    @property
    def count(self) -> int:
        """
        The total amount of successes and failures collected.
        """
        return self.n_successes + self.n_failures

    @property
    def mean(self) -> float:
        """
        The success rate i.e. n_successes / (n_successes + n_failures).
        """
        return self.n_successes / self.count

    @property
    def std(self) -> float:
        """
        The corrected standard deviation (Bessel's correction) of the binary distribution of successes and failures.
        """
        return sqrt((self.n_successes * self.n_failures) / (self.count * (self.count - 1)))

    @validate_call
    def update(self, rewards: List[BinaryReward]):
        """
        Update n_successes and and n_failures.

        Parameters
        ----------
        rewards: List[BinaryReward]
            A list of binary rewards.
        """
        self.n_successes += sum(rewards)
        self.n_failures += len(rewards) - sum(rewards)

    def sample_proba(self) -> Probability:
        """
        Sample the probability of getting a positive reward.

        Returns
        -------
        prob: Probability
            Probability of getting a positive reward.
        """
        return betavariate(self.n_successes, self.n_failures)  # type: ignore


class Beta(BaseBeta):
    """
    Beta Distribution model for Bernoulli multi-armed bandits.
    """


class BetaCC(BaseBeta):
    """
    Beta Distribution model for Bernoulli multi-armed bandits with cost control.

    Parameters
    ----------
    cost: NonNegativeFloat
        Cost associated to the Beta distribution.
    """

    cost: NonNegativeFloat


class BaseBetaMO(Model):
    """
    Beta Distribution model for Bernoulli multi-armed bandits with multi-objectives.

    Parameters
    ----------
    counters: List[Beta] of shape (n_objectives,)
        List of Beta distributions.
    """

    counters: List[Beta]

    @validate_call
    def sample_proba(self) -> List[Probability]:
        """
        Sample the probability of getting a positive reward.

        Returns
        -------
        prob: List[Probability]
            Probabilities of getting a positive reward for each objective.
        """
        return [x.sample_proba() for x in self.counters]

    @validate_call
    def update(self, rewards: List[List[BinaryReward]]):
        """
        Update the Beta model using the provided rewards.

        Parameters
        ----------
        rewards: List[List[BinaryReward]]
            A list of rewards, where each reward is in turn a list containing the reward of the Beta model
            associated to each objective.
            For example, `[[1, 1], [1, 0], [1, 1], [1, 0], [1, 1]]`.
        """
        if any(len(x) != len(self.counters) for x in rewards):
            raise AttributeError("The shape of rewards is incorrect")

        for i, counter in enumerate(self.counters):
            counter.update([r[i] for r in rewards])


class BetaMO(BaseBetaMO):
    """
    Beta Distribution model for Bernoulli multi-armed bandits with multi-objectives.

    Parameters
    ----------
    counters: List[Beta] of shape (n_objectives,)
        List of Beta distributions.
    """


class BetaMOCC(BaseBetaMO):
    """
    Beta Distribution model for Bernoulli multi-armed bandits with multi-objectives and cost control.

    Parameters
    ----------
    counters: List[BetaCC] of shape (n_objectives,)
        List of Beta distributions.
    cost: NonNegativeFloat
        Cost associated to the Beta distribution.
    """

    cost: NonNegativeFloat


class StudentT(PyBanditsBaseModel):
    """
    Student's t-distribution.

    Parameters
    ----------
    mu: float
        Mean of the Student's t-distribution.
    sigma: float
        Standard deviation of the Student's t-distribution.
    nu: float
        Degrees of freedom.
    """

    mu: confloat(allow_inf_nan=False) = 0.0
    sigma: confloat(allow_inf_nan=False) = 10.0
    nu: confloat(allow_inf_nan=False) = 5.0


class BaseBayesianLogisticRegression(Model):
    """
    Base Bayesian Logistic Regression model.

    It is modeled as:

        y = sigmoid(alpha + beta1 * x1 + beta2 * x2 + ... + betaN * xN)

    where the alpha and betas coefficients are Student's t-distributions.

    Parameters
    ----------
    alpha: StudentT
        Student's t-distribution of the alpha coefficient.
    betas: StudentT
        Student's t-distributions of the betas coefficients.
    params_sample: Dict
        Parameters for the function pymc.sample()
    """

    alpha: StudentT
    betas: List[StudentT] = Field(..., min_items=1)

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def check_context_matrix(self, context: ArrayLike):
        """
        Check and cast context matrix.

        Parameters
        ----------
        context: ArrayLike of shape (n_samples, n_features)
            Matrix of contextual features.

        Returns
        -------
        context: pandas DataFrame of shape (n_samples, n_features)
            Matrix of contextual features.
        """
        try:
            n_cols_context = array(context).shape[1]
        except Exception as e:
            raise AttributeError(f"Context must be an ArrayLike with {len(self.betas)} columns: {e}.")
        if n_cols_context != len(self.betas):
            raise AttributeError(f"Shape mismatch: context must have {len(self.betas)} columns.")

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def sample_proba(self, context: ArrayLike) -> Tuple[Probability, float]:
        """
        Compute the probability of getting a positive reward from the sampled regression coefficients and the context.

        Parameters
        ----------
        context : ArrayLike
            Context matrix of shape (n_samples, n_features).

        Returns
        -------
        prob: ndarray of shape (n_samples)
            Probability of getting a positive reward.
        weighted_sum: ndarray of shape (n_samples)
            Weighted sums between contextual feature values and sampled coefficients.
        """

        # check input args
        self.check_context_matrix(context=context)

        # extend context with a column of 1 to handle the dot product with the intercept
        context_ext = c_[ones((len(context), 1)), context]

        # sample alpha and beta coefficient values from student-t distributions once for each sample
        alpha = t.rvs(df=self.alpha.nu, loc=self.alpha.mu, scale=self.alpha.sigma, size=len(context_ext))
        betas = array(
            [
                t.rvs(df=self.betas[i].nu, loc=self.betas[i].mu, scale=self.betas[i].sigma, size=len(context_ext))
                for i in range(len(self.betas))
            ]
        )

        # create coefficients matrix
        coeff = insert(arr=betas, obj=0, values=alpha, axis=0)

        # extract the weighted sum between the context and the coefficients
        weighted_sum = multiply(context_ext, coeff.T).sum(axis=1)

        # compute the probability with the sigmoid function
        prob = 1.0 / (1.0 + exp(-weighted_sum))

        return prob, weighted_sum

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def update(
        self,
        context: ArrayLike,
        rewards: List[BinaryReward],
        tune=500,
        draws=1000,
        chains=2,
        init="adapt_diag",
        cores=1,
        target_accept=0.95,
        progressbar=False,
        return_inferencedata=False,
        **kwargs,
    ):
        """
        Update the model parameters.

        Parameters
        ----------
        context : ArrayLike
            Context matrix of shape (n_samples, n_features).
        rewards: List[BinaryReward]
            A list of binary rewards.
        """

        # check input args
        self.check_context_matrix(context=context)
        if len(context) != len(rewards):
            AttributeError("Shape mismatch: context and rewards must have the same length.")

        with PymcModel() as _:
            # update intercept (alpha) and coefficients (betas)
            # if model was never updated priors_parameters = default arguments
            # else priors_parameters are calculated from traces of the previous update
            alpha = PymcStudentT("alpha", mu=self.alpha.mu, sigma=self.alpha.sigma, nu=self.alpha.nu)
            betas = [
                PymcStudentT("beta" + str(i), mu=self.betas[i].mu, sigma=self.betas[i].sigma, nu=self.betas[i].nu)
                for i in range(len(self.betas))
            ]

            context = Data("context", context)
            rewards = Data("rewards", rewards)

            # Likelihood (sampling distribution) of observations
            weighted_sum = Deterministic("weighted_sum", alpha + dot(betas, context.T))
            p = Deterministic("p", sigmoid(weighted_sum))

            # Bernoulli random vector with probability of success given by sigmoid function and actual data as observed
            _ = Bernoulli("likelihood", p=p, observed=rewards)

            # update traces object by sampling from posterior distribution
            trace = sample(
                tune=tune,
                draws=draws,
                chains=chains,
                init=init,
                cores=cores,
                target_accept=target_accept,
                progressbar=progressbar,
                return_inferencedata=return_inferencedata,
                **kwargs,
            )

            # compute mean and std of the coefficients distributions
            self.alpha.mu = mean(trace["alpha"])
            self.alpha.sigma = std(trace["alpha"], ddof=1)
            for i in range(len(self.betas)):
                self.betas[i].mu = mean(trace["beta" + str(i)])
                self.betas[i].sigma = std(trace["beta" + str(i)], ddof=1)


class BayesianLogisticRegression(BaseBayesianLogisticRegression):
    """
    Bayesian Logistic Regression model.

    It is modeled as:

        y = sigmoid(alpha + beta1 * x1 + beta2 * x2 + ... + betaN * xN)

    where the alpha and betas coefficients are Student's t-distributions.

    Parameters
    ----------
    alpha: StudentT
        Student's t-distribution of the alpha coefficient.
    betas: StudentT
        Student's t-distributions of the betas coefficients.
    params_sample: Dict
        Parameters for the function pymc.sample()
    """


class BayesianLogisticRegressionCC(BaseBayesianLogisticRegression):
    """
    Bayesian Logistic Regression model with cost control.

    It is modeled as:

        y = sigmoid(alpha + beta1 * x1 + beta2 * x2 + ... + betaN * xN)

    where the alpha and betas coefficients are Student's t-distributions.

    Parameters
    ----------
    alpha: StudentT
        Student's t-distribution of the alpha coefficient.
    betas: StudentT
        Student's t-distributions of the betas coefficients.
    params_sample: Dict
        Parameters for the function pymc.sample()
    cost: NonNegativeFloat
        Cost associated to the Bayesian Logistic Regression model.
    """

    cost: NonNegativeFloat


def create_bayesian_logistic_regression_cold_start(n_betas: PositiveInt) -> BayesianLogisticRegression:
    """
    Utility function to create a Bayesian Logistic Regression model, with default parameters.

    It is modeled as:

        y = sigmoid(alpha + beta1 * x1 + beta2 * x2 + ... + betaN * xN)

    where the alpha and betas coefficients are Student's t-distributions.

    Parameters
    ----------
    n_betas : PositiveInt
        The number of betas of the Bayesian Logistic Regression model. This is also the number of features expected
        after in the context matrix.

    Returns
    -------
    blr: BayesianLogisticRegression
        The Bayesian Logistic Regression model.
    """
    return BayesianLogisticRegression(alpha=StudentT(), betas=[StudentT() for _ in range(n_betas)])


def create_bayesian_logistic_regression_cc_cold_start(
    n_betas: PositiveInt, cost: NonNegativeFloat
) -> BayesianLogisticRegressionCC:
    """
    Utility function to create a Bayesian Logistic Regression model with cost control, with default parameters.

    It is modeled as:

        y = sigmoid(alpha + beta1 * x1 + beta2 * x2 + ... + betaN * xN)

    where the alpha and betas coefficients are Student's t-distributions.

    Parameters
    ----------
    n_betas : PositiveInt
        The number of betas of the Bayesian Logistic Regression model. This is also the number of features expected
        after in the context matrix.
    cost: NonNegativeFloat
        Cost associated to the Bayesian Logistic Regression model.

    Returns
    -------
    blr: BayesianLogisticRegressionCC
        The Bayesian Logistic Regression model.
    """
    return BayesianLogisticRegressionCC(alpha=StudentT(), betas=[StudentT() for _ in range(n_betas)], cost=cost)

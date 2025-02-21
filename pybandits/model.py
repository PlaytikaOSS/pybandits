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
import warnings
from abc import ABC, abstractmethod
from random import betavariate
from typing import Any, List, Literal, Optional, Tuple, Union, Dict
import pytensor.tensor as pt

import numpy as np
import pymc.math as pmath
from numpy import array, c_, insert, mean, multiply, ones, sqrt, std
from numpy.typing import ArrayLike
from pymc import Bernoulli, Data, Deterministic, fit, sample
from pymc import Model as PymcModel
from pymc import StudentT as PymcStudentT
import pymc as pm
from pytensor.tensor import TensorVariable, dot
from scipy.stats import t

from pybandits.base import BinaryReward, Probability, PyBanditsBaseModel
from pybandits.pydantic_version_compatibility import (
    PYDANTIC_VERSION_1,
    PYDANTIC_VERSION_2,
    Field,
    NonNegativeFloat,
    PositiveInt,
    confloat,
    model_validator,
    root_validator,
    pydantic_version,
    validate_call,
)

UpdateMethods = Literal["MCMC", "VI"]


class Model(PyBanditsBaseModel, ABC):
    """
    Class to model the prior distributions.
    """

    @abstractmethod
    def sample_proba(self) -> Probability:
        """
        Sample the probability of getting a positive reward.
        """

    @abstractmethod
    def update(self, rewards: List[Any]):
        """
        Update the model parameters.
        """


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


class BetaMO(Model):
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

    @classmethod
    def cold_start(cls, n_objectives: PositiveInt, **kwargs) -> "BetaMO":
        """
        Utility function to create a Bayesian Logistic Regression model  or child model with cost control,
        with default parameters.

        It is modeled as:

            y = sigmoid(alpha + beta1 * x1 + beta2 * x2 + ... + betaN * xN)

        where the alpha and betas coefficients are Student's t-distributions.

        Parameters
        ----------
        n_betas : PositiveInt
            The number of betas of the Bayesian Logistic Regression model. This is also the number of features expected
            after in the context matrix.
        kwargs: Dict[str, Any]
            Additional arguments for the Bayesian Logistic Regression child model.

        Returns
        -------
        blr: BayesianLogisticRegrssion
            The Bayesian Logistic Regression model.
        """
        counters = n_objectives * [Beta()]
        blr = cls(counters=counters, **kwargs)
        return blr


class BetaMOCC(BetaMO):
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

class StudentTArray(PyBanditsBaseModel):
    shape: Optional[List[PositiveInt]] = None
    params_dict: Optional[Dict[str, Union[List[float],List[List[float]]]]] = None
    mu: confloat(allow_inf_nan=False) = 0.0
    sigma: confloat(allow_inf_nan=False) = 10.0
    nu: confloat(allow_inf_nan=False) = 5.0

    @root_validator(pre=False, skip_on_failure=False)
    def initialize_arrays(cls, values):
        if values.get("params_dict") is None:
            if values.get("shape") is None:
                raise ValueError("either legal 'shape' or 'params_dict' must be specified")
            
            shape = values.get("shape")
            values["params_dict"] = {}
            values["params_dict"]["mu"] = (np.zeros(shape) + values.get("mu")).tolist()
            values["params_dict"]["sigma"] = np.full(shape, values.get("sigma")).tolist()
            values["params_dict"]["nu"] = np.full(shape, values.get("nu")).tolist()
        else:
            mu = values["params_dict"].get('mu')
            sigma = values["params_dict"].get('sigma')
            nu = values["params_dict"].get('nu')
            
            if not (mu and sigma and nu):
                raise ValueError("params_dict must contain mu, sigma, and nu")
            if not (np.array(mu).shape == np.array(sigma).shape == np.array(nu).shape):
                raise ValueError("mu, sigma, and nu must have the same sizes")

        return values

    def __eq__(self, other):
        for key, value in self.params_dict.items():
            if not np.array_equal(value, other.params_dict[key]):
                return False
        return True


    class Config:
        arbitrary_types_allowed = True



class BaseBayesianModel(Model, ABC):
    update_method: str = "MCMC"
    update_kwargs: Optional[dict] = None
    _default_update_kwargs = dict(draws=1000, progressbar=False, return_inferencedata=False)
    _default_mcmc_kwargs = dict(
        tune=500,
        draws=1000,
        chains=2,
        init="adapt_diag",
        cores=1,
        target_accept=0.95,
        progressbar=False,
        return_inferencedata=False,
    )

    _default_variational_inference_kwargs = dict(method="advi", 
                                                 n=2000)

    class Config:
        arbitrary_types_allowed = True

    if pydantic_version == PYDANTIC_VERSION_1:

        @model_validator(mode="before")
        @classmethod
        def arrange_update_kwargs(cls, values):
            update_kwargs = cls._get_value_with_default("update_kwargs", values)
            update_method = cls._get_value_with_default("update_method", values)
            if update_kwargs is None:
                update_kwargs = cls._default_update_kwargs
            if update_method == "VI":
                update_kwargs = {**cls._default_variational_inference_kwargs, **update_kwargs}
            elif update_method == "MCMC":
                update_kwargs = {**cls._default_mcmc_kwargs, **update_kwargs}
            else:
                raise ValueError("Invalid update method.")
            values["update_kwargs"] = update_kwargs
            values["update_method"] = update_method
            return values

    elif pydantic_version == PYDANTIC_VERSION_2:

        @model_validator(mode="after")
        def arrange_update_kwargs(self):
            if self.update_kwargs is None:
                self.update_kwargs = self._default_update_kwargs
            if self.update_method == "VI":
                self.update_kwargs = {**self._default_variational_inference_kwargs, **self.update_kwargs}
            elif self.update_method == "MCMC":
                self.update_kwargs = {**self._default_mcmc_kwargs, **self.update_kwargs}
            else:
                raise ValueError("Invalid update method.")
            return self

    else:
        raise ValueError(f"Unsupported pydantic version: {pydantic_version}")

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def check_context_matrix(self, context: ArrayLike):
        try:
            n_cols_context = np.array(context).shape[1]
        except Exception as e:
            raise AttributeError(f"Context must be an ArrayLike with {self.expected_input} columns: {e}.")
        if n_cols_context != self.expected_input:
            raise AttributeError(f"Shape mismatch: context must have {self.expected_input} columns.")
    
    @property
    @abstractmethod
    def expected_input(self):
        pass

    @abstractmethod
    def sample_proba(self, context: ArrayLike) -> Tuple[Probability, float]:
        pass

    @abstractmethod
    def update(self, context: ArrayLike, rewards: List[BinaryReward]):
        pass

    @abstractmethod
    def cold_start(cls, **kwargs) -> "BaseBayesianModel":
        pass


class BayesianLogisticRegression2(BaseBayesianModel):
    """
    Base Bayesian Logistic Regression model.

    It is modeled as:

        y = sigmoid(alpha + beta1 * x1 + beta2 * x2 + ... + betaN * xN)

    where the alpha and betas coefficients are Student's t-distributions.

    Parameters
    ----------
    alpha : StudentT
        Student's t-distribution of the alpha coefficient.
    betas : StudentT
        Student's t-distributions of the betas coefficients.
    update_method : UpdateMethods, defaults to "MCMC"
        The strategy for computing posterior quantities of the Bayesian models in the update function. Such as Markov
        chain Monte Carlo ("MCMC") or Variational Inference ("VI"). Check UpdateMethods in pybandits.model for the
        full list.
    update_kwargs : Optional[dict], uses default values if not specified
        Additional arguments to pass to the update method.
    """

    alpha: StudentT
    if pydantic_version == PYDANTIC_VERSION_1:
        betas: List[StudentT] = Field(..., min_items=1)
    elif pydantic_version == PYDANTIC_VERSION_2:
        betas: List[StudentT] = Field(..., min_length=1)
    else:
        raise ValueError("Invalid version.")

    @property
    def expected_input(self):
        return len(self.betas)
    
    @classmethod
    def _stable_sigmoid(cls, x: Union[np.ndarray, TensorVariable]) -> Union[np.ndarray, TensorVariable]:
        """
        Vectorized sigmoid function that avoids overflow and underflow.
        Compatible with both numpy and PyMC3 tensors.

        Parameters
        ----------
        x : Union[np.ndarray, TensorVariable]
            Input values.

        Returns
        -------
        prob : Union[np.ndarray, TensorVariable]
            Sigmoid function applied to the input values.
        """
        backend = np if isinstance(x, np.ndarray) else pmath
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            prob = backend.where(x >= 0, 1 / (1 + backend.exp(-x)), backend.exp(x) / (1 + backend.exp(x)))
        return prob

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
        prob = self._stable_sigmoid(weighted_sum)

        return prob, weighted_sum

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def update(self, context: ArrayLike, rewards: List[BinaryReward]):
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
            beta_mu = [b.mu for b in self.betas]
            beta_sigma = [b.sigma for b in self.betas]
            beta_nu = [b.nu for b in self.betas]
            betas = PymcStudentT("betas", mu=beta_mu, sigma=beta_sigma, nu=beta_nu, shape=len(self.betas))

            context = Data("context", context, mutable=False)
            rewards = Data("rewards", rewards, mutable=False)

            # Likelihood (sampling distribution) of observations
            weighted_sum = Deterministic("weighted_sum", alpha + dot(betas, context.T))
            p = Deterministic("p", self._stable_sigmoid(weighted_sum))

            # Bernoulli random vector with probability of success given by sigmoid function and actual data as observed
            _ = Bernoulli("likelihood", p=p, observed=rewards)

            # update traces object by sampling from posterior distribution
            if self.update_method == "VI":
                # variational inference
                update_kwargs = self.update_kwargs.copy()
                approx = fit(method=update_kwargs.pop("method"))
                trace = approx.sample(**update_kwargs)
            elif self.update_method == "MCMC":
                # MCMC
                trace = sample(**self.update_kwargs)
            else:
                raise ValueError("Invalid update method.")

            # compute mean and std of the coefficients distributions
            self.alpha.mu = mean(trace["alpha"])
            self.alpha.sigma = std(trace["alpha"], ddof=1)
            betas_mu = mean(trace["betas"], axis=0)
            betas_std = std(trace["betas"], axis=0, ddof=1)
            self.betas = [
                StudentT(mu=mu, sigma=sigma, nu=beta.nu) for mu, sigma, beta in zip(betas_mu, betas_std, self.betas)
            ]

    @classmethod
    def cold_start(
            cls,
            n_features: PositiveInt,
            update_method: UpdateMethods = "MCMC",
            update_kwargs: Optional[dict] = None,
            **kwargs,
    ) -> "BayesianLogisticRegression2":
        """
        Utility function to create a Bayesian Logistic Regression model  or child model with cost control,
        with default parameters.

        It is modeled as:

            y = sigmoid(alpha + beta1 * x1 + beta2 * x2 + ... + betaN * xN)

        where the alpha and betas coefficients are Student's t-distributions.

        Parameters
        ----------
        n_features : PositiveInt
            The number of betas of the Bayesian Logistic Regression model. This is also the number of features expected
            after in the context matrix.
        update_method : UpdateMethods, defaults to "MCMC"
            The strategy for computing posterior quantities of the Bayesian models in the update function. Such as Markov
            chain Monte Carlo ("MCMC") or Variational Inference ("VI"). Check UpdateMethods in pybandits.model for the
            full list.
        update_kwargs : Optional[dict], uses default values if not specified
            Additional arguments to pass to the update method.
        kwargs: Dict[str, Any]
            Additional arguments for the Bayesian Logistic Regression child model.

        Returns
        -------
        blr: BayesianLogisticRegrssion
            The Bayesian Logistic Regression model.
        """
        return cls(
            alpha=StudentT(),
            betas=[StudentT() for _ in range(n_features)],
            update_method=update_method,
            update_kwargs=update_kwargs,
            **kwargs,
        )


class BayesianNeuralNetwork(Model):
    update_method: str = "MCMC"
    update_kwargs: Optional[Union[Dict[str, Any], dict[str, Dict[str, Any]]]] = None

    posterior_params: List[Dict[str, StudentTArray]] 

    _default_update_trace_kwargs = dict(draws=1000, progressbar=False, return_inferencedata=False) 
    _default_mcmc_trace_kwargs = dict( 
        tune=500,
        draws=1000,
        chains=2,
        init="adapt_diag",
        cores=1,
        target_accept=0.95,
        progressbar=False,
        return_inferencedata=False,
    )  

    _default_variational_inference_fit_kwargs = dict(method="advi") 
    _default_variational_inference_trace_kwargs = dict(n=2000)
    

    class Config:
        arbitrary_types_allowed = True

    if pydantic_version == PYDANTIC_VERSION_1:

        @model_validator(mode="before")
        @classmethod
        def arrange_update_kwargs(cls, values):
            update_kwargs = cls._get_value_with_default("update_kwargs", values)
            update_method = cls._get_value_with_default("update_method", values)
                    
            if update_kwargs is None: 
                update_kwargs= dict()
                update_kwargs["trace"] = cls._default_update_trace_kwargs
                
            if update_method == "VI":       
                update_kwargs["trace"] = {**cls._default_variational_inference_trace_kwargs, **update_kwargs.get("trace",{})}
                update_kwargs["fit"] = {**cls._default_variational_inference_fit_kwargs, **update_kwargs.get("fit",{})}

            elif update_method == "MCMC":     
                update_kwargs["trace"] = {**cls._default_mcmc_trace_kwargs, **update_kwargs.get("trace",{})}
            else:
                raise ValueError("Invalid update method.")
                
            values["update_kwargs"] = update_kwargs
            values["update_method"] = update_method
            return values

    elif pydantic_version == PYDANTIC_VERSION_2:
        @model_validator(mode="after")
        def arrange_update_kwargs(self):
            
            if self.update_kwargs is None:
                self.update_kwargs = dict()
                self.update_kwargs["trace"] = self._default_update_trace_kwargs
 
            if self.update_method == "VI":
                    self.update_kwargs["trace"] = {**self._default_variational_inference_trace_kwargs, **self.update_kwargs.get("trace",{})}
                    self.update_kwargs["fit"] = {**self._default_variational_inference_fit_kwargs, **self.update_kwargs.get("fit",{})}
 
            elif self.update_method == "MCMC":
                self.update_kwargs["trace"] = {**self._default_mcmc_trace_kwargs, **self.update_kwargs.get("trace",{})}
            else:
                raise ValueError("Invalid update method.")
            return self

    else:
        raise ValueError(f"Unsupported pydantic version: {pydantic_version}")

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def check_context_matrix(self, context: ArrayLike):
        try:
            n_cols_context = np.array(context).shape[1]
        except Exception as e:
            raise AttributeError(f"Context must be an ArrayLike with {self.expected_input} columns: {e}.")
        if n_cols_context != self.expected_input:
            raise AttributeError(f"Shape mismatch: context must have {self.expected_input} columns.")
    

    @property
    def expected_input(self):
        return self.posterior_params[0]["w"].shape[0]
    
    def create_model(self, x, y, is_sampelwise):
        
        with pm.Model() as _model:
            # Define data variables using minibatches  
            bnn_output = pm.MutableData("ann_output", y)

            for layer_ind in range(len(self.posterior_params)):
                layer_params = self.posterior_params[layer_ind]
                w_shape = np.array(layer_params["w"].params_dict["mu"]).shape # without it n_features = 1 doesn't work
                w = PymcStudentT(f"w{layer_ind}", **layer_params["w"].params_dict, shape=w_shape)
                b = PymcStudentT(f"b{layer_ind}", **layer_params["b"].params_dict)
                
                if layer_ind == 0:
                    if is_sampelwise:    
                        x_tensor = pt.vector("x_tensor")
                        linear_transform_func = pm.math.dot(x_tensor, w) + b
                        compiled_linear_transform = pm.pytensorf.compile_pymc(inputs=[x_tensor], outputs=linear_transform_func)
                        linear_transform = pt.as_tensor_variable([compiled_linear_transform(row) for row in x], name=f"linear_transform{layer_ind}")
                    else:
                        bnn_input = pm.MutableData("bnn_input", x)
                        linear_transform = pm.Deterministic(f"linear_transform{layer_ind}", pm.math.dot(bnn_input, w) + b)
                else:
                    linear_transform = pm.Deterministic(f"linear_transform{layer_ind}", pm.math.dot(act, w) + b)

                
                if layer_ind < len(self.posterior_params) - 1:
                    act = pm.math.tanh(linear_transform)

            logit = pm.Deterministic("logit", linear_transform.squeeze())
            prob = pm.Deterministic("prob", pm.math.sigmoid(logit))

            # Binary classification -> Bernoulli likelihood
            out = pm.Bernoulli(
                "out",
                p=prob,
                observed=bnn_output
            )
        return _model

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def sample_proba(self, context: ArrayLike) -> Tuple[Probability, float]:
        # check input args
        self.check_context_matrix(context=context) 

        _context = np.array(context, ndmin = 2)
        dummy_y = np.zeros(len(context), dtype=np.int64)
        _model = self.create_model(_context, dummy_y, is_sampelwise=True)
        
        with _model:
            trace = pm.sample_prior_predictive(samples=1)
        
        prob = trace['prior']['prob'].values.reshape(-1)
        weighted_sum = trace['prior']['logit'].values.reshape(-1)


        return prob, weighted_sum

    def update(self, context: ArrayLike, rewards: List[BinaryReward]):
        self.check_context_matrix(context=context)
        if len(context) != len(rewards):
            AttributeError("Shape mismatch: context and rewards must have the same length.")

        _context = np.array(context, ndmin = 2)
        _model = self.create_model(x=_context, y=rewards, is_sampelwise=False)
        with _model:
            # update traces object by sampling from posterior distribution
            if self.update_method == "VI":
                # variational inference
                update_kwargs = self.update_kwargs.copy()
                approx = fit(**update_kwargs["fit"])
                trace = approx.sample(**update_kwargs["trace"])
            elif self.update_method == "MCMC":
                # MCMC
                trace = sample(**self.update_kwargs["trace"])
            else:
                raise ValueError("Invalid update method.")

    
        for layer_ind in range(len(self.posterior_params)):
            name = f"w{layer_ind}"
        
            w_mu = np.mean(trace[name], axis=0)
            w_sigma = np.std(trace[name], axis=0)
            
            self.posterior_params[layer_ind]["w"].params_dict["mu"] = w_mu.tolist()
            self.posterior_params[layer_ind]["w"].params_dict["sigma"] = w_sigma.tolist()
            
            name = f"b{layer_ind}"
            b_mu = np.mean(trace[name], axis=0)
            b_sigma = np.std(trace[name], axis=0)
   
            self.posterior_params[layer_ind]["b"].params_dict["mu"] = b_mu.tolist()
            self.posterior_params[layer_ind]["b"].params_dict["sigma"] = b_sigma.tolist()
   

    @classmethod
    def cold_start(cls, dim_list : List[PositiveInt], update_method: UpdateMethods= "MCMC",
                   update_kwargs: Optional[dict] = None,
                   **kwargs) -> "BayesianNeuralNetwork":
            
        _dim_list = dim_list.copy()
        _dim_list.append(1)
        # if any(dim <= 0 for dim in _dim_list):
        #     raise ValueError("All dimensions must be positive integers.")

        posterior_params = []
        for layer_ind in range(len(_dim_list) - 1):
            input_dim = _dim_list[layer_ind]
            output_dim = _dim_list[layer_ind + 1]
            w_param = StudentTArray(shape=[input_dim, output_dim])
            b_param = StudentTArray(shape=[output_dim])
            posterior_params.append(dict(w=w_param, b=b_param))
        
        return cls(posterior_params=posterior_params, update_method=update_method, update_kwargs=update_kwargs)

    def __eq__(self, other):
        for i, layer in enumerate(self.posterior_params):
            if not layer["w"] == other.posterior_params[i]["w"] or not layer["b"] == other.posterior_params[i]["b"]:
                return False
        return True
 

class BayesianLogisticRegression(BayesianNeuralNetwork):
    alpha: StudentT
    if pydantic_version == PYDANTIC_VERSION_1:
        betas: List[StudentT] = Field(..., min_items=1)
    elif pydantic_version == PYDANTIC_VERSION_2:
        betas: List[StudentT] = Field(..., min_length=1)
    else:
        raise ValueError("Invalid version.")
    
    @model_validator(mode="before")
    def set_posterior_params(cls, values):
        input_dim = len(values["betas"])
        output_dim = 1

        w_param = StudentTArray(shape=[input_dim, output_dim])
        betas = values["betas"].copy()
        
        for i,beta in enumerate(betas):
            if type(beta) == dict:
                beta = StudentT(**beta) # handle from_state
            w_param.params_dict["mu"][i][0] = beta.mu
            w_param.params_dict["sigma"][i][0] = beta.sigma
            w_param.params_dict["nu"][i][0] = beta.nu

        b_param = StudentTArray(shape=[output_dim])
        alpha = values["alpha"].copy()
        
        if type(alpha) == dict:
            alpha = StudentT(**alpha)

        b_param.params_dict["mu"][0] = alpha.mu
        b_param.params_dict["sigma"][0] = alpha.sigma
        b_param.params_dict["nu"][0] = alpha.nu


        values["posterior_params"] = [dict(w=w_param, b=b_param)]
        return values
    
    @classmethod
    def cold_start(cls, n_features, update_method: UpdateMethods = "MCMC",
                   update_kwargs: Optional[Union[Dict[str, Any], dict[str, Dict[str, Any]]]] = None,
                   **kwargs) -> "BayesianLogisticRegression":
        return cls(
        alpha=StudentT(),
        betas=[StudentT() for _ in range(n_features)],
        update_method=update_method,
        update_kwargs=update_kwargs,
        **kwargs,
    )


class BayesianLogisticRegressionCC(BayesianLogisticRegression):
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
    update_method : UpdateMethods, defaults to "MCMC"
        The strategy for computing posterior quantities of the Bayesian models in the update function. Such as Markov
        chain Monte Carlo ("MCMC") or Variational Inference ("VI"). Check UpdateMethods in pybandits.model for the
        full list.
    update_kwargs : Optional[dict], uses default values if not specified
        Additional arguments to pass to the update method.
    cost: NonNegativeFloat
        Cost associated to the Bayesian Logistic Regression model.
    """

    cost: NonNegativeFloat


import torch.nn as nn


class Torch_BNN(nn.Module):
    def __init__(self, in_dim, out_dim=1, hid_dim=10):
        super().__init__()
        self.activation = nn.Tanh()  # or nn.ReLU()
        self.layer1 = nn.Linear(in_dim, hid_dim)  # Input to hidden layer
        self.layer2 = nn.Linear(hid_dim, out_dim)  # Hidden to output layer

    def forward(self, x_features, x_actions):
        x = torch.cat((x_features, x_actions))
        x = self.activation(self.layer1(x))
        logits = self.layer2(x).squeeze()
        return logits


class QuantitativeBNNModel:
    def __init__(self, in_dim, hid_dim, mu=0, sigma=10.):
        self.model = BayesianNeuralNetwork(in_dim, hid_dim, mu, sigma)
        self.torch_model = Torch_BNN(in_dim, hid_dim)

        def sample_proba(self, x):
            trace = self._model.sample(x)
            probabilities = trace['prior_predictive']['out'].squeeze().mean(axis=0).values
            return probabilities


    

    

if __name__ == '__main__':
    c_params = [-4, 2, 3, 3, 1, -3]
    n_bias_features = 3
    n_features = 5
    n_samples_train = 30000
    n_samples_val = 2000
    n_samples_test = 10000

    a1 = StudentTArray(shape=(3, 3))
    a2 = StudentTArray(shape=(2, 3))
    a1 == a2


    BayesianLogisticRegression(alpha=StudentT(), betas=[StudentT() for _ in range(3)])
        #posterior_params: List[Dict[str, StudentTArray]] 
    posterior_params = [dict(w=StudentTArray(shape=(3, 3)), b=StudentTArray(shape=(3,)))]
    betas = [[StudentT() for _ in range(3)], [StudentT() for _ in range(3)]]
    
    # xx = BayesianNeuralNetwork.cold_start(dim_list=[5, 10, 1], update_method="VI")
    # xxx = BayesianLogisticRegression2.cold_start(n_features=5, update_method="VI")
    # xxx = BayesianLogisticRegression(alpha=StudentT(mu=1, sigma=2), betas=[StudentT(mu=2, sigma=4), StudentT(mu=6, sigma=7)])
    # import json
    # from pybandits.utils import to_serializable_dict
    # d = {'1': xx}
    # tt = to_serializable_dict(d)
  
    # ttt = json.loads(json.dumps(tt, default=dict))


    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


    def create_data(c_params, n_samples, n_features, n_bias_features):
        features_obs = np.random.uniform(0, 1, size=(n_samples, n_features))
        amount = np.random.uniform(0, 2, size=(n_samples))
        logit_bias = c_params[-1] + np.matmul(features_obs[:, 0:n_bias_features], c_params[0:n_bias_features])
        logit_quant = c_params[n_bias_features] * np.multiply(features_obs[:, n_bias_features], amount) - c_params[
            n_bias_features + 1] * np.multiply(features_obs[:, n_bias_features + 1], amount ** 2)
        probs_obs = sigmoid(logit_bias + logit_quant)
        y_obs = np.random.binomial(1, probs_obs)
        x_obs = np.hstack((features_obs, amount.reshape(-1, 1)))
        # x = torch.from_numpy(x_obs).float()
        # y = torch.from_numpy(y_obs).float()

        return x_obs, y_obs, probs_obs


    # train
    x_train, y_train, probs_obs_train = create_data(c_params, n_samples_train, n_features, n_bias_features)
    x_val, y_val, probs_obs_val = create_data(c_params, n_samples_train, n_features, n_bias_features)
    x_test, y_test, probs_obs_test = create_data(c_params, n_samples_test, n_features, n_bias_features)
    #bayesian_model = BayesianNeuralNetwork.cold_start(dim_list=[x_train.shape[1], 10], update_method="VI", update_kwargs={"trace":{"n": 100}})
    #bayesian_model = BayesianNeuralNetwork.cold_start(dim_list=[x_train.shape[1], 10], update_method="MCMC")
    
    n_features = x_train.shape[1]
   
    #### 
    n_samples = 1000
    n_features = 10
    bayesian_model =  BayesianLogisticRegression(alpha=StudentT(mu=1, sigma=2), betas=n_features * [StudentT()])
    x_train = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, n_features))
    y_train = np.random.binomial(1, 0.5, size=(n_samples,))
    bayesian_model.sample_proba(x_train)
    ###

    bayesian_model.update(x_train, y_train)
    #prob, _ = bayesian_model.sample_proba(x_train)

#     import torch
#     import matplotlib.pyplot as plt
#
#
#     def sigmoid(x):
#         return 1 / (1 + np.exp(-x))
#
#
#     # %%
#     def create_data(c_params, n_samples, n_features, n_bias_features):
#         features_obs = np.random.uniform(0, 1, size=(n_samples, n_features))
#         amount = np.random.uniform(0, 2, size=(n_samples))
#         logit_bias = c_params[-1] + np.matmul(features_obs[:, 0:n_bias_features], c_params[0:n_bias_features])
#         logit_quant = c_params[n_bias_features] * np.multiply(features_obs[:, n_bias_features], amount) - c_params[
#             n_bias_features + 1] * np.multiply(features_obs[:, n_bias_features + 1], amount ** 2)
#         probs_obs = sigmoid(logit_bias + logit_quant)
#         y_obs = np.random.binomial(1, probs_obs)
#         x_obs = np.hstack((features_obs, amount.reshape(-1, 1)))
#         x = torch.from_numpy(x_obs).float()
#         y = torch.from_numpy(y_obs).float()
#
#         return x, y, probs_obs
#
#
#     c_params = [-4, 2, 3, 3, 1, -3]
#     n_bias_features = 3
#     n_features = 5
#     n_samples_train = 30000
#     n_samples_val = 2000
#     n_samples_test = 10000
#
#     # train
#     x_train, y_train, probs_obs_train = create_data(c_params, n_samples_train, n_features, n_bias_features)
#     x_val, y_val, probs_obs_val = create_data(c_params, n_samples_train, n_features, n_bias_features)
#     x_test, y_test, probs_obs_test = create_data(c_params, n_samples_test, n_features, n_bias_features)
#
#     bayesian_model = BayesianNeuralNetwork(x_train.shape[1], 10)
#
#     num_iteration = 5
#     for iter in range(num_iteration):
#         trace = bayesian_model.sample(x_test)
#         probabilities = trace['prior_predictive']['out'].squeeze().mean(axis=0).values
#         eps = 1e-5
#         logloss = -(y_test * np.log(probabilities + eps)).mean() - (
#                 (1 - y_test) * np.log(1 - probabilities + eps)).mean()
#         print(f"logloss: {logloss}")
#         x_train, y_train, probs_obs_train = create_data(c_params, 1000, n_features, n_bias_features)
#         bayesian_model.update(x_train, y_train, num_iter=1000, learning_rate=0.01)
#
#         plt.figure()
#         plt.plot(bayesian_model.approx.hist)
#         plt.ylabel("ELBO")
#         plt.xlabel(f"iteration {iter}");
#         plt.show()
#
#         print(1)
#
# from pydantic import BaseModel, Field
#
#
# class User(BaseModel):
#     id: int
#     name: str
#     age2: int
#     age: int = Field(..., ge=0)  # Age must be a non-negative integer
#
#     def __init__(self, **data):
#         super().__init__(**data)
#         self.age2 = 2 * self.age
#
#
# user = User(id=1, name="Alice", age=30)
# print(user)
#
# #%%
# import warnings
# from abc import ABC, abstractmethod
# from random import betavariate
# from typing import Any, List, Literal, Optional, Tuple, Union, Dict
#
# import numpy as np
# import pymc.math as pmath
# from numpy import array, c_, insert, mean, multiply, ones, sqrt, std
# from numpy.typing import ArrayLike
# from pymc import Bernoulli, Data, Deterministic, fit, sample
# from pymc import Model as PymcModel
# from pymc import StudentT as PymcStudentT
# import pymc as pm
# from pytensor.tensor import TensorVariable, dot
# from scipy.stats import t
#
# from pybandits.base import BinaryReward, Probability, PyBanditsBaseModel
# from pybandits.pydantic_version_compatibility import (
#     PYDANTIC_VERSION_1,
#     PYDANTIC_VERSION_2,
#     Field,
#     NonNegativeFloat,
#     PositiveInt,
#     confloat,
#     model_validator,
#     pydantic_version,
#     validate_call,
# )
#
# UpdateMethods = Literal["MCMC", "VI"]
# class Model(PyBanditsBaseModel, ABC):
#     """
#     Class to _model the prior distributions.
#     """
#
#     @abstractmethod
#     def sample_proba(self) -> Probability:
#         """
#         Sample the probability of getting a positive reward.
#         """
#
#     @abstractmethod
#     def update(self, rewards: List[Any]):
#         """
#         Update the _model parameters.
#         """
#
#
#
######
# import pymc as pm
# import numpy as np
# with self.actions["a5"]._model:
#     trace = pm.sample_prior_predictive(samples=1000)

# for name in ["w0","b0"]:
#     print(name)
#     print(np.mean(trace["prior"][name].values.squeeze(), axis=0))
#     print(np.std(trace["prior"][name].values.squeeze(), axis=0))






#####

import warnings
from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import NDArray

RescalerType = Literal[
    "linear",
    "atan",
    "log1p",
    "log",
    "exp",
    "power",
    "sigmoid",
    "sinh",
]


class ReReTransform:
    """Transforms distribution so as to map the anchor point into mean."""

    def __init__(self, type: RescalerType | None = None, offset: float = 0, power: float = 1) -> None:
        """Creates a Rescale-Rebalance transform with the selected pre-defined rescaler.

        Args:
            type (RescalerType, optional): Which rescaler to use; if None, set to "linear". Defaults to None. Must be one of:
                - "linear"
                - "atan"
                - "log1p"
                - "log"
                - "exp"
                - "power"
                - "sigmoid"
                - "sinh"
            offset (float): Sets the argument that is mapped to 0 (if possible). Defaults to 0.
            power (float): What power to apply for type "power". Defaults to 1.
        """
        self._g = None
        self._g_inv = None
        self._u_is_vectorized = False
        self._gamma = None

        match type:
            case None | "linear":
                self._u = lambda x: x - offset
                self._u_inv = lambda x: x + offset
            case "atan":
                self._u = lambda x: np.arctan(x - offset)
                self._u_inv = lambda x: np.tan(x) + offset
            case "log1p":
                self._u = lambda x: np.log1p(x - offset)
                self._u_inv = lambda x: np.expm1(x) + offset
            case "log":
                self._u = lambda x: np.log(x)  # "no offset" option
                self._u_inv = lambda x: np.exp(x)
            case "exp":
                self._u = np.exp  # cannot shift argument to be mapped to 0
                self._u_inv = np.log
            case "power":
                assert power > 0, "Only positive powers are accepted"
                self._u = lambda x: np.sign(x - offset) * (np.abs(x - offset))**power
                self._u_inv = lambda x: np.sign(x) * (np.abs(x))**(1 / power) + offset
            case "sigmoid":
                self._u = lambda x: 2 / (1 + np.exp(offset - x)) - 1
                self._u_inv = lambda x: offset - np.log(2 / (1 + x) - 1)
            case "sinh":
                self._u = lambda x: np.sinh(x - offset)
                self._u_inv = lambda x: np.asinh(x) + offset
            case _:
                raise ValueError(f"No standard rescaler of type \"{type}\".")

    @classmethod
    def from_custom_function(cls,
                             rescaler: Callable,
                             inverse_rescaler: Callable,
                             input_upper_bound: float | None = None,
                             input_lower_bound: float | None = None,
                             input_sample: NDArray | None = None,
                             force_accept: bool = False) -> "ReReTransform":
        """Creates a transform from a fully defined custom function.

        Args:
            rescaler (Callable): Forward rescale function.
            inverse_rescaler (Callable): Inverse of the rescale function (verified).
            input_upper_bound (float, optional): If not None, sets the upper limit on input values. Defaults to None.
            input_lower_bound (float, optional): If not None, sets the lower limit on input values. Defaults to None.
            input_sample (NDArray, optional): If not None, is used to verify the validity of the provided function. Defaults to None.
            force_accept (bool, optional): Whether to bypass the validity checks. Defaults to False.

        Returns:
            ReReTransform: A generated transform instance.
        """
        new_obj = ReReTransform()

        assert inverse_rescaler is not None, "Inverse of rescale function is required"

        if input_sample is not None:
            rescaler = cls._make_vectorized(rescaler, input_sample)
            inverse_rescaler = cls._make_vectorized(inverse_rescaler, rescaler(input_sample))
            new_obj._u_is_vectorized = True

        if not force_accept:
            is_monotonic = new_obj._check_monotonicity(rescaler, input_upper_bound, input_lower_bound, input_sample)
            assert is_monotonic, "Rescaler must be strictly monotonic"

            funcs_are_inverse = new_obj._check_funcs_are_inverse(rescaler, inverse_rescaler, input_upper_bound,
                                                                 input_lower_bound, input_sample)
            assert funcs_are_inverse, "Inverse provided is not actually inverse"

        new_obj._u = rescaler
        new_obj._u_inv = inverse_rescaler

        return new_obj

    @staticmethod
    def _make_vectorized(u: Callable, y: NDArray):
        sample = y[:min(2, len(y))]
        try:
            u(sample)
            return u
        except TypeError:
            return np.vectorize(u)

    @staticmethod
    def _check_monotonicity(rescaler: Callable,
                            input_upper_bound: float | None = None,
                            input_lower_bound: float | None = None,
                            input_sample: NDArray | None = None) -> bool:
        if input_sample is not None:
            input_sample_gen = input_sample
        elif input_upper_bound is not None and input_lower_bound is not None:
            input_sample_gen = np.linspace(input_lower_bound, input_upper_bound, 10_000)
        else:
            raise NotImplementedError("Monotonity assertion with no domain specification will be introduced in future versions.")

        assert len(input_sample_gen) > 1, "Sample has to have at least 2 items"
        sorted_sample = np.unique(input_sample_gen)
        diff_val_sorted = np.diff(rescaler(sorted_sample))
        return np.all(diff_val_sorted > 0).item() or np.all(diff_val_sorted < 0).item()

    @staticmethod
    def _check_funcs_are_inverse(rescaler: Callable,
                                 inverse_rescaler: Callable,
                                 input_upper_bound: float | None = None,
                                 input_lower_bound: float | None = None,
                                 input_sample: NDArray | None = None) -> bool:
        if input_sample is not None:
            return np.allclose(inverse_rescaler(rescaler(input_sample)), input_sample)
        raise NotImplementedError("Invertibility assertion without data sample will be introduced in future versions.")

    def fit(self, y: NDArray, anchor: float, use_jackknife: bool = False, force: bool = False) -> None:
        """Fits the transformer to data sample.

        Args:
            y (NDArray): Sample to fit to.
            anchor (float): The transformation anchor (value that maps to mean).
            use_jackknife (bool, optional): Whether to use jackknife bias reduction. Defaults to False.
            force (bool, optional): Whether to bypass validty checks. Defaults to False.
        """
        if not force:
            assert np.min(y) < anchor and np.max(y) > anchor, "Anchor must be an internal point of the distribution"

        if not self._u_is_vectorized:
            self._u = self._make_vectorized(self._u, y)
            self._u_inv = self._make_vectorized(self._u_inv, self._u(y))
            self._u_is_vectorized = True

        u_anchor = self._u(anchor)
        u = self._u(y) - u_anchor  # force u(m)=0

        if not use_jackknife:
            gamma = np.mean(u) / np.mean(np.abs(u))
        else:
            warnings.warn("Current implementation of jackknife estimation is not numerically stable, which will be fixed in future versions.")
            n = len(y)
            gamma_n = np.mean(u) / np.mean(np.abs(u))
            gamma_less_i = (np.mean(u) - u / n) / (np.mean(np.abs(u)) - np.abs(u) / n)
            gamma = n * gamma_n - (n - 1) * np.mean(gamma_less_i)
        self._gamma = gamma

        def _g(y: NDArray) -> NDArray:
            u = self._u(y) - u_anchor
            return u - gamma * np.abs(u)

        def _g_inv(y: NDArray) -> NDArray:
            denom = 1 - np.sign(y) * gamma
            u = y / denom
            return self._u_inv(u + u_anchor)

        self._g = _g
        self._g_inv = _g_inv

    def transform(self, y: NDArray) -> NDArray:
        assert self._g is not None, "The transform has to be fitted first"
        return self._g(y)

    def fit_transform(self, y: NDArray, anchor: float, use_jackknife: bool = False) -> NDArray:
        self.fit(y, anchor, use_jackknife)
        return self.transform(y)

    def inverse_transform(self, y: NDArray) -> NDArray:
        assert self._g_inv is not None, "The transform has to be fitted first"
        return self._g_inv(y)

    def get_gamma(self) -> float:
        assert self._gamma is not None, "The transform has to be fit first"
        return self._gamma.item()

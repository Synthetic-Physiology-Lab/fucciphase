from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
from scipy import optimize


def logistic(
    x: Union[float, np.ndarray], center: float, sigma: float, sign: float = 1.0
) -> Union[float, np.ndarray]:
    """Logistic function."""
    tiny = 1.0e-15
    arg = sign * (x - center) / max(tiny, sigma)
    return 1.0 / (1.0 + np.exp(arg))


def accumulation_function(
    x: Union[float, np.ndarray],
    center: float,
    sigma: float,
    offset_intensity: float = 0,
) -> Union[float, np.ndarray]:
    """Function to describe accumulation of sensor."""
    return 1.0 - logistic(x, center, sigma) - offset_intensity


def degradation_function(
    x: Union[float, np.ndarray],
    center: float,
    sigma: float,
    offset_intensity: float = 0,
) -> Union[float, np.ndarray]:
    """Function to describe degradation of sensor."""
    return 1.0 - logistic(x, center, sigma, sign=-1.0) - offset_intensity


class FUCCISensor(ABC):
    """Base class for a FUCCI sensor."""

    @abstractmethod
    def __init__(
        self,
        channels: List[str],
        phase_percentages: List[float],
        center: List[float],
        sigma: List[float],
        thresholds: List[float],
    ) -> None:
        pass

    @property
    @abstractmethod
    def fluorophores(self) -> int:
        """Number of fluorophores."""
        pass

    @property
    @abstractmethod
    def phases(self) -> List[str]:
        """Function to hard-code the supported phases of a sensor."""
        pass

    @property
    def channels(self) -> List[str]:
        """Names of channels."""
        return self._channels

    @channels.setter
    def channels(self, value: List[str]) -> None:
        if len(value) != len(self.phases):
            raise ValueError("You need to provide one channel per phase.")
        self._channels = value

    @property
    def thresholds(self) -> List[float]:
        """Thresholds for assigning phases."""
        return self._thresholds

    @thresholds.setter
    def thresholds(self, value: List[float]) -> None:
        if len(value) != len(self.channels):
            raise ValueError("Provide one threshold per channel.")
        # check that the thresholds are between 0 and 1
        if not all(0 < t < 1 for t in value):
            raise ValueError("Thresholds must be between 0 and 1.")

        self._thresholds = value

    @property
    def phase_percentages(self) -> List[float]:
        """Percentage of individual phases."""
        return self._phase_percentages

    @phase_percentages.setter
    def phase_percentages(self, values: List[float]) -> None:
        if len(values) != len(self.phases):
            raise ValueError("Pass percentage for each phase.")

        # check that the sum of phase borders is less than 1
        if not np.isclose(sum(values), 1.0):
            raise ValueError("Phase borders do not sum to 1.")

        self._phase_percentages = values

    @abstractmethod
    def return_discrete_phase(self, phase_markers: List[bool]) -> str:
        """Get the discrete phase based on phase markers.

        Notes
        -----
        Discrete phase refers to, for example, G1 or S phase.
        The phase_markers must match the number of used fluorophores.
        """
        pass

    @abstractmethod
    def return_estimated_cycle_percentage(
        self, phase: str, intensities: List[float]
    ) -> float:
        """Estimate percentage based on sensor intensities."""
        pass

    def set_accumulation_and_degradation_parameters(
        self, center: List[float], sigma: List[float]
    ) -> None:
        """Pass list of functions for logistic functions.

        Parameters
        ----------
        center: List[float]
            List of center values for accumulation and degradation curves.
        sigma: List[float]
            List of width values for accumulation and degradation curves.
        """
        if len(center) != 2 * self.fluorophores:
            raise ValueError("Need to supply 2 center values per fluorophore.")
        if len(sigma) != 2 * self.fluorophores:
            raise ValueError("Need to supply 2 width values per fluorophore.")
        self._center_values = center
        self._sigma_values = sigma


class FUCCISASensor(FUCCISensor):
    """FUCCI(SA) sensor."""

    def __init__(
        self,
        channels: List[str],
        phase_percentages: List[float],
        center: List[float],
        sigma: List[float],
        thresholds: List[float],
    ) -> None:
        self.channels = channels
        self.phase_percentages = phase_percentages
        self.set_accumulation_and_degradation_parameters(center, sigma)

    @property
    def fluorophores(self) -> int:
        """Number of fluorophores."""
        return 2

    @property
    def phases(self) -> List[str]:
        """Function to hard-code the supported phases of a sensor."""
        return ["G1", "G1/S", "S/G2/M"]

    def return_discrete_phase(self, phase_markers: List[bool]) -> str:
        """Return the discrete phase based channel ON / OFF data for the
        FUCCI(SA) sensor.
        """
        if not len(phase_markers) == 2:
            raise ValueError(
                "The markers for G1 and S/G2/M channel have" "to be provided!"
            )
        g1_on = phase_markers[0]
        s_g2_on = phase_markers[1]
        # low intensity at the very beginning of cycle
        if not g1_on and not s_g2_on:
            return "G1"
        elif g1_on and not s_g2_on:
            return "G1"
        elif not g1_on and s_g2_on:
            return "S/G2/M"
        # G1/S transition phase
        else:
            return "G1/S"

    def _find_g1_percentage(self, intensity: float) -> float:
        """Find percentage in G1 phase."""
        g1_perc = self.phase_percentages[0]
        # intensity below expected minimal intensity
        if intensity < accumulation_function(
            0, self._center_values[0], self._sigma_values[0]
        ):
            return 0.0
        elif intensity > accumulation_function(
            g1_perc, self._center_values[0], self._sigma_values[0]
        ):
            return g1_perc
        return float(
            optimize.bisect(
                accumulation_function,
                0.0,
                g1_perc,
                args=(self._center_values[0], self._sigma_values[0], intensity),
            )
        )

    def _find_g1s_percentage(self, intensity: float) -> float:
        """Find percentage in G1/S phase."""
        g1_perc = self.phase_percentages[0]
        g1s_perc = self.phase_percentages[1]
        if intensity > degradation_function(
            g1_perc, self._center_values[1], self._sigma_values[1]
        ):
            return g1_perc
        elif intensity < degradation_function(
            g1_perc + g1s_perc, self._center_values[1], self._sigma_values[1]
        ):
            return g1_perc + g1s_perc
        return float(
            optimize.bisect(
                degradation_function,
                g1_perc,
                g1_perc + g1s_perc,
                args=(self._center_values[1], self._sigma_values[1], intensity),
            )
        )

    def _find_sg2m_percentage(self, intensity: float) -> float:
        """Find percentage in S/G2/M phase."""
        g1_perc = self.phase_percentages[0]
        g1s_perc = self.phase_percentages[1]
        # if intensity is very small, it is M phase
        if intensity < 0.5 * accumulation_function(
            g1_perc + g1s_perc, self._center_values[2], self._sigma_values[2]
        ):
            return 100.0
        elif intensity < accumulation_function(
            g1_perc + g1s_perc, self._center_values[2], self._sigma_values[2]
        ):
            return g1_perc + g1s_perc
        elif intensity > accumulation_function(
            100, self._center_values[2], self._sigma_values[2]
        ):
            return 100.0
        return float(
            optimize.bisect(
                accumulation_function,
                g1_perc + g1s_perc,
                100.0,
                args=(self._center_values[2], self._sigma_values[2], intensity),
            )
        )

    def return_estimated_cycle_percentage(
        self, phase: str, intensities: List[float]
    ) -> float:
        """Estimate a cell cycle percentage based on intensities.

        Parameters
        ----------
        phase: str
            Name of phase
        intensities: List[float]
            List of channel intensities for all fluorophores
        """
        if phase not in self.phases:
            raise ValueError(f"Phase {phase} is not defined for this sensor.")
        if phase == "G1":
            return self._find_g1_percentage(intensities[0])
        if phase == "G1/S":
            return self._find_g1s_percentage(intensities[0])
        else:
            return self._find_sg2m_percentage(intensities[1])

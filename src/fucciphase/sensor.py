from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
import pandas as pd
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
        phase_percentages: List[float],
        center: List[float],
        sigma: List[float],
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
    def phase_percentages(self) -> List[float]:
        """Percentage of individual phases."""
        return self._phase_percentages

    @phase_percentages.setter
    def phase_percentages(self, values: List[float]) -> None:
        if len(values) != len(self.phases):
            raise ValueError("Pass percentage for each phase.")

        # check that the sum of phase borders is less than 100
        if not np.isclose(sum(values), 100.0, atol=0.2):
            raise ValueError("Phase percentages do not sum to 100.")

        self._phase_percentages = values

    @abstractmethod
    def get_phase(self, phase_markers: Union[List[bool], "pd.Series[bool]"]) -> str:
        """Get the discrete phase based on phase markers.

        Notes
        -----
        Discrete phase refers to, for example, G1 or S phase.
        The phase_markers must match the number of used fluorophores.
        """
        pass

    @abstractmethod
    def get_estimated_cycle_percentage(
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

    @abstractmethod
    def get_expected_intensities(
        self, percentage: Union[float, np.ndarray]
    ) -> List[Union[float, np.ndarray]]:
        """Return value of calibrated curves."""
        pass


class FUCCISASensor(FUCCISensor):
    """FUCCI(SA) sensor."""

    def __init__(
        self, phase_percentages: List[float], center: List[float], sigma: List[float]
    ) -> None:
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

    def get_phase(self, phase_markers: Union[List[bool], "pd.Series[bool]"]) -> str:
        """Return the discrete phase based channel ON / OFF data for the
        FUCCI(SA) sensor.
        """
        if not len(phase_markers) == 2:
            raise ValueError(
                "The markers for G1 and S/G2/M channel haveto be provided!"
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
        """Find percentage in G1 phase.

        Parameters
        ----------
        intensity: float
            Intensity of cyan / green channel

        Notes
        -----
        Checks the accumulation function of the first colour.
        First colour means the colour indicating G1 phase.

        """
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
        """Find percentage in G1/S phase.

        Parameters
        ----------
        intensity: float
            Intensity of cyan / green channel

        Notes
        -----
        Checks the degradation function of the first colour.
        First colour means the colour indicating G1 phase.
        """
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
        """Find percentage in S/G2/M phase.

        Parameters
        ----------
        intensity: float
            Intensity of second colour (magenta / red)

        Notes
        -----
        Checks the accumulation function of the second colour.
        Second colour means the colour indicating S/G2/M phase.
        """
        g1_perc = self.phase_percentages[0]
        g1s_perc = self.phase_percentages[1]

        # check if intensity is below smallest expected intensity
        if intensity < accumulation_function(
            g1_perc + g1s_perc, self._center_values[2], self._sigma_values[2]
        ):
            return g1_perc + g1s_perc
        # if intensity is very small, it is M phase
        if intensity < 0.3 * accumulation_function(
            100.0, self._center_values[2], self._sigma_values[2]
        ):
            return 100.0
        # return middle of interval if values are close
        g1s_level = accumulation_function(
            g1_perc + g1s_perc, self._center_values[2], self._sigma_values[2]
        )
        final_level = accumulation_function(
            100.0, self._center_values[2], self._sigma_values[2]
        )

        if np.isclose(g1s_level, final_level):
            return g1s_perc + 0.5 * (100.0 - g1s_perc - g1_perc)
        try:
            if np.greater_equal(intensity, final_level):
                intensity = intensity - 2.0 * (intensity - final_level)  # type: ignore[assignment]
            return float(
                optimize.bisect(
                    accumulation_function,
                    g1_perc + g1s_perc,
                    100.0,
                    args=(self._center_values[2], self._sigma_values[2], intensity),
                )
            )
        except ValueError:
            print(
                "WARNING: could not infer percentage in SG2M phase, using average phase"
            )
            return g1s_perc + 0.5 * (100.0 - g1s_perc - g1_perc)

    def get_estimated_cycle_percentage(
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

    def get_expected_intensities(
        self, percentage: Union[float, np.ndarray]
    ) -> List[Union[float, np.ndarray]]:
        """Return value of calibrated curves."""
        g1_acc = accumulation_function(
            percentage, self._center_values[0], self._sigma_values[0]
        )
        g1_deg = degradation_function(
            percentage, self._center_values[1], self._sigma_values[1]
        )
        s_g2_m_acc = accumulation_function(
            percentage, self._center_values[2], self._sigma_values[2]
        )
        s_g2_m_deg = degradation_function(
            percentage, self._center_values[3], self._sigma_values[3]
        )
        return [g1_acc + g1_deg - 1.0, s_g2_m_acc + s_g2_m_deg - 1.0]


def get_fuccisa_default_sensor() -> FUCCISASensor:
    """Return sensor with default values.

    Should only be used if the cell cycle percentage is not of interest.
    """
    return FUCCISASensor(
        phase_percentages=[25, 25, 50], center=[0, 0, 0, 0], sigma=[0, 0, 0, 0]
    )


class PIPFUCCISensor(FUCCISensor):
    """PIP-FUCCI sensor."""

    def __init__(
        self, phase_percentages: List[float], center: List[float], sigma: List[float]
    ) -> None:
        self.phase_percentages = phase_percentages
        self.set_accumulation_and_degradation_parameters(center, sigma)

    @property
    def fluorophores(self) -> int:
        """Number of fluorophores."""
        return 2

    @property
    def phases(self) -> List[str]:
        """Function to hard-code the supported phases of a sensor."""
        return ["G1", "S", "G2/M"]

    def get_phase(self, phase_markers: Union[List[bool], "pd.Series[bool]"]) -> str:
        """Return the discrete phase based channel ON / OFF data for the
        FUCCI(SA) sensor.
        """
        if not len(phase_markers) == 2:
            raise ValueError(
                "The markers for G1 and S/G2/M channel haveto be provided!"
            )
        g1_on = phase_markers[0]
        s_on = phase_markers[1]
        # low intensity at the very beginning of cycle
        if not g1_on and not s_on:
            return "S"
        elif g1_on and not s_on:
            return "G1"
        elif not g1_on and s_on:
            return "S"
        else:
            return "G2/M"

    def _find_g1_percentage(self, intensity: float) -> float:
        """Find percentage in G1 phase.

        Parameters
        ----------
        intensity: float
            Intensity of cyan / green channel

        Notes
        -----
        Checks the accumulation function of the first colour.
        First colour means the colour indicating G1 phase.

        """
        raise NotImplementedError("Percentage estimate not yet implemented!")

    def _find_s_percentage(self, intensity: float) -> float:
        """Find percentage in S phase.

        Parameters
        ----------
        intensity: float
            Intensity of cyan / green channel

        Notes
        -----
        Checks the degradation function of the first colour.
        First colour means the colour indicating G1 phase.
        """
        raise NotImplementedError("Percentage estimate not yet implemented!")

    def _find_g2m_percentage(self, intensity: float) -> float:
        """Find percentage in G2/M phase.

        Parameters
        ----------
        intensity: float
            Intensity of second colour (magenta / red)

        Notes
        -----
        Checks the accumulation function of the second colour.
        Second colour means the colour indicating S/G2/M phase.
        """
        raise NotImplementedError("Percentage estimate not yet implemented!")

    def get_estimated_cycle_percentage(
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
        raise NotImplementedError("Percentage estimate not yet implemented!")
        if phase not in self.phases:
            raise ValueError(f"Phase {phase} is not defined for this sensor.")
        # TODO fill the following structure with life!
        if phase == "G1":
            return self._find_g1_percentage(intensities[0])
        if phase == "S":
            return self._find_s_percentage(intensities[0])
        else:
            return self._find_g2m_percentage(intensities[1])

    def get_expected_intensities(
        self, percentage: Union[float, np.ndarray]
    ) -> List[Union[float, np.ndarray]]:
        """Return value of calibrated curves."""
        raise NotImplementedError("Intensity estimate not yet implemented!")


def get_pipfucci_default_sensor() -> PIPFUCCISensor:
    """Return sensor with default values.

    Should only be used if the cell cycle percentage is not of interest.
    """
    # TODO update values
    return PIPFUCCISensor(
        phase_percentages=[25, 25, 50], center=[0, 0, 0, 0], sigma=[0, 0, 0, 0]
    )

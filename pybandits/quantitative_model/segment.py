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


import numpy as np
from pydantic import field_validator

from pybandits.base import Float01, ForbiddenRegion, PyBanditsBaseModel


class Segment(PyBanditsBaseModel):
    """
    This class is used to represent a segment of the quantities space.
    A segment is defined by a list of intervals, thus representing a hyper rectangle.

    Parameters
    ----------
    intervals: tuple[tuple[Float01, Float01], ...]
        Intervals of the segment.
    """

    intervals: tuple[tuple[Float01, Float01], ...]

    @property
    def mins(self) -> np.ndarray:
        return self.intervals_array[:, 0]

    @property
    def maxs(self) -> np.ndarray:
        return self.intervals_array[:, 1]

    @property
    def intervals_array(self) -> np.ndarray:
        array_form = np.array(self.intervals)
        if array_form.size == 0:
            return np.array([[], []]).T
        return np.array(self.intervals)

    @field_validator("intervals", mode="before")
    @classmethod
    def segment_intervals_to_tuple(cls, value):
        if isinstance(value, np.ndarray):
            if value.ndim != 2:
                raise ValueError("Intervals must be a 2-dimensional array of shape (n, 2).")
            if value.shape[1] != 2:
                raise ValueError("Intervals must have shape (n, 2).")
            return tuple(tuple(v) for v in value)
        return value

    def split(self) -> tuple["Segment", "Segment"]:
        middles = (self.mins + self.maxs) / 2
        left_intervals = np.concatenate([np.atleast_2d(self.mins).T, np.atleast_2d(middles).T], axis=1)
        right_intervals = np.concatenate([np.atleast_2d(middles).T, np.atleast_2d(self.maxs).T], axis=1)
        return Segment(intervals=left_intervals), Segment(intervals=right_intervals)

    def forbidden_region_outside(self) -> ForbiddenRegion:
        """
        Build a :data:`ForbiddenRegion` that forbids every quantity outside this segment's box.

        The segment is treated as the *allowed* hyper-rectangle: a quantity is allowed only where
        ``mins_i <= x_i <= maxs_i`` for every dimension, and forbidden anywhere outside it. The returned
        margin follows the ``ForbiddenRegion`` convention (``> 0`` => forbidden): it is the signed distance
        outside the box, ``max_i max(mins_i - x_i, x_i - maxs_i)`` -- positive when any coordinate falls
        outside its interval (forbidden) and ``<= 0`` when every coordinate lies within (allowed). The margin
        gives the optimizer directional information toward the allowed box.

        Returns
        -------
        ForbiddenRegion
            A signed-margin callable that forbids everything outside this segment.
        """
        mins = self.mins
        maxs = self.maxs

        def forbidden_region(x: np.ndarray) -> float:
            x = np.asarray(x, dtype=float)
            return float(np.max(np.maximum(mins - x, x - maxs)))

        return forbidden_region

    def __add__(self, other: "Segment") -> "Segment":
        """
        Add two adjacent segments.

        Parameters
        ----------
        other : Segment
            Segment to add.

        Returns
        -------
        Segment
            The merged segment.
        """
        if not self.is_adjacent(other):
            raise ValueError("Segments must be adjacent.")

        new_intervals = np.column_stack((np.minimum(self.mins, other.mins), np.maximum(self.maxs, other.maxs)))

        return Segment(intervals=new_intervals)

    def __hash__(self) -> int:
        return hash(tuple(tuple(interval) for interval in self.intervals))

    def __contains__(self, value: float | np.ndarray) -> bool:
        """
        Check if a value is contained in segment.

        Parameters
        ----------
        value : float | np.ndarray
            Value to check.

        Returns
        -------
        bool
            Whether the value is contained in the segment.
        """
        if (isinstance(value, np.ndarray) and value.shape[0] != self.intervals_array.shape[0]) or (
            isinstance(value, float) and len(self.intervals_array) != 1
        ):
            raise ValueError("Tested value must have the same shape as the intervals.")
        return bool(
            np.all(
                np.logical_and(
                    (self.mins <= value),
                    np.logical_or((value < self.maxs), np.logical_and(value == self.maxs, self.maxs == 1)),
                )
            )
        )

    def __eq__(self, other) -> bool:
        return np.all(self.intervals_array == other.intervals_array)

    def is_adjacent(self, other: "Segment") -> bool:
        """
            Check if two segments are adjacent. Segments are adjacent if they share a face,
        meaning they have identical intervals in all dimensions except one, where they touch.

        Parameters
        ----------
        other : Segment
            Segment to check for adjacency.

        Returns
        -------
        bool
            Whether the segments are adjacent.
        """

        if self.intervals_array.shape[0] != other.intervals_array.shape[0]:
            raise ValueError("Segments must have the same shape.")

        # Create a mask for dimensions where intervals differ between segments
        diff_mask = ~np.all(self.intervals_array == other.intervals_array, axis=1)
        # Count how many dimensions have different intervals
        n_differences = np.sum(diff_mask)

        # Check if the differing dimensions are adjacent
        if n_differences == 1:
            adjacent_mask = np.logical_or(
                (self.maxs[diff_mask] == other.mins[diff_mask]), (self.mins[diff_mask] == other.maxs[diff_mask])
            )
            # Segments are adjacent if exactly one dimension differs and it's adjacent
            return bool(adjacent_mask[0])
        else:
            return False

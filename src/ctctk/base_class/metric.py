# -*- coding: utf-8 -*-
# This code is adopted from https://github.com/yzhangcs/parser/blob/main/supar/utils/metric.py

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

class Metric(object):

    def __init__(self, reverse: Optional[bool] = None, eps: float = 1e-12) -> Metric:
        super().__init__()

        self.n = 0.0
        self.count = 0.0
        self.reverse = reverse
        self.eps = eps

    def __repr__(self):
        return ' '.join([f"{key}: {val:6.2%}" for key, val in self.values.items()])

    def __lt__(self, other: Metric) -> bool:
        if not hasattr(self, 'main_score'):
            return True
        if not hasattr(other, 'main_score'):
            return False
        return (self.main_score < other.main_score) if not self.reverse else (self.main_score > other.main_score)

    def __le__(self, other: Metric) -> bool:
        if not hasattr(self, 'main_score'):
            return True
        if not hasattr(other, 'main_score'):
            return False
        return (self.main_score <= other.main_score) if not self.reverse else (self.main_score >= other.main_score)

    def __gt__(self, other: Metric) -> bool:
        if not hasattr(self, 'main_score'):
            return False
        if not hasattr(other, 'main_score'):
            return True
        return (self.main_score > other.main_score) if not self.reverse else (self.main_score < other.main_score)

    def __ge__(self, other: Metric) -> bool:
        if not hasattr(self, 'main_score'):
            return False
        if not hasattr(other, 'main_score'):
            return True
        return (self.main_score >= other.main_score) if not self.reverse else (self.main_score <= other.main_score)

    def __add__(self, other: Metric) -> Metric:
        return other

    @property
    def main_score(self):
        raise AttributeError

    @property
    def values(self):
        raise AttributeError

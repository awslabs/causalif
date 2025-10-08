# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Causalif prompt templates"""

from typing import List

class CausalifPrompts:
    """Causalif prompt templates based on the paper"""
    
    @staticmethod
    def background_reminder(factors: List[str], domains: List[str]) -> str:
        return f"""As a scientific researcher in the domains of {', '.join(domains)}, you need to clarify the statistical relationship between some pairs of factors. You first need to get clear of the meanings of the factors in {factors}, which are from your domains, and clarify the interaction between each pair of those factors."""

    @staticmethod
    def association_context() -> str:
        return """The association relationship between two factors A and B can be associated or independent, and this association relationship can be clarified by the following principles:

        1. If A and B are statistically associated or correlated, they are associated, otherwise they are independent.
        2. The association relationship can be strongly clarified if there is statistical evidence supporting it.
        3. If there is no obvious statistical evidence supporting the association relationship between A and B, it can also be clarified if there is any evidence showing that A and B are likely to be associated or independent statistically.
        4. If there is no evidence to clarify the association relationship between A and B, then it is unknown."""

    @staticmethod
    def association_type_context() -> str:
        return """If two factors A and B are associated, they may be directly associated or indirectly associated with respect to a set of Given Third Factors, and it can be clarified by the following principle:

        1. The first principle is to try to find statistical evidence from the given knowledge to clarify the following association types. If you cannot find statistical evidence, at least find evidence that is likely to be able to statistically clarify the association type between A and B. If no obvious evidence can be found, the association type is unknown.
        2. If the evidence shows that any factors from the Given Third Factors mediate the association between A and B, then A and B are indirectly associated via these factors.
        3. If the evidence shows that by controlling any factors from the Given Third Factors, A and B are not associated any more, then A and B are associated indirectly.
        4. If the evidence shows that A and B are still associated even if we control any of the given third factors, then A and B are directly associated.
        5. If you think A and B are indirectly associated via any of the given third factors, it must be true that: (1) A and the third factors are directly associated; (2) B and the third factors are directly associated."""

    @staticmethod
    def causal_direction_context() -> str:
        return """If variable A is the cause of variable B, then the change of A's value causes a change of B's value, but not vice versa."""
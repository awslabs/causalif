# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core data structures and enums for Causalif"""

from enum import Enum

class AssociationResponse(str, Enum):
    ASSOCIATED = "ASSOCIATED"
    INDEPENDENT = "INDEPENDENT"
    UNKNOWN = "UNKNOWN"

class AssociationType(str, Enum):
    DIRECTLY_ASSOCIATED = "DIRECTLY_ASSOCIATED"
    INDIRECTLY_ASSOCIATED = "INDIRECTLY_ASSOCIATED"
    UNKNOWN = "UNKNOWN"

class CausalDirection(str, Enum):
    A_CAUSES_B = "A_CAUSES_B"
    B_CAUSES_A = "B_CAUSES_A"
    UNKNOWN = "UNKNOWN"

class KnowledgeBase:
    """Represents a knowledge base for Causalif"""
    def __init__(self, kb_type: str, content: str = None, source: str = None):
        self.kb_type = kb_type  # "BG" for background, "DOC" for document, "PC" for statistical
        self.content = content
        self.source = source
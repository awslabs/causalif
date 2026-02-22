# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core data structures and enums for CausalIF"""

from enum import Enum


# Keep all your existing enums and classes unchanged
class AssociationResponse(str, Enum):
    ASSOCIATED = "ASSOCIATED"
    INDEPENDENT = "INDEPENDENT"
    UNKNOWN = "UNKNOWN"


class KnowledgeBase:
    """Represents a knowledge base for CausalIF"""
    def __init__(self, kb_type: str, content: str = None, source: str = None):
        self.kb_type = kb_type  
        self.content = content
        self.source = source

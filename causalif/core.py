# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core data structures for CausalIF"""


class KnowledgeBase:
    """Represents a knowledge base for CausalIF"""
    def __init__(self, kb_type: str, content: str = None, source: str = None):
        self.kb_type = kb_type  
        self.content = content
        self.source = source

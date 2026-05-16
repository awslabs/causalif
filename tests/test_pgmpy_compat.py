# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for visualization edge labels, do-operator annotation, and pgmpy integration."""

import math
import numpy as np
import pandas as pd
import networkx as nx
import pytest


class TestVisualizationEdgeLabels:
    """Test that visualization correctly renders do-operator labels on edges."""

    def _make_engine_stub(self):
        """Create a minimal object that satisfies visualize_graph's engine interface."""
        class EngineStub:
            enable_causal_estimate = False
            max_degrees = 5

            def analyze_degrees_of_separation(self, graph, target):
                return {'factors_by_degree': {}, 'max_degree_found': 0}

        return EngineStub()

    def test_edge_label_prior_strength_only(self):
        """Without do-operator data, label should show prior_strength."""
        from causalif.visualization import visualize_graph

        engine = self._make_engine_stub()
        graph = nx.DiGraph()
        graph.add_edge('X', 'Y', prior_strength=0.8)

        fig = visualize_graph(engine, graph, title="Test", target_factor=None)
        assert fig is not None

        text_traces = [t for t in fig.data if t.mode == 'text']
        assert len(text_traces) == 1
        labels = text_traces[0].text
        assert any('0.80' in str(lbl) for lbl in labels)

    def test_edge_label_do_probability_positive(self):
        """With do-operator data, label should show P=value↑ for positive."""
        from causalif.visualization import visualize_graph

        engine = self._make_engine_stub()
        graph = nx.DiGraph()
        graph.add_edge('X', 'Y', prior_strength=0.5, do_probability=0.35, do_direction='positive')

        fig = visualize_graph(engine, graph, title="Test", target_factor=None)
        assert fig is not None

        text_traces = [t for t in fig.data if t.mode == 'text']
        labels = text_traces[0].text
        assert any('P=0.35' in str(lbl) for lbl in labels)
        assert any('↑' in str(lbl) for lbl in labels)

    def test_edge_label_do_probability_negative(self):
        """Negative direction should show down arrow."""
        from causalif.visualization import visualize_graph

        engine = self._make_engine_stub()
        graph = nx.DiGraph()
        graph.add_edge('A', 'B', prior_strength=0.5, do_probability=0.42, do_direction='negative')

        fig = visualize_graph(engine, graph, title="Test", target_factor=None)
        text_traces = [t for t in fig.data if t.mode == 'text']
        labels = text_traces[0].text
        assert any('P=0.42' in str(lbl) for lbl in labels)
        assert any('↓' in str(lbl) for lbl in labels)

    def test_edge_label_do_probability_neutral(self):
        """Neutral direction should show right arrow."""
        from causalif.visualization import visualize_graph

        engine = self._make_engine_stub()
        graph = nx.DiGraph()
        graph.add_edge('A', 'B', prior_strength=0.5, do_probability=0.01, do_direction='neutral')

        fig = visualize_graph(engine, graph, title="Test", target_factor=None)
        text_traces = [t for t in fig.data if t.mode == 'text']
        labels = text_traces[0].text
        assert any('→' in str(lbl) for lbl in labels)

    def test_edge_label_colors_by_direction(self):
        """Edge label colors should differ by direction."""
        from causalif.visualization import visualize_graph

        engine = self._make_engine_stub()
        graph = nx.DiGraph()
        graph.add_edge('A', 'B', prior_strength=0.5, do_probability=0.3, do_direction='positive')
        graph.add_edge('B', 'C', prior_strength=0.5, do_probability=0.4, do_direction='negative')

        fig = visualize_graph(engine, graph, title="Test", target_factor=None)
        text_traces = [t for t in fig.data if t.mode == 'text']
        assert len(text_traces) == 1
        colors = text_traces[0].textfont.color
        assert 'lime' in colors
        assert '#ff6666' in colors

    def test_undirected_edge_no_do_label(self):
        """Undirected (dashed) edges should not show do-operator labels."""
        from causalif.visualization import visualize_graph

        engine = self._make_engine_stub()
        graph = nx.DiGraph()
        graph.add_edge('A', 'B', prior_strength=0.5, undirected=True,
                       do_probability=0.5, do_direction='positive')

        fig = visualize_graph(engine, graph, title="Test", target_factor=None)
        text_traces = [t for t in fig.data if t.mode == 'text']
        labels = text_traces[0].text
        # Undirected edges should have empty labels
        assert all(lbl == '' for lbl in labels)

    def test_edge_width_uses_do_probability(self):
        """Edge width should be modulated by do_probability when available."""
        from causalif.visualization import visualize_graph

        engine = self._make_engine_stub()

        # Graph with only prior_strength
        graph1 = nx.DiGraph()
        graph1.add_edge('X', 'Y', prior_strength=0.5)

        # Graph with do_probability (higher value)
        graph2 = nx.DiGraph()
        graph2.add_edge('X', 'Y', prior_strength=0.5, do_probability=0.9, do_direction='positive')

        fig1 = visualize_graph(engine, graph1, title="Test1", target_factor=None)
        fig2 = visualize_graph(engine, graph2, title="Test2", target_factor=None)

        # Get line traces (edges)
        line_traces1 = [t for t in fig1.data if t.mode == 'lines']
        line_traces2 = [t for t in fig2.data if t.mode == 'lines']

        # Edge with do_probability=0.9 should be wider than prior_strength=0.5
        width1 = line_traces1[0].line.width
        width2 = line_traces2[0].line.width
        assert width2 > width1

    def test_hover_text_includes_do_info(self):
        """Hover text should include do-operator probability and direction."""
        from causalif.visualization import visualize_graph

        engine = self._make_engine_stub()
        graph = nx.DiGraph()
        graph.add_edge('X', 'Y', prior_strength=0.5, do_probability=0.33, do_direction='negative')

        fig = visualize_graph(engine, graph, title="Test", target_factor=None)

        # Find line traces with hover text
        line_traces = [t for t in fig.data if t.mode == 'lines' and t.hovertext]
        assert len(line_traces) > 0
        hover = line_traces[0].hovertext
        assert 'Do-Operator P (ATE): 0.3300' in hover
        assert 'Inversely related' in hover


class TestVisualizeResults:
    """Test visualize_causalif_results with do-operator edge data."""

    def test_do_data_preserved_in_result_visualization(self):
        """Edge attributes including do_probability should survive serialization."""
        from causalif.visualization import visualize_causalif_results

        result = {
            'success': True,
            'max_degrees_used': 3,
            'max_parallel_queries_used': 10,
            'target_factor': 'Y',
            'causal_graph': {
                'nodes': ['X', 'Y'],
                'edges': [('X', 'Y', {'prior_strength': 0.7, 'do_probability': 0.25, 'do_direction': 'negative'})],
            },
        }

        fig = visualize_causalif_results(result)
        assert fig is not None

        text_traces = [t for t in fig.data if t.mode == 'text']
        assert len(text_traces) == 1
        labels = text_traces[0].text
        assert any('P=0.25' in str(lbl) for lbl in labels)
        assert any('↓' in str(lbl) for lbl in labels)

    def test_old_format_edges_still_work(self):
        """Edges without attributes dict (old format) should still render."""
        from causalif.visualization import visualize_causalif_results

        result = {
            'success': True,
            'max_degrees_used': 3,
            'max_parallel_queries_used': 10,
            'target_factor': 'Y',
            'causal_graph': {
                'nodes': ['X', 'Y'],
                'edges': [('X', 'Y')],  # Old format: no attributes
            },
        }

        fig = visualize_causalif_results(result)
        assert fig is not None


class TestLegend:
    """Test that the legend includes do-operator information."""

    def test_legend_includes_do_operator_section(self):
        """Legend should include the do-operator edge label explanation."""
        from causalif.visualization import visualize_graph

        class EngineStub:
            enable_causal_estimate = False
            max_degrees = 3

            def analyze_degrees_of_separation(self, graph, target):
                return {'factors_by_degree': {1: ['X']}, 'max_degree_found': 1}

        engine = EngineStub()
        graph = nx.DiGraph()
        graph.add_edge('X', 'Y', prior_strength=0.5, do_probability=0.3, do_direction='positive')

        fig = visualize_graph(engine, graph, title="Test", target_factor='Y')
        assert fig is not None

        # Check annotations for legend text
        annotations = fig.layout.annotations
        legend_annotations = [a for a in annotations if hasattr(a, 'text') and 'Edge Labels' in (a.text or '')]
        assert len(legend_annotations) > 0
        legend_text = legend_annotations[0].text
        assert 'Do-Operator' in legend_text
        assert 'Directly related' in legend_text
        assert 'Inversely related' in legend_text

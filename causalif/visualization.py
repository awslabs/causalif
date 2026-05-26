# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visualization utilities for CausalIF"""

import logging
from typing import Dict, Union
import math
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def visualize_graph(engine, graph: Union[nx.Graph, nx.DiGraph], title: str = "Graph", target_factor: str = None) -> go.Figure:
    """Enhanced visualization with proper arrows for directed graphs, degree highlighting, Bayesian edge strengths, and causal inference
    
    Args:
        engine: CausalIFEngine instance (for accessing enable_causal_estimate, max_degrees, etc.)
        graph: Graph to visualize
        title: Title for the visualization
        target_factor: Optional target factor for degree-based coloring
        
    Returns:
        Plotly Figure object
    """
    if len(graph.nodes()) == 0:
        logger.info(f"No nodes to display in {title}")
        return None
    
    # Remove isolated nodes (nodes with no edges)
    isolated_nodes = list(nx.isolates(graph))
    if isolated_nodes:
        logger.info(f"Removing {len(isolated_nodes)} isolated nodes from visualization: {isolated_nodes}")
        graph = graph.copy()  # Don't modify original
        graph.remove_nodes_from(isolated_nodes)
        
        if len(graph.nodes()) == 0:
            logger.info(f"No connected nodes to display in {title}")
            return None

    # Get causal inference information if available
    causal_summary = None
    adjustment_sets = {}
    direct_causes = []
    direct_effects = []
    all_confounders = set()
    
    if engine.enable_causal_estimate and target_factor and isinstance(graph, nx.DiGraph):
        try:
            causal_summary = engine.get_causal_summary(target_factor, graph)
            direct_causes = causal_summary.get('direct_causes', [])
            direct_effects = causal_summary.get('direct_effects', [])
            adjustment_sets = causal_summary.get('adjustment_sets', {})
            
            # Collect all confounders (variables in any adjustment set)
            for adj_set in adjustment_sets.values():
                if adj_set:
                    all_confounders.update(adj_set)
            
            logger.info(f"[Causal Inference Visualization]")
            logger.info(f"  Target: {target_factor}")
            logger.info(f"  Direct causes: {direct_causes}")
            logger.info(f"  Direct effects: {direct_effects}")
            logger.info(f"  Confounders: {list(all_confounders)}")
        except Exception as e:
            logger.warning(f"Could not get causal summary: {e}")

    n_nodes = len(graph.nodes())
    # Scale layout spacing and iterations with node count so larger graphs spread out.
    # Use a larger k value to push nodes further apart and make edge labels readable.
    layout_k = max(5, 3.5 * math.sqrt(n_nodes))
    layout_iters = max(100, 30 * n_nodes)
    pos = nx.spring_layout(graph, seed=42, k=layout_k, iterations=layout_iters) if graph.edges() else {
        node: (i, 0) for i, node in enumerate(graph.nodes())
    }

    fig = go.Figure()

    degrees_map = {}
    if target_factor and target_factor in graph.nodes():
        degrees_analysis = engine.analyze_degrees_of_separation(graph, target_factor)
        for degree, factors in degrees_analysis['factors_by_degree'].items():
            for factor in factors:
                degrees_map[factor] = degree

    # Track undirected edges we've already drawn (to avoid drawing both directions)
    drawn_undirected_edges = set()
    
    # Track trace indices for dashed vs solid (for dropdown filter)
    dashed_trace_indices = []
    solid_trace_indices = []
    
    # Track edge labels separately for solid vs dashed edges
    solid_edge_labels_x = []
    solid_edge_labels_y = []
    solid_edge_labels_text = []
    dashed_edge_labels_x = []
    dashed_edge_labels_y = []
    dashed_edge_labels_text = []
    
    # Track which annotation indices belong to solid edges (arrows)
    solid_annotation_indices = []
    
    for edge in graph.edges():
        # Skip if this is the reverse direction of an undirected edge we've already drawn
        if isinstance(graph, nx.DiGraph):
            is_undirected = graph[edge[0]][edge[1]].get('undirected', False)
            if is_undirected:
                edge_pair = tuple(sorted([edge[0], edge[1]]))
                if edge_pair in drawn_undirected_edges:
                    continue  # Skip this direction, already drawn
                drawn_undirected_edges.add(edge_pair)
        
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        # Get causal strength from edge data
        edge_data = graph[edge[0]][edge[1]]
        is_undirected_edge = isinstance(graph, nx.DiGraph) and edge_data.get('undirected', False)
        strength = edge_data.get('prior_strength', 0) or 0
        do_probability = edge_data.get('do_probability', None)
        do_direction = edge_data.get('do_direction', None)
        
        # Store edge label position (midpoint, offset slightly for readability) and text
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        # Offset label perpendicular to edge direction for readability
        dx = x1 - x0
        dy = y1 - y0
        length = math.sqrt(dx * dx + dy * dy)
        if length > 0:
            # Perpendicular offset (rotated 90 degrees)
            offset_scale = 0.03
            mid_x += (-dy / length) * offset_scale
            mid_y += (dx / length) * offset_scale
        
        # Determine label text
        label_text = ""
        if do_probability is not None and not is_undirected_edge:
            direction_symbol = ""
            if do_direction == "positive":
                direction_symbol = "↑"
            elif do_direction == "negative":
                direction_symbol = "↓"
            elif do_direction == "neutral":
                direction_symbol = "→"
            elif do_direction == "shift":
                direction_symbol = "⇄"
            label_text = f"P={do_probability:.2f}{direction_symbol}"
        elif is_undirected_edge and do_probability is not None:
            # Dashed edge with ATE data (ATE=0 or failed) — show label
            label_text = f"P={do_probability:.2f}→"
        elif strength > 0 and not is_undirected_edge:
            label_text = f"{strength:.2f}"
        
        # Split labels into solid vs dashed buckets for dropdown filtering
        if is_undirected_edge:
            dashed_edge_labels_x.append(mid_x)
            dashed_edge_labels_y.append(mid_y)
            dashed_edge_labels_text.append(label_text)
        else:
            solid_edge_labels_x.append(mid_x)
            solid_edge_labels_y.append(mid_y)
            solid_edge_labels_text.append(label_text)
        
        edge_color = 'red'
        edge_width = 2
        
        # Undirected/dashed edges: fixed size and color
        if is_undirected_edge:
            edge_color = 'red'
            edge_width = 1.5
        # Color based on degrees from target (if target specified)
        elif target_factor:
            max_degree = max(degrees_map.get(edge[0], 0), degrees_map.get(edge[1], 0))
            if max_degree <= 1:
                edge_color = 'red'
                edge_width = 3
            elif max_degree <= 2:
                edge_color = 'orange'
                edge_width = 2.5
            elif max_degree <= 3:
                edge_color = 'green'
                edge_width = 2
            elif max_degree <= 4:
                edge_color = 'lightblue'
                edge_width = 1.5
            else:
                edge_color = 'lightgray'
                edge_width = 1
        
        # Modulate edge width by causal strength (only for directed edges)
        # Prefer do_probability (ATE) over prior_strength when available
        effective_strength = do_probability if (do_probability is not None and do_probability > 0) else strength
        if effective_strength > 0 and not is_undirected_edge:
            edge_width = edge_width * (0.5 + effective_strength)

        if isinstance(graph, nx.DiGraph):
            # Check if this edge is marked as undirected
            is_undirected = graph[edge[0]][edge[1]].get('undirected', False)
            
            dx, dy = x1 - x0, y1 - y0
            length = math.sqrt(dx*dx + dy*dy)
            if length > 0:
                dx_norm, dy_norm = dx / length, dy / length
                node_radius = 0.08
                x1_short = x1 - dx_norm * node_radius
                y1_short = y1 - dy_norm * node_radius
                x0_short = x0 + dx_norm * node_radius
                y0_short = y0 + dy_norm * node_radius

                # Create hover text with degree, strength, and adjustment set information
                if is_undirected:
                    hover_text = f"{edge[0]} ↔ {edge[1]} (undirected - no data)"
                    # Use dashed line for undirected edges
                    line_dash = 'dash'
                else:
                    hover_text = f"{edge[0]} → {edge[1]}"
                    if do_probability is not None and do_probability > 0:
                        hover_text += f"<br>Do-Operator P (ATE): {do_probability:.4f}"
                        if do_direction:
                            direction_label = {
                                'positive': 'Directly related (↑cause → ↑effect)',
                                'negative': 'Inversely related (↑cause → ↓effect)',
                                'neutral': 'Neutral (no significant shift)',
                            }.get(do_direction, do_direction)
                            hover_text += f"<br>Direction: {direction_label}"
                    if strength > 0:
                        hover_text += f"<br>Prior Strength: {strength:.3f}"
                    line_dash = 'solid'
                    
                    # Add adjustment set information if available
                    if edge[1] == target_factor and edge[0] in adjustment_sets:
                        adj_set = adjustment_sets[edge[0]]
                        if adj_set:
                            hover_text += f"<br><br>⚠️ <b>Adjustment Set:</b><br>Control for: {', '.join(adj_set)}"
                        else:
                            hover_text += f"<br><br>✓ <b>No adjustment needed</b>"
                
                if target_factor:
                    max_degree = max(degrees_map.get(edge[0], 0), degrees_map.get(edge[1], 0))
                    hover_text += f"<br>Max Degree from Target: {max_degree}"
                
                fig.add_trace(go.Scatter(
                    x=[x0_short, x1_short], y=[y0_short, y1_short], 
                    line=dict(width=edge_width, color=edge_color, dash=line_dash),
                    hoverinfo='text',
                    hovertext=hover_text,
                    mode='lines', 
                    showlegend=False
                ))
                # Track trace index for dropdown filter
                if is_undirected:
                    dashed_trace_indices.append(len(fig.data) - 1)
                else:
                    solid_trace_indices.append(len(fig.data) - 1)

                # Add arrow for ALL directed edges (not undirected)
                if not is_undirected:
                    arrow_x = x0_short + 0.95 * (x1_short - x0_short)
                    arrow_y = y0_short + 0.95 * (y1_short - y0_short)
                    arrow_end_x = arrow_x + 0.02 * dx_norm
                    arrow_end_y = arrow_y + 0.02 * dy_norm
                    
                    fig.add_annotation(
                        x=arrow_end_x, y=arrow_end_y,
                        ax=arrow_x - 0.02 * dx_norm, 
                        ay=arrow_y - 0.02 * dy_norm,
                        arrowhead=2, arrowsize=2, arrowwidth=2,
                        arrowcolor=edge_color, showarrow=True,
                        axref='x', ayref='y',
                        xref='x', yref='y'
                    )
                    solid_annotation_indices.append(len(fig.layout.annotations) - 1)
            else:
                # Fallback for zero-length edges (shouldn't happen but handle gracefully)
                fig.add_trace(go.Scatter(
                    x=[x0, x1], y=[y0, y1], 
                    line=dict(width=edge_width, color=edge_color),
                    hoverinfo='text',
                    hovertext=f"{edge[0]} → {edge[1]}",
                    mode='lines', 
                    showlegend=False
                ))
                
        else:
            # Undirected graph
            hover_text = f"{edge[0]} - {edge[1]}"
            if do_probability is not None and do_probability > 0:
                hover_text += f"<br>Do-Operator P (ATE): {do_probability:.4f}"
                if do_direction:
                    direction_label = {
                        'positive': 'Directly related (↑cause → ↑effect)',
                        'negative': 'Inversely related (↑cause → ↓effect)',
                        'neutral': 'Neutral (no significant shift)',
                    }.get(do_direction, do_direction)
                    hover_text += f"<br>Direction: {direction_label}"
            if strength > 0:
                hover_text += f"<br>Prior Strength: {strength:.3f}"
            if target_factor:
                max_degree = max(degrees_map.get(edge[0], 0), degrees_map.get(edge[1], 0))
                hover_text += f"<br>Max Degree from Target: {max_degree}"
                
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                line=dict(width=edge_width, color=edge_color),
                hoverinfo='text',
                hovertext=hover_text,
                mode='lines', 
                showlegend=False
            ))

    node_x, node_y, node_text, node_colors, node_hover = [], [], [], [], []
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x); node_y.append(y)
        node_text.append(str(node))

        # Determine node color based on causal role (if causal inference is enabled)
        if causal_summary and target_factor:
            if node == target_factor:
                node_colors.append(10)  # Highest value for target (red)
            elif node in direct_causes:
                node_colors.append(8)   # High value for direct causes (green)
            elif node in direct_effects:
                node_colors.append(6)   # Medium-high for direct effects (blue)
            elif node in all_confounders:
                node_colors.append(4)   # Medium for confounders (orange)
            else:
                node_colors.append(2)   # Low for other nodes (gray)
        elif target_factor and node in degrees_map:
            node_colors.append(degrees_map[node])
        else:
            degree = graph.degree(node) if not isinstance(graph, nx.DiGraph) else \
                     graph.in_degree(node) + graph.out_degree(node)
            node_colors.append(degree)
        
        # Build hover text with causal role information
        hover_parts = [f"<b>Node: {node}</b>"]
        
        if causal_summary and target_factor:
            if node == target_factor:
                hover_parts.append("🎯 <b>TARGET FACTOR</b>")
            elif node in direct_causes:
                hover_parts.append("🟢 <b>Direct Cause</b> (influences target)")
                if node in adjustment_sets and adjustment_sets[node]:
                    hover_parts.append(f"   Adjust for: {', '.join(adjustment_sets[node])}")
            elif node in direct_effects:
                hover_parts.append("🔵 <b>Direct Effect</b> (influenced by target)")
            elif node in all_confounders:
                hover_parts.append("🟠 <b>Confounder</b> (control variable)")
            else:
                hover_parts.append("⚪ Other Factor")
        
        if target_factor and node in degrees_map:
            hover_parts.append(f"Degree from {target_factor}: {degrees_map.get(node, 'N/A')}")
        
        hover_parts.append(f"Connections: {graph.degree(node)}")
        node_hover.append("<br>".join(hover_parts))

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        text=node_text, textposition="middle center",
        textfont=dict(size=max(8, 14 - n_nodes // 10), color='orange', family='Arial Black'),
        marker=dict(
            size=max(30, 55 - n_nodes), color=node_colors, 
            colorscale='RdYlGn_r' if causal_summary else ('Purples' if target_factor else 'Blues'),
            line=dict(width=2, color='white'),
            showscale=False
        ),
        hoverinfo='text', hovertext=node_hover
    )

    graph_type = "Directed" if isinstance(graph, nx.DiGraph) else "Undirected"
    degree_info = f" (Max {engine.max_degrees} degrees)" if (target_factor and engine.max_degrees is not None) else ""
    causal_info = " + Causal Inference" if causal_summary else ""
    fig.add_trace(node_trace)
    
    # Add edge strength labels — separate traces for solid vs dashed (for dropdown filtering)
    def _make_label_colors(texts):
        colors = []
        for text in texts:
            if '↑' in text:
                colors.append('lime')
            elif '↓' in text:
                colors.append('#ff6666')
            elif '→' in text:
                colors.append('yellow')
            elif text:
                colors.append('cyan')
            else:
                colors.append('cyan')
        return colors
    
    solid_labels_trace_idx = None
    if solid_edge_labels_x:
        solid_labels_trace_idx = len(fig.data)
        fig.add_trace(go.Scatter(
            x=solid_edge_labels_x, 
            y=solid_edge_labels_y,
            mode='text',
            text=solid_edge_labels_text,
            textposition="middle center",
            textfont=dict(size=10, color=_make_label_colors(solid_edge_labels_text), family='Arial Bold'),
            hoverinfo='skip',
            showlegend=False
        ))
    
    dashed_labels_trace_idx = None
    if dashed_edge_labels_x:
        dashed_labels_trace_idx = len(fig.data)
        fig.add_trace(go.Scatter(
            x=dashed_edge_labels_x, 
            y=dashed_edge_labels_y,
            mode='text',
            text=dashed_edge_labels_text,
            textposition="middle center",
            textfont=dict(size=10, color=_make_label_colors(dashed_edge_labels_text), family='Arial Bold'),
            hoverinfo='skip',
            showlegend=False
        ))
    
    # Add legend for edge colors and causal inference roles
    if target_factor:
        legend_text = "<b>Edge Color by Distance from Target:</b><br>" + \
                     "<span style='color:red'>━━━</span> 1 degree (direct)<br>" + \
                     "<span style='color:orange'>━━━</span> 2 degrees<br>" + \
                     "<span style='color:green'>━━━</span> 3 degrees<br>" + \
                     "<span style='color:lightblue'>━━━</span> 4 degrees<br>" + \
                     "<span style='color:lightgray'>━━━</span> 5+ degrees<br><br>" + \
                     "<b>Edge Type:</b><br>" + \
                     "<span style='color:white'>━━━→</span> Directed (Bayesian-inferred)<br>" + \
                     "<span style='color:white'>┄┄┄</span> Undirected (Associated)<br><br>" + \
                     "<b>Edge Labels (Do-Operator):</b><br>" + \
                     "P=value↑ Directly related<br>" + \
                     "P=value↓ Inversely related<br>" + \
                     "P=value→ Neutral"
        
        # Add causal inference legend if enabled
        if causal_summary:
            legend_text += "<br><br><b>Causal Inference (Node Colors):</b><br>" + \
                          "🎯 <span style='color:#ff4444'>Target Factor</span><br>" + \
                          "🟢 <span style='color:#44ff44'>Direct Cause</span> (influences target)<br>" + \
                          "🔵 <span style='color:#4444ff'>Direct Effect</span> (influenced by target)<br>" + \
                          "🟠 <span style='color:#ffaa44'>Confounder</span> (control variable)<br>" + \
                          "⚪ <span style='color:#aaaaaa'>Other Factor</span><br><br>" + \
                          "<i>Hover over edges to see adjustment sets</i>"
        
        fig.add_annotation(
            xref="paper", yref="paper",
            x=1.02, y=0.98,
            text=legend_text,
            showarrow=False,
            font=dict(size=10, color='white', family='Arial'),
            bgcolor='rgba(0,0,0,0.7)',
            borderpad=4,
            align='left',
            xanchor='left',
            yanchor='top'
        )
    
    # Scale figure size with node count — more nodes need more room
    fig_width = max(1100, 600 + 70 * n_nodes)
    fig_height = max(700, 400 + 45 * n_nodes)

    # Build dropdown filter for dashed/solid edges (including labels and arrows)
    total_traces = len(fig.data)
    dropdown_buttons = []
    if dashed_trace_indices:
        # Collect all solid-related trace indices (edge lines + labels)
        all_solid_traces = solid_trace_indices[:]
        if solid_labels_trace_idx is not None:
            all_solid_traces.append(solid_labels_trace_idx)
        
        # Collect all dashed-related trace indices (edge lines + labels)
        all_dashed_traces = dashed_trace_indices[:]
        if dashed_labels_trace_idx is not None:
            all_dashed_traces.append(dashed_labels_trace_idx)

        # "All Edges" — show everything, all arrows visible
        all_visible = [True] * total_traces
        all_annotations = list(fig.layout.annotations)
        dropdown_buttons.append(dict(
            label="All Edges",
            method="update",
            args=[
                {"visible": all_visible},
                {"annotations": all_annotations}
            ]
        ))
        
        # "Verified Only" — hide dashed edges and their labels, keep arrows
        verified_visible = [True] * total_traces
        for idx in all_dashed_traces:
            verified_visible[idx] = False
        dropdown_buttons.append(dict(
            label="Verified Only (hide dashed)",
            method="update",
            args=[
                {"visible": verified_visible},
                {"annotations": all_annotations}
            ]
        ))
        
        # "Unverified Only" — hide solid edges, their labels, AND arrows
        unverified_visible = [True] * total_traces
        for idx in all_solid_traces:
            unverified_visible[idx] = False
        # Build annotations list with arrow annotations hidden (keep legend annotation)
        unverified_annotations = []
        for i, ann in enumerate(all_annotations):
            if i in solid_annotation_indices:
                # Hide this arrow annotation
                ann_dict = ann.to_plotly_json()
                ann_dict['showarrow'] = False
                ann_dict['visible'] = False
                unverified_annotations.append(ann_dict)
            else:
                unverified_annotations.append(ann)
        dropdown_buttons.append(dict(
            label="Unverified Only (dashed)",
            method="update",
            args=[
                {"visible": unverified_visible},
                {"annotations": unverified_annotations}
            ]
        ))

    fig.update_layout(
        title=dict(
            text=f"{title} ({graph_type}){degree_info}{causal_info} - {len(graph.nodes())} nodes, {len(graph.edges())} edges - Bayesian", 
            x=0.5, font=dict(size=16, color='white')
        ),
        showlegend=False, hovermode='closest',
        margin=dict(b=20, l=5, r=200, t=80),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=fig_width, height=fig_height, 
        plot_bgcolor='black',
        paper_bgcolor='black',
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            x=0.0, y=1.12,
            xanchor="left", yanchor="top",
            bgcolor="rgba(50,50,50,0.8)",
            font=dict(color="white", size=11),
            buttons=dropdown_buttons,
            showactive=True,
        )] if dropdown_buttons else [],
    )

    return fig


def visualize_causalif_results(causalif_result: Dict) -> go.Figure:
    """Create visualization from CausalIF results with degree-based coloring"""
    from .engine import CausalIFEngine  # Import here to avoid circular dependency

    if not causalif_result['success']:
        logger.warning("Cannot visualize failed CausalIF analysis")
        return None

    max_degrees = causalif_result.get('max_degrees_used', 5)
    max_parallel_queries = causalif_result.get('max_parallel_queries_used', 50)

    # Build a minimal dataframe from the causal graph nodes so the engine
    # has a valid (non-hardcoded) dataframe.  Only the column names and
    # max_degrees / max_parallel_queries are used by the visualization path.
    graph_nodes = causalif_result['causal_graph'].get('nodes', [])
    if graph_nodes:
        dummy_df = pd.DataFrame({col: [0] for col in graph_nodes})
    else:
        dummy_df = pd.DataFrame({'_placeholder': [0]})

    viz_engine = CausalIFEngine(
        model=None, dataframe=dummy_df,
        max_degrees=max_degrees, max_parallel_queries=max_parallel_queries,
    )

    causal_graph = nx.DiGraph()
    causal_graph.add_nodes_from(causalif_result['causal_graph']['nodes'])
    # Add edges with attributes (including 'strength')
    for edge_data in causalif_result['causal_graph']['edges']:
        if len(edge_data) == 3:  # (source, target, attributes)
            causal_graph.add_edge(edge_data[0], edge_data[1], **edge_data[2])
        else:  # (source, target) - fallback for old format
            causal_graph.add_edge(edge_data[0], edge_data[1])

    target_factor = causalif_result.get('target_factor')
    method_status = "Bayesian Inference"
    return visualize_graph(viz_engine, causal_graph, f"CausalIF Results ({method_status}, Max {max_degrees} degrees)", target_factor)

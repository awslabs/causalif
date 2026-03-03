# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt templates for CausalIF"""

from typing import List, Dict


class CausalIFPrompts:
    """CausalIF prompt templates based on the paper"""
    
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
        """
        Return context explaining direct vs indirect association for LACR 1 algorithm.
        
        This context is used in Step 3 (Association Type Verifier) of the 4-step
        constraint-based causal (CC) prompt strategy from the paper.
        
        Reference: Section 3.2.1 of "Causal Graph Discovery with Retrieval-Augmented 
        Generation based Large Language Models" (https://arxiv.org/html/2402.15301v2)
        
        Returns:
            str: Context text explaining direct association, indirect association, and unknown
        """
        return """The association relationship between two factors A and B can be direct or indirect:

1. Direct Association: A and B are associated, and this association cannot be eliminated by controlling for (conditioning on) any other variables in the given factor set. In causal terms, A and B cannot be d-separated by any subset of the other variables.

2. Indirect Association: A and B are associated, but this association can be eliminated by controlling for (conditioning on) some intermediary variables. These intermediary variables mediate the relationship between A and B.

3. If you cannot determine whether the association is direct or indirect based on available evidence, it is unknown."""
    
    @staticmethod
    def association_type_verifier(
        factor_a: str,
        factor_b: str,
        factors: List[str],
        domains: List[str],
        document_content: str
    ) -> str:
        """
        Generate prompt for Step 3 (Association Type Verifier) with document.
        
        This method implements Step 3 of the 4-step constraint-based causal (CC) prompt
        strategy from the paper. It determines whether the association between two factors
        is direct or indirect based on document evidence.
        
        Reference: Section 3.2.1, Algorithm 1 (LACR 1) of "Causal Graph Discovery with 
        Retrieval-Augmented Generation based Large Language Models" 
        (https://arxiv.org/html/2402.15301v2)
        
        Args:
            factor_a: First factor in the pair being analyzed
            factor_b: Second factor in the pair being analyzed
            factors: Complete list of all factors in the variable set V
            domains: List of domain names for context
            document_content: Retrieved document content for evidence
            
        Returns:
            str: Formatted prompt for association type verification with document
            
        Expected Response Format:
            Thoughts: [Analysis of the question]
            Answer: (D) Directly Associated (E) Indirectly Associated (C) Unknown
            Intermediary Factors: [List if (E), otherwise "None"]
            Reference: [Supporting sentence from document, skip if (C)]
        """
        # Get background reminder
        background = CausalIFPrompts.background_reminder(factors, domains)
        
        # Get association type context
        context = CausalIFPrompts.association_type_context()
        
        # Build list of available factors for mediation (exclude factor_a and factor_b)
        available_factors = [f for f in factors if f not in [factor_a, factor_b]]
        available_factors_str = ', '.join(available_factors) if available_factors else "None"
        
        # Build the prompt
        prompt = f"""{background}

Your task is to thoroughly read the given 'Document'. Then, based on the knowledge from the given 'Document', determine whether the association between the pair of 'Main factors' is direct or indirect according to the 'Association Type Context'.

Document: {document_content}

Main factors: {factor_a} and {factor_b}

Available factors for mediation: {available_factors_str}

Association Type Context:
{context}

Association Type Question: Are {factor_a} and {factor_b} directly associated or indirectly associated?

Expected Response Format:
Thoughts: [Write your thoughts on the question]
Answer: (D) Directly Associated (E) Indirectly Associated (C) Unknown
Intermediary Factors: [If you chose (E), list the intermediary factors that mediate the relationship. Otherwise, write "None"]
Reference: [Skip this if you chose option C above. Otherwise, provide a supporting sentence from the document for your choice]"""
        
        return prompt
    
    @staticmethod
    def association_type_verifier_bg(
        factor_a: str,
        factor_b: str,
        factors: List[str],
        domains: List[str]
    ) -> str:
        """
        Generate prompt for Step 3 (Association Type Verifier) with background knowledge.
        
        This method implements Step 3 of the 4-step constraint-based causal (CC) prompt
        strategy from the paper. It determines whether the association between two factors
        is direct or indirect based on the LLM's background knowledge (training data).
        
        Reference: Section 3.2.1, Algorithm 1 (LACR 1) of "Causal Graph Discovery with 
        Retrieval-Augmented Generation based Large Language Models" 
        (https://arxiv.org/html/2402.15301v2)
        
        Args:
            factor_a: First factor in the pair being analyzed
            factor_b: Second factor in the pair being analyzed
            factors: Complete list of all factors in the variable set V
            domains: List of domain names for context
            
        Returns:
            str: Formatted prompt for association type verification with background knowledge
            
        Expected Response Format:
            Thoughts: [Analysis of the question]
            Answer: (D) Directly Associated (E) Indirectly Associated (C) Unknown
            Intermediary Factors: [List if (E), otherwise "None"]
        """
        # Get background reminder
        background = CausalIFPrompts.background_reminder(factors, domains)
        
        # Get association type context
        context = CausalIFPrompts.association_type_context()
        
        # Build list of available factors for mediation (exclude factor_a and factor_b)
        available_factors = [f for f in factors if f not in [factor_a, factor_b]]
        available_factors_str = ', '.join(available_factors) if available_factors else "None"
        
        # Build the prompt
        prompt = f"""{background}

Your task is to thoroughly use the knowledge in your training data to solve a task. Your task is: based on your background knowledge, determine whether the association between the pair of 'Main factors' is direct or indirect according to the 'Association Type Context'.

Main factors: {factor_a} and {factor_b}

Available factors for mediation: {available_factors_str}

Association Type Context:
{context}

Association Type Question: Are {factor_a} and {factor_b} directly associated or indirectly associated?

Expected Response Format:
Thoughts: [Write your thoughts on the question]
Answer: (D) Directly Associated (E) Indirectly Associated (C) Unknown
Intermediary Factors: [If you chose (E), list the intermediary factors that mediate the relationship. Otherwise, write "None"]"""
        
        return prompt

    
    @staticmethod
    def interpretation_prompt(
        original_query: str,
        target_factor: str,
        graph_description: str,
        influences_text: str,
        stats_text: str
    ) -> str:
        """
        Generate prompt for LLM interpretation of causal graph results.
        
        Args:
            original_query: The original user query
            target_factor: The target factor being analyzed
            graph_description: Formatted description of the causal graph
            influences_text: Text describing strongest influences
            stats_text: Text describing network statistics
            
        Returns:
            Formatted prompt for LLM interpretation
        """
        return f"""You are a causal inference expert analyzing data. Your task is to interpret a causal graph and provide actionable insights.

                ORIGINAL QUESTION:
                {original_query}

                TARGET FACTOR BEING ANALYZED:
                {target_factor}

                CAUSAL GRAPH ANALYSIS:
                {graph_description}
                {influences_text}
                {stats_text}

                METHODOLOGY:
                This analysis used:
                1. CausalIF 1: LLM + RAG voting to identify associated variables (PRIOR)
                2. CausalIF 2: Bayesian structure learning to determine causal directions (POSTERIOR)
                3. LLM-based prior strength estimation for causal edges

                TASK:
                Based on the causal graph above, provide a clear, actionable analysis that:

                1. **ROOT CAUSES (Direct - Degree 1)**: Identify the primary direct causes affecting {target_factor}. Explain which factors have the strongest causal impact and why. Focus on degree 1 and 2 factors that directly influence the target.

                2. **INDIRECT FACTORS (Degree 2+)**: Analyze the indirect factors organized by degree of separation. For each degree level:
                   - Identify the key factors at that degree
                   - Explain how they indirectly influence {target_factor} through intermediate factors
                   - Highlight the most important causal chains (e.g., Factor A → Factor B → {target_factor})
                   - Assess whether these indirect effects are significant enough to warrant attention

                3. **CAUSAL MECHANISMS**: Explain the mechanisms by which both direct and indirect factors affect {target_factor}. Describe the causal pathways and how changes propagate through the network.

                4. **ACTIONABLE INSIGHTS**: What are the key takeaways? Which factors should be monitored or controlled? Prioritize based on:
                   - Causal strength (how strong is the effect?)
                   - Degree of separation (how direct is the influence?)
                   - Practical controllability (can we actually intervene on this factor?)

                5. **RECOMMENDED INTERVENTIONS**: Suggest specific interventions or actions that could be taken to influence {target_factor}. Consider:
                   - Direct interventions on degree 1 factors (immediate impact)
                   - Indirect interventions on upstream factors (longer-term, systemic changes)
                   - Which approach is more feasible and effective?

                6. **IMPORTANT CONSIDERATIONS**: 
                   - Highlight any important multi-step causal chains that might not be obvious
                   - Identify potential unintended consequences of interventions
                   - Note any factors that appear at multiple degrees or in multiple pathways
                   - Discuss the trade-offs between intervening on direct vs. indirect factors

                Provide a structured, professional analysis that a business stakeholder can understand and act upon. Use the causal strengths and degrees of separation to prioritize your recommendations. Pay special attention to the indirect factors at each degree level.

                FORMAT YOUR RESPONSE AS:
                ## Root Causes (Direct Factors - Degree 1)
                [Your analysis of direct causes]

                ## Indirect Factors by Degree
                ### Degree 2 Factors
                [Analysis of factors 2 steps away]

                ### Degree 3+ Factors
                [Analysis of factors 3+ steps away, if present]

                ## Causal Mechanisms  
                [Your analysis of how factors influence the target]

                ## Key Insights
                [Your prioritized insights]

                ## Recommended Interventions
                [Your specific, actionable recommendations]

                ## Important Considerations
                [Caveats, trade-offs, and additional context]
                """


def format_causal_graph_for_llm(causal_relationships: List[Dict], target_factor: str) -> str:
    """
    Format causal graph structure into readable text for LLM consumption.
    
    Args:
        causal_relationships: List of causal edges with metadata
        target_factor: The target factor being analyzed
        
    Returns:
        Formatted string describing the causal graph
    """
    if not causal_relationships:
        return "No causal relationships found in the analysis."
    
    # Separate direct and indirect causes
    direct_causes = []
    indirect_causes_by_degree = {}  # Group by degree of separation
    other_relationships = []
    
    for rel in causal_relationships:
        cause = rel.get('cause', '')
        effect = rel.get('effect', '')
        degree = rel.get('degree_from_target', 999)
        
        # Check if this edge directly affects the target
        if effect == target_factor:
            direct_causes.append(rel)
        elif cause == target_factor:
            # Reverse relationship (target causes something else)
            other_relationships.append(rel)
        else:
            # Indirect relationship - group by degree
            if degree not in indirect_causes_by_degree:
                indirect_causes_by_degree[degree] = []
            indirect_causes_by_degree[degree].append(rel)
    
    # Build formatted output
    output = []
    
    # Section 1: Direct causes (degree 1)
    if direct_causes:
        output.append("=" * 80)
        output.append("DIRECT CAUSES OF " + target_factor.upper() + " (Degree 1)")
        output.append("=" * 80)
        for i, rel in enumerate(sorted(direct_causes, key=lambda x: x.get('causal_strength', 0) or 0, reverse=True), 1):
            strength = rel.get('causal_strength', 0) or 0
            strength_label = "very strong" if strength > 0.7 else "strong" if strength > 0.5 else "moderate" if strength > 0.3 else "weak"
            
            output.append(f"\n{i}. {rel['cause']} → {rel['effect']}")
            output.append(f"   • Causal strength: {strength:.2f} ({strength_label})")
            if 'evidence' in rel and rel['evidence']:
                output.append(f"   • Evidence: {rel['evidence']}")
    
    # Section 2: Indirect causes organized by degree
    if indirect_causes_by_degree:
        output.append("\n\n" + "=" * 80)
        output.append("INDIRECT CAUSAL FACTORS (by Degree of Separation)")
        output.append("=" * 80)
        
        # Sort degrees and process each
        for degree in sorted(indirect_causes_by_degree.keys()):
            if degree == 999:  # Skip unknown degrees
                continue
                
            rels_at_degree = indirect_causes_by_degree[degree]
            output.append(f"\n--- DEGREE {degree} FACTORS ({len(rels_at_degree)} relationships) ---")
            
            # Find unique factors at this degree that influence the target
            factors_influencing_target = set()
            for rel in rels_at_degree:
                # Check if this factor is on a path to the target
                if 'path_to_target' in rel and rel['path_to_target']:
                    path = rel['path_to_target']
                    # The first element in the path is the factor
                    if len(path) > 0:
                        factors_influencing_target.add(path[0])
            
            if factors_influencing_target:
                output.append(f"\nFactors at degree {degree} influencing {target_factor}:")
                for factor in sorted(factors_influencing_target):
                    output.append(f"  • {factor}")
            
            # Show key relationships at this degree
            output.append(f"\nKey causal relationships at degree {degree}:")
            sorted_rels = sorted(rels_at_degree, key=lambda x: x.get('causal_strength', 0) or 0, reverse=True)
            for i, rel in enumerate(sorted_rels[:10], 1):  # Top 10 per degree
                strength = rel.get('causal_strength', 0) or 0
                output.append(f"\n  {i}. {rel['cause']} → {rel['effect']}")
                output.append(f"     - Strength: {strength:.2f}")
                
                if 'path_to_target' in rel and rel['path_to_target']:
                    path_str = " → ".join(rel['path_to_target'])
                    output.append(f"     - Path to target: {path_str}")
    
    # Section 3: Effects of target factor
    if other_relationships:
        output.append("\n\n" + "=" * 80)
        output.append(f"EFFECTS OF {target_factor.upper()} (What it influences)")
        output.append("=" * 80)
        sorted_effects = sorted(other_relationships, key=lambda x: x.get('causal_strength', 0) or 0, reverse=True)
        for i, rel in enumerate(sorted_effects[:10], 1):  # Top 10 effects
            strength = rel.get('causal_strength', 0) or 0
            output.append(f"\n{i}. {rel['cause']} → {rel['effect']}")
            output.append(f"   • Strength: {strength:.2f}")
    
    return "\n".join(output)


def generate_llm_interpretation(causalif_result: Dict, original_query: str, model) -> str:
    """
    Use LLM to interpret the causal graph and answer the original question.
    
    Args:
        causalif_result: Output from causalif() function
        original_query: Original user query
        model: LLM model instance
        
    Returns:
        Natural language interpretation of the causal analysis
    """
    if not causalif_result.get('success', False):
        return "Could not generate interpretation due to CausalIF analysis failure."
    
    if model is None:
        return "LLM model not available for interpretation. Please configure the model using set_causalif_engine()."
    
    # Extract key information
    target_factor = causalif_result.get('target_factor', 'unknown')
    causal_relationships = causalif_result.get('causal_relationships', [])
    strongest_influences = causalif_result.get('strongest_causal_influences', [])
    network_summary = causalif_result.get('network_summary', {})
    causal_inference_summary = causalif_result.get('causal_inference_summary')  # NEW
    
    # Format causal graph for LLM
    graph_description = format_causal_graph_for_llm(causal_relationships, target_factor)
    
    # Build context about strongest influences
    influences_text = ""
    if strongest_influences:
        influences_text = "\n\nSTRONGEST CAUSAL INFLUENCES ON " + target_factor.upper() + ":\n"
        for i, inf in enumerate(strongest_influences[:5], 1):
            factor_name = inf.get('influencing_factor', 'unknown')
            strength = inf.get('causal_strength', 0) or 0
            degree = inf.get('degree', 'N/A')
            influences_text += f"{i}. {factor_name}: strength {strength:.2f}, degree {degree}\n"
    
    # NEW: Build causal inference context (adjustment sets, confounders)
    causal_inference_text = ""
    if causal_inference_summary:
        causal_inference_text = "\n\n" + "=" * 80 + "\n"
        causal_inference_text += "CAUSAL INFERENCE ANALYSIS (Adjustment Sets & Confounders)\n"
        causal_inference_text += "=" * 80 + "\n"
        
        direct_causes = causal_inference_summary.get('direct_causes', [])
        direct_effects = causal_inference_summary.get('direct_effects', [])
        adjustment_sets = causal_inference_summary.get('adjustment_sets', {})
        
        if direct_causes:
            causal_inference_text += f"\nDirect Causes of {target_factor}:\n"
            for cause in direct_causes:
                adj_set = adjustment_sets.get(cause, [])
                if adj_set:
                    causal_inference_text += f"  • {cause} → {target_factor}\n"
                    causal_inference_text += f"    ⚠️  CONFOUNDING DETECTED: Must control for {', '.join(adj_set)}\n"
                    causal_inference_text += f"    To get unbiased estimate of {cause}'s effect, adjust for these confounders\n"
                else:
                    causal_inference_text += f"  • {cause} → {target_factor}\n"
                    causal_inference_text += f"    ✓ No adjustment needed (no confounding)\n"
        
        if direct_effects:
            causal_inference_text += f"\nDirect Effects of {target_factor}:\n"
            for effect in direct_effects:
                causal_inference_text += f"  • {target_factor} → {effect}\n"
        
        # Collect all unique confounders
        all_confounders = set()
        for adj_set in adjustment_sets.values():
            if adj_set:
                all_confounders.update(adj_set)
        
        if all_confounders:
            causal_inference_text += f"\nConfounding Variables (Control Variables):\n"
            causal_inference_text += f"These variables create confounding and should be controlled for:\n"
            for confounder in sorted(all_confounders):
                causal_inference_text += f"  • {confounder}\n"
            causal_inference_text += f"\nIMPORTANT: When estimating causal effects or planning interventions,\n"
            causal_inference_text += f"you must control for these confounders to avoid biased estimates.\n"
    
    # Build network statistics
    stats_text = ""
    if network_summary:
        stats_text = f"\n\nNETWORK STATISTICS:\n"
        stats_text += f"- Total causal relationships: {network_summary.get('total_causal_relationships', 0)}\n"
        stats_text += f"- Direct causes of {target_factor}: {network_summary.get('factors_influencing_target', 0)}\n"
        stats_text += f"- Factors analyzed: {network_summary.get('total_factors', 0)}\n"
        if causal_inference_summary:
            stats_text += f"- Causal inference enabled: Yes\n"
            stats_text += f"- Confounders identified: {len(all_confounders) if 'all_confounders' in locals() else 0}\n"
    
    # Get prompt from CausalIFPrompts - now includes causal inference text
    prompt = CausalIFPrompts.interpretation_prompt(
        original_query=original_query,
        target_factor=target_factor,
        graph_description=graph_description,
        influences_text=influences_text,
        stats_text=stats_text
    )
    
    # Append causal inference information to the prompt
    if causal_inference_text:
        prompt += causal_inference_text
        prompt += "\n\nIMPORTANT: In your analysis, explicitly mention:\n"
        prompt += "1. Which causal relationships require controlling for confounders\n"
        prompt += "2. What the confounders are and why they matter\n"
        prompt += "3. How to design experiments or interventions that account for confounding\n"
    
    try:
        print("\n" + "="*80)
        print("Generating LLM interpretation of causal graph...")
        if causal_inference_summary:
            print("  ✓ Including causal inference analysis (adjustment sets & confounders)")
        print("="*80)
        
        response = model.invoke(prompt)
        interpretation = response.content if hasattr(response, 'content') else str(response)
        
        print("✓ LLM interpretation generated successfully\n")
        
        return interpretation
        
    except Exception as e:
        print(f"Error generating LLM interpretation: {e}")
        return f"Error generating interpretation: {str(e)}"

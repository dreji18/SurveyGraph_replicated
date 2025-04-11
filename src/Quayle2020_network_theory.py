#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quayle2020 Network Theory Implementation with sample dataset
@author: dreji18
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import List, Dict, Set, Tuple
import random
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

class AttitudeNetwork:
    def __init__(self):
        # Initialize the bipartite network
        self.network = nx.Graph()
        self.people = set()
        self.attitudes = set()
        # Track attitude expressions over time
        self.expression_history = []
        
    def add_person(self, person_id: str, potential_attitudes: List[str]):
        ## Add a person with their potential attitudes
        self.people.add(person_id)
        self.network.add_node(person_id, type='person', 
                            potential_attitudes=set(potential_attitudes),
                            expressed_attitudes=set())
    
    def express_attitude(self, person_id: str, attitude: str, timestamp: int):
        ## Person expresses an attitude from their potential set
        if person_id not in self.people:
            raise ValueError(f"Person {person_id} not in network")
            
        potential_attitudes = self.network.nodes[person_id]['potential_attitudes']
        if attitude not in potential_attitudes:
            raise ValueError(f"Attitude {attitude} not in person's potential set")
            
        # Add attitude to network if new
        if attitude not in self.attitudes:
            self.attitudes.add(attitude)
            self.network.add_node(attitude, type='attitude')
            
        # Create edge between person and attitude
        self.network.add_edge(person_id, attitude, timestamp=timestamp)
        
        # Update expressed attitudes
        self.network.nodes[person_id]['expressed_attitudes'].add(attitude)
        
        # Record expression
        self.expression_history.append({
            'person': person_id,
            'attitude': attitude,
            'timestamp': timestamp
        })
    
    def get_agreement_network(self) -> nx.Graph:
        ## Generate network of people connected by shared attitudes
        agreement_network = nx.Graph()
        
        # Add all people
        agreement_network.add_nodes_from(self.people)
        
        # Connect people who share attitudes
        for p1 in self.people:
            for p2 in self.people:
                if p1 < p2:  # Avoid duplicate edges
                    p1_attitudes = self.network.nodes[p1]['expressed_attitudes']
                    p2_attitudes = self.network.nodes[p2]['expressed_attitudes']
                    shared = len(p1_attitudes.intersection(p2_attitudes))
                    if shared > 0:
                        agreement_network.add_edge(p1, p2, weight=shared)
                        
        return agreement_network
    
    def get_attitude_clusters(self) -> List[Set[str]]:
        ## Identify clusters of attitudes that tend to be expressed together
        # Create attitude-attitude network based on co-expression
        attitude_network = nx.Graph()
        attitude_network.add_nodes_from(self.attitudes)
        
        for person in self.people:
            expressed = list(self.network.nodes[person]['expressed_attitudes'])
            for i in range(len(expressed)):
                for j in range(i+1, len(expressed)):
                    a1, a2 = expressed[i], expressed[j]
                    if attitude_network.has_edge(a1, a2):
                        attitude_network[a1][a2]['weight'] += 1
                    else:
                        attitude_network.add_edge(a1, a2, weight=1)
        
        # Use community detection to find clusters
        communities = nx.community.louvain_communities(attitude_network)
        return communities
    
    def identify_identity_groups(self, min_similarity: float = 0.5) -> List[Set[str]]:
        ## Find groups of people with similar attitude expressions
        agreement_network = self.get_agreement_network()
        
        # Normalize edge weights by maximum possible agreements
        for u, v in agreement_network.edges():
            max_possible = min(
                len(self.network.nodes[u]['expressed_attitudes']),
                len(self.network.nodes[v]['expressed_attitudes'])
            )
            agreement_network[u][v]['weight'] /= max_possible
            
        # Remove edges below similarity threshold
        edges_to_remove = [(u, v) for u, v in agreement_network.edges() 
                          if agreement_network[u][v]['weight'] < min_similarity]
        agreement_network.remove_edges_from(edges_to_remove)
        
        # Find connected components (identity groups)
        return list(nx.connected_components(agreement_network))
    
    def plot_bipartite_network(self, figsize=(12, 8)):
        ## Visualize the bipartite network of people and attitudes"""
        plt.figure(figsize=figsize)
        
        # Create layout
        pos = nx.spring_layout(self.network)
        
        # Draw nodes
        people_nodes = [n for n in self.network.nodes() if n in self.people]
        attitude_nodes = [n for n in self.network.nodes() if n in self.attitudes]
        
        nx.draw_networkx_nodes(self.network, pos, nodelist=people_nodes, 
                             node_color='lightblue', node_size=500, label='People')
        nx.draw_networkx_nodes(self.network, pos, nodelist=attitude_nodes,
                             node_color='lightgreen', node_size=300, label='Attitudes')
        
        # Draw edges
        nx.draw_networkx_edges(self.network, pos, alpha=0.5)
        
        # Add labels
        nx.draw_networkx_labels(self.network, pos)
        
        plt.title("Bipartite Network of People and Attitudes")
        plt.legend()
        plt.axis('off')
        return plt

    def plot_agreement_network(self, figsize=(10, 10)):
        ## Visualize the network of people connected by shared attitudes"""
        agreement_network = self.get_agreement_network()
        
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(agreement_network)
        
        # Get edge weights for width and color
        edges = agreement_network.edges()
        weights = [agreement_network[u][v]['weight'] for u, v in edges]
        
        # Draw the network
        nx.draw_networkx_nodes(agreement_network, pos, node_color='lightblue', 
                             node_size=500)
        nx.draw_networkx_edges(agreement_network, pos, edge_color=weights,
                             width=np.array(weights)*2, edge_cmap=plt.cm.Blues)
        nx.draw_networkx_labels(agreement_network, pos)
        
        plt.title("Agreement Network (Edge Width = Shared Attitudes)")
        plt.axis('off')
        return plt

    def plot_attitude_heatmap(self, figsize=(12, 8)):
        ## Create a heatmap showing which attitudes are expressed by each person"""
        # Create matrix of expressions
        matrix = np.zeros((len(self.people), len(self.attitudes)))
        people_list = sorted(list(self.people))
        attitudes_list = sorted(list(self.attitudes))
        
        for i, person in enumerate(people_list):
            expressed = self.network.nodes[person]['expressed_attitudes']
            for j, attitude in enumerate(attitudes_list):
                matrix[i, j] = 1 if attitude in expressed else 0
        
        # Create heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(matrix, xticklabels=attitudes_list, yticklabels=people_list,
                   cmap='YlOrRd', cbar_kws={'label': 'Expressed'})
        plt.title("Attitude Expression Patterns")
        plt.xlabel("Attitudes")
        plt.ylabel("People")
        plt.xticks(rotation=45, ha='right')
        return plt

    def plot_expression_timeline(self, figsize=(15, 6)):
        ## Visualize how attitudes are expressed over time"""
        df = pd.DataFrame(self.expression_history)
        
        plt.figure(figsize=figsize)
        for person in self.people:
            person_expr = df[df['person'] == person]
            plt.scatter(person_expr['timestamp'], person_expr['attitude'], 
                       label=person, alpha=0.7)
            
        plt.title("Timeline of Attitude Expressions")
        plt.xlabel("Time")
        plt.ylabel("Attitudes")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        return plt

# Example usage with sample data
def create_sample_data():
    # Create some sample attitudes that are logically consistent
    progressive_political = ['pro_immigration', 'pro_welfare', 'pro_regulation']
    conservative_political = ['anti_immigration', 'anti_welfare', 'anti_regulation']
    
    urban_cultural = ['likes_jazz', 'likes_rock', 'prefers_cities', 'tech_positive']
    rural_cultural = ['likes_country', 'likes_folk', 'prefers_rural', 'tech_skeptical']
    
    # Create sample people with logically consistent attitude sets
    people_data = {
        'P1': ['pro_immigration', 'pro_welfare', 'pro_regulation', 
               'likes_jazz', 'prefers_cities', 'tech_positive'],  # Urban progressive
        
        'P2': ['anti_immigration', 'anti_welfare', 'anti_regulation',
               'likes_country', 'prefers_rural', 'tech_skeptical'],  # Rural conservative
        
        'P3': ['pro_immigration', 'pro_welfare', 'anti_regulation',
               'likes_rock', 'prefers_cities', 'tech_positive'],  # Mixed urban progressive
        
        'P4': ['anti_immigration', 'anti_welfare', 'pro_regulation',
               'likes_folk', 'prefers_rural', 'tech_skeptical'],  # Mixed rural conservative
        
        'P5': ['pro_immigration', 'anti_welfare', 'pro_regulation',
               'likes_jazz', 'likes_rock', 'prefers_cities']  # Urban mixed
    }
    
    return people_data

def create_polarized_sample_data():
    # Create strongly polarized attitude sets
    
    # Progressive urban tech enthusiasts
    progressive_urban = {
        'P1': ['pro_immigration', 'pro_welfare', 'pro_regulation', 
               'likes_electronic', 'prefers_cities', 'tech_positive', 
               'pro_climate_action', 'vegan_diet'],
        
        'P2': ['pro_immigration', 'pro_welfare', 'pro_regulation',
               'likes_jazz', 'prefers_cities', 'tech_positive',
               'pro_climate_action', 'vegetarian_diet']
    }
    
    # Conservative rural traditionalists
    conservative_rural = {
        'P3': ['anti_immigration', 'anti_welfare', 'anti_regulation',
               'likes_country', 'prefers_rural', 'tech_skeptical',
               'climate_skeptical', 'traditional_diet'],
        
        'P4': ['anti_immigration', 'anti_welfare', 'anti_regulation',
               'likes_folk', 'prefers_rural', 'tech_skeptical',
               'climate_skeptical', 'traditional_diet']
    }
    
    # Libertarian tech entrepreneurs
    libertarian_tech = {
        'P5': ['anti_regulation', 'anti_welfare', 'pro_immigration',
               'likes_rock', 'prefers_suburbs', 'tech_positive',
               'market_solution', 'flexible_diet']
    }
    
    # All groups combined
    people_data = {**progressive_urban, **conservative_rural, **libertarian_tech}
    
    return people_data

def create_immigration_attitude_data():
    # Different societal groups and their potential attitudes
    
    # Progressive Urban Group
    progressive_attitudes = {
        'P1': [
            'pro_immigration', 'support_diverse_workforce',
            'gender_equality', 'cultural_integration',
            'multilingual_support', 'equal_pay',
            'childcare_support', 'workplace_inclusion'
        ],
        'P2': [
            'pro_immigration', 'support_family_reunion',
            'gender_equality', 'cultural_celebration',
            'education_access', 'healthcare_access',
            'women_entrepreneurship', 'community_programs'
        ]
    }
    
    # Conservative Group
    conservative_attitudes = {
        'P3': [
            'immigration_control', 'traditional_values',
            'language_requirements', 'skills_based_entry',
            'assimilation_preference', 'national_security',
            'merit_based_pay', 'self_reliance'
        ],
        'P4': [
            'immigration_control', 'cultural_preservation',
            'traditional_family', 'economic_priority',
            'citizenship_requirements', 'border_security',
            'domestic_workforce', 'welfare_restrictions'
        ]
    }
    
    # Moderate/Mixed Views
    moderate_attitudes = {
        'P5': [
            'regulated_immigration', 'integration_programs',
            'work_life_balance', 'skill_development',
            'equal_opportunity', 'market_driven_policy',
            'public_private_partnership', 'merit_based_entry'
        ],
        'P6': [
            'regulated_immigration', 'cultural_adaptation',
            'family_support', 'economic_contribution',
            'language_learning', 'community_engagement',
            'workplace_safety', 'social_integration'
        ]
    }
    
    # Immigrant Perspectives
    immigrant_attitudes = {
        'P7': [
            'cultural_identity', 'family_reunification',
            'career_advancement', 'education_value',
            'community_support', 'gender_specific_challenges',
            'discrimination_awareness', 'economic_opportunity'
        ],
        'P8': [
            'dual_culture', 'children_education',
            'professional_growth', 'healthcare_importance',
            'social_networks', 'workplace_rights',
            'cultural_exchange', 'language_preservation'
        ]
    }

    # Combine all groups
    people_data = {
        **progressive_attitudes,
        **conservative_attitudes,
        **moderate_attitudes,
        **immigrant_attitudes
    }
    
    return people_data

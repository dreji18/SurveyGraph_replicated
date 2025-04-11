from src.utility_functions import *
import numpy as np

class SurveyGraph:
    def __init__(self, matrix, mode, method_value, similarity_metric):
        self.matrix = matrix
        self.mode = mode
        self.method_value = method_value
        self.similarity_metric = similarity_metric
        self.g_agent = {'network': {}}
        self.g_symbolic = {'network': {}}

    def _dummy_project(self, layer):
        n = len(self.matrix)
        graph = {'network': {}}
        for i in range(n):
            for j in range(i + 1, n):
                similarity = np.dot(self.matrix[i], self.matrix[j])
                if similarity > self.method_value:
                    graph['network'].setdefault(i, []).append({'u': j, 'w': similarity})
                    graph['network'].setdefault(j, []).append({'u': i, 'w': similarity})
        if layer == 'agent':
            self.g_agent = graph
        else:
            self.g_symbolic = graph

    def make_proj_agent_lcc(self): self._dummy_project('agent')
    def make_proj_agent_ad(self): self._dummy_project('agent')
    def make_proj_agent_similar(self): self._dummy_project('agent')
    def make_proj_symbolic_lcc(self): self._dummy_project('symbolic')
    def make_proj_symbolic_ad(self): self._dummy_project('symbolic')
    def make_proj_symbolic_similar(self): self._dummy_project('symbolic')


def rmake_proj_agent_lcc(df, mvalue, c, sim_metric):
    surveytmp = df_to_cppvector(df)
    surveytmp = normalise_columns(surveytmp)
    S = SurveyGraph(surveytmp, 0, mvalue, sim_metric)
    S.make_proj_agent_lcc()
    return cppvector_to_df(S.g_agent, c)


def rmake_proj_agent_ad(df, mvalue, c, sim_metric):
    surveytmp = df_to_cppvector(df)
    surveytmp = normalise_columns(surveytmp)
    S = SurveyGraph(surveytmp, 1, mvalue, sim_metric)
    S.make_proj_agent_ad()
    return cppvector_to_df(S.g_agent, c)


def rmake_proj_agent_similar(df, mvalue, c, sim_metric):
    surveytmp = df_to_cppvector(df)
    surveytmp = normalise_columns(surveytmp)
    S = SurveyGraph(surveytmp, 2, mvalue, sim_metric)
    S.make_proj_agent_similar()
    return cppvector_to_df(S.g_agent, c)


def rmake_proj_symbolic_lcc(df, mvalue, c, sim_metric):
    surveytmp = df_to_cppvector(df)
    surveytmp = normalise_columns(surveytmp)
    S = SurveyGraph(surveytmp, 0, mvalue, sim_metric)
    S.make_proj_symbolic_lcc()
    return cppvector_to_df(S.g_symbolic, c)


def rmake_proj_symbolic_ad(df, mvalue, c, sim_metric):
    surveytmp = df_to_cppvector(df)
    surveytmp = normalise_columns(surveytmp)
    S = SurveyGraph(surveytmp, 1, mvalue, sim_metric)
    S.make_proj_symbolic_ad()
    return cppvector_to_df(S.g_symbolic, c)


def rmake_proj_symbolic_similar(df, mvalue, c, sim_metric):
    surveytmp = df_to_cppvector(df)
    surveytmp = normalise_columns(surveytmp)
    S = SurveyGraph(surveytmp, 2, mvalue, sim_metric)
    S.make_proj_symbolic_similar()
    return cppvector_to_df(S.g_symbolic, c)

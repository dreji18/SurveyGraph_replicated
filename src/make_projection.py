from src.projection_functions import *

def make_projection(data, layer, threshold_method=None, method_value=None, 
                    centre=None, similarity_metric=None):

    if layer not in {"agent", "symbolic"}:
        print("layer needs to be either agent or symbolic")
        return None

    centre = int(centre) if isinstance(centre, bool) else (
        1 if (centre is None and layer == "agent") else
        0 if (centre is None and layer == "symbolic") else
        centre
    )

    if similarity_metric != "manhattan" and similarity_metric is not None:
        print("overriding similarity metric to manhattan distance")

    similarity_metric = 0  # always mapped to int(0) for now

    proj_funcs = {
        "agent": {
            "target_lcc": rmake_proj_agent_lcc,
            "target_ad": rmake_proj_agent_ad,
            "raw_similarity": rmake_proj_agent_similar,
            "default": lambda d, c, s: rmake_proj_agent_lcc(d, 0.97, c, s)
        },
        "symbolic": {
            "target_lcc": rmake_proj_symbolic_lcc,
            "target_ad": rmake_proj_symbolic_ad,
            "raw_similarity": rmake_proj_symbolic_similar,
            "default": lambda d, c, s: rmake_proj_symbolic_lcc(d, 0.97, c, s)
        }
    }

    funcs = proj_funcs[layer]

    if threshold_method and method_value is not None:
        func = funcs.get(threshold_method)
        if func:
            return func(data, method_value, centre, similarity_metric)
        else:
            print("threshold_method must be target_lcc, target_ad, or raw_similarity")
            default_func = (
                rmake_proj_agent_lcc if layer == "agent" else rmake_proj_symbolic_ad
            )
            default_value = 0.95 if layer == "agent" else 1
            return default_func(data, default_value, centre, similarity_metric)

    return funcs["default"](data, centre, similarity_metric)

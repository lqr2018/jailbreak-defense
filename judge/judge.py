from .no_judge import NoJudge

JUDGE_METHODS = {
    "no-judge": NoJudge
}

def load_judge_method(judge_name, goal, **kwargs):
    if judge_name not in JUDGE_METHODS:
        judge_name, model_name = judge_name.split("@")
    else:
        model_name = None

    if judge_name not in JUDGE_METHODS:
        raise NotImplementedError

    return JUDGE_METHODS[judge_name](goal, model_name, **kwargs)


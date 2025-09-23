def _default_compute_score_format(data_source, solution_str, extra_info=None):
    if data_source == 'hotpotqa/hotpot_qa' or data_source == 'bdsaglam/musique' or data_source == 'xanhho/2WikiMultihopQA':
        from . import qa_em_and_format
        res = qa_em_and_format.compute_score_format(solution_str)
    elif data_source == 'openai/gsm8k':
        from . import gsm8k
        res = gsm8k.compute_score_format(solution_str)
    elif data_source == 'BytedTsinghua-SIA/DAPO-Math-17k':
        from . import retool
        res = retool.compute_score_format(solution_str)
    elif data_source in ['WTQ', 'TableBench', 'TabMWP', 'HiTab', 'FinQA']:
        from . import tqa
        res = tqa.compute_score_format(solution_str)
    elif data_source in ['TabFact']:
        from . import tfv
        res = tfv.compute_score_format(solution_str)
    else:
        raise NotImplementedError
    
    if isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
    
def _default_compute_score_answer(data_source, solution_str, ground_truth, extra_info=None):
    if data_source == 'hotpotqa/hotpot_qa' or data_source == 'bdsaglam/musique' or data_source == 'xanhho/2WikiMultihopQA':
        from . import qa_em_and_format
        res = qa_em_and_format.compute_score_em(solution_str, ground_truth)
    elif data_source == 'openai/gsm8k':
        from . import gsm8k
        res = gsm8k.compute_score_answer(solution_str, ground_truth)
    elif data_source == 'BytedTsinghua-SIA/DAPO-Math-17k':
        from . import retool
        res = retool.compute_score_answer(solution_str, ground_truth)
    elif data_source in ['WTQ', 'TableBench', 'TabMWP', 'HiTab', 'FinQA']:
        from . import tqa
        res = tqa.compute_score_answer(solution_str, ground_truth)
    elif data_source in ['TabFact']:
        from . import tfv
        res = tfv.compute_score_answer(solution_str, ground_truth)
    else:
        raise NotImplementedError
    
    if isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
    
def _default_compute_score_format_answer(data_source, solution_str, ground_truth, extra_info=None):
    if data_source == 'hotpotqa/hotpot_qa' or data_source == 'bdsaglam/musique' or data_source == 'xanhho/2WikiMultihopQA':
        from . import qa_em_and_format
        res = qa_em_and_format.compute_score_format_answer(solution_str, ground_truth)
    elif data_source == 'openai/gsm8k':
        from . import gsm8k
        res = gsm8k.compute_score_format_answer(solution_str, ground_truth)
    elif data_source == 'BytedTsinghua-SIA/DAPO-Math-17k':
        from . import retool
        res = retool.compute_score_format_answer(solution_str, ground_truth)
    elif data_source in ['WTQ', 'TableBench', 'TabMWP', 'HiTab', 'FinQA']:
        from . import tqa
        res = tqa.compute_score_format_answer(solution_str, ground_truth, extra_info)
    elif data_source in ['TabFact']:
        from . import tfv
        res = tfv.compute_score_format_answer(solution_str, ground_truth, extra_info)
    else:
        raise NotImplementedError
    
    if isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
    
def _default_compute_tool_call(data_source, extra_info=None):
    if data_source in ['WTQ', 'TableBench', 'TabMWP', 'HiTab', 'FinQA']:
        from . import tqa
        res = tqa.compute_tool_call(extra_info)
    elif data_source in ['TabFact']:
        from . import tfv
        res = tfv.compute_tool_call(extra_info)
    else:
        raise NotImplementedError
    
    if isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
    
def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    return {
        "score": _default_compute_score_format_answer(data_source, solution_str, ground_truth, extra_info),
        "acc": _default_compute_score_answer(data_source, solution_str, ground_truth, extra_info),
        "format": _default_compute_score_format(data_source, solution_str, extra_info),
        "tool_call_score": _default_compute_tool_call(data_source, extra_info),
    }
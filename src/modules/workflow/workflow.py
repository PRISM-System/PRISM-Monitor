from pathlib import Path
from src.modules.llm.llm import LLMCallManager

def workflow(log):
    prompt = f"""
    주어진 제조공정 로그를 분석하여 종합적인 보고서를 작성하세요.
    ## 제조공정 로그
    {log}
    """
    analysis = LLMCallManager.invoke(prompt=prompt)
    print('workflow analysis result:', analysis)

    prompt = f"""
    주어진 제조공정 보고서를 요약해주세요.
    ## 제조공정 보고서
    {analysis}
    """
    summary = LLMCallManager.invoke(prompt=prompt)
    print('workflow summary result:', summary)
    return {
        'summary': summary,
        'result': analysis
    }
    

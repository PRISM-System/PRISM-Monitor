# task_executor.py
import json
from typing import Dict, Any, Callable, List

def analyze_parameter_status(params: Dict[str, Any]) -> Dict[str, Any]:
    # 예시: 모니터링/알람 요약 산출 (실장비 연동 없음)
    return {"status": "OK", "summary": "Analyzed parameter trends", "params": params}

def forecast_quality_risk(params: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "OK", "summary": "Predicted risk trajectory", "params": params}

def recommend_control_update(params: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "OK", "summary": "Proposed control adjustments with sim results", "params": params}

def summarize_recent_events(params: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "OK", "summary": "Summarized recent alarms and KPIs", "params": params}

def search_and_optimize_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "OK", "summary": "Generated optimization plan and DoE", "params": params}

ACTION_REGISTRY: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "analyze_parameter_status": analyze_parameter_status,
    "forecast_quality_risk": forecast_quality_risk,
    "recommend_control_update": recommend_control_update,
    "summarize_recent_events": summarize_recent_events,
    "search_and_optimize_parameters": search_and_optimize_parameters,
}

def execute_instruction(instruction: Dict[str, Any]) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    for task in instruction.get("tasks", []):
        action = task.get("action")
        fn = ACTION_REGISTRY.get(action)
        if not fn:
            results.append({"task_id": task.get("task_id"), "status": "FAILED", "error": f"Unknown action: {action}"})
            continue
        res = fn(task.get("parameters", {}))
        results.append({"task_id": task.get("task_id"), "status": "DONE", "result": res})
    return {"instruction_id": instruction.get("instruction_id"), "results": results}

if __name__ == "__main__":
    # stdin 예시 실행: predictions.jsonl에서 첫 모델 출력 읽기
    import sys
    if len(sys.argv) < 2:
        print("Usage: python task_executor.py <instruction.json>")
        sys.exit(1)
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        instruction = json.load(f)
    out = execute_instruction(instruction)
    print(json.dumps(out, ensure_ascii=False, indent=2))
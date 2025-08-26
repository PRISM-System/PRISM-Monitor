import os
import requests
import json
import numpy as np
import pandas as pd
import pickle

from typing import Dict, Any, List, Optional
from tensorflow import keras

from models.anomaly_detect import EnhancedSemiconductorRealTimeMonitor
from prism_core.core.tools.base import BaseTool
from prism_core.core.tools.schemas import ToolRequest, ToolResponse



class AnomalyDetectTool(BaseTool):
    """
    이상 이벤트 탐지 도구
    """
    
    def __init__(self, 
                 model_dir: str = "pretrained_models/anomaly_detect",
                 model_metadata_file: str = "anomaly_model_metadata",
                 client_id: str = "monitoring",
                ):
        super().__init__(
            name="anomaly_detect",
            description="주어진 제조 공정 데이터의 이상 여부 탐지 및 분석",
            parameters_schema={
                "type": "object",
                "properties": {
                    "rows": {"type": "array", "items": {"type": "object"}, "description": "이상 탐지를 수행할 제조 공정 데이터의 일부 (데이터프레임 형식)"},
                },
                "required": ["rows"]
            }
        )
        # 에이전트별 설정 또는 기본값 사용
        self._model_metadata_file = model_metadata_file
        self._client_id = client_id
        self._detector = EnhancedSemiconductorRealTimeMonitor(model_dir=model_dir)
        
        
    def execute(self, request: ToolRequest) -> ToolResponse:
        """Tool 실행"""
        try:
            params = request.parameters
            unified_df = params["rows"]
            anomaly_records, svg_content, analysis_results = self._detector.fast_anomaly_detection(unified_df=unified_df)
            
            return ToolResponse(
                success=True,
                data={
                    "anomaly_records": anomaly_records,
                    "svg_content": svg_content,
                    "analysis_results": analysis_results
                }
            )
        except Exception as e:
            return ToolResponse(
                success=False,
                error=f"규정 준수 검증 실패: {str(e)}"
            )

    
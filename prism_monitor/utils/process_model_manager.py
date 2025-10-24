"""
공정별 모델 관리자

각 공정별로 독립적인 모델을 관리하고 로딩합니다.
"""

import os
import json
import pickle
from typing import Dict, List, Tuple, Optional
from tensorflow import keras


class ProcessModelManager:
    """
    공정별 모델을 관리하는 매니저 클래스

    각 공정(CMP, Etch, CVD, Ion Implant, Photo)마다 독립적인 모델을 로드하고 관리합니다.
    """

    def __init__(self, base_model_dir: str = "models"):
        """
        Args:
            base_model_dir: 모델 베이스 디렉토리 (기본값: "models")
        """
        self.base_model_dir = base_model_dir
        self.loaded_models = {}  # 모델 캐싱: {process_name: (model, scaler, metadata)}

        # 공정별 모델 경로 매핑 (20개 공정)
        self.process_model_paths = {
            # Semiconductor (4개)
            'semiconductor_cmp_001': 'semiconductor_cmp_001',
            'semiconductor_etch_002': 'semiconductor_etch_002',
            'semiconductor_deposition_003': 'semiconductor_deposition_003',
            'semiconductor_full_004': 'semiconductor_full_004',
            # Chemical (4개)
            'chemical_reactor_001': 'chemical_reactor_001',
            'chemical_distillation_002': 'chemical_distillation_002',
            'chemical_refining_003': 'chemical_refining_003',
            'chemical_full_004': 'chemical_full_004',
            # Automotive (4개)
            'automotive_welding_001': 'automotive_welding_001',
            'automotive_painting_002': 'automotive_painting_002',
            'automotive_press_003': 'automotive_press_003',
            'automotive_assembly_004': 'automotive_assembly_004',
            # Battery (4개)
            'battery_formation_001': 'battery_formation_001',
            'battery_coating_002': 'battery_coating_002',
            'battery_aging_003': 'battery_aging_003',
            'battery_production_004': 'battery_production_004',
            # Steel (4개)
            'steel_rolling_001': 'steel_rolling_001',
            'steel_converter_002': 'steel_converter_002',
            'steel_casting_003': 'steel_casting_003',
            'steel_production_004': 'steel_production_004',
        }

    def get_model_for_process(self, process_name: str) -> Tuple[Optional[keras.Model], Optional[object], Optional[Dict]]:
        """
        특정 공정의 모델 로드

        Args:
            process_name: 공정명 (예: 'semi_cmp_sensors', 'semi_etch_sensors', ...)

        Returns:
            (model, scaler, metadata) 튜플
            - model: Keras 모델
            - scaler: StandardScaler 또는 RobustScaler
            - metadata: 모델 메타데이터 딕셔너리

        Raises:
            ValueError: 모델을 찾을 수 없는 경우
        """
        # 이미 로드된 경우 캐시에서 반환
        if process_name in self.loaded_models:
            print(f"캐시에서 모델 로드: {process_name}")
            return self.loaded_models[process_name]

        # 공정별 모델 디렉토리
        process_model_dir = os.path.join(
            self.base_model_dir,
            self.process_model_paths.get(process_name, process_name)
        )

        if not os.path.exists(process_model_dir):
            raise ValueError(f"Model directory not found for process: {process_name} (path: {process_model_dir})")

        # 메타데이터 파일 확인
        metadata_file = os.path.join(process_model_dir, "model_metadata.json")
        if not os.path.exists(metadata_file):
            raise ValueError(f"Model metadata not found for process: {process_name}")

        # 메타데이터 로드
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        # 모델 파일 로드
        model_file = os.path.join(process_model_dir, "autoencoder_model.h5")
        if not os.path.exists(model_file):
            raise ValueError(f"Model file not found: {model_file}")

        try:
            from tensorflow.keras.metrics import MeanSquaredError
            import tensorflow as tf
            # GPU CuDNN 버전 불일치 우회 - CPU에서 모델 로드
            with tf.device('/CPU:0'):
                model = keras.models.load_model(model_file, custom_objects={"mse": MeanSquaredError()})
        except Exception as e:
            raise ValueError(f"Failed to load model for {process_name}: {e}")

        # 스케일러 로드
        scaler_file = os.path.join(process_model_dir, "scaler.pkl")
        scaler = None
        if os.path.exists(scaler_file):
            with open(scaler_file, 'rb') as f:
                scaler = pickle.load(f)
        else:
            print(f"Warning: Scaler not found for {process_name}")

        # 캐싱
        self.loaded_models[process_name] = (model, scaler, metadata)

        print(f"모델 로드 완료: {process_name} (version: {metadata.get('model_version', 'unknown')})")

        return model, scaler, metadata

    def list_available_processes(self) -> List[str]:
        """
        사용 가능한 공정 모델 목록 조회

        Returns:
            공정명 리스트 (모델이 존재하는 공정만)
        """
        available = []

        for process_name, model_path in self.process_model_paths.items():
            full_path = os.path.join(self.base_model_dir, model_path, "model_metadata.json")
            if os.path.exists(full_path):
                available.append(process_name)

        return available

    def get_model_info(self, process_name: str) -> Dict:
        """
        특정 공정 모델의 메타데이터 조회

        Args:
            process_name: 공정명

        Returns:
            모델 정보 딕셔너리
        """
        process_model_dir = os.path.join(
            self.base_model_dir,
            self.process_model_paths.get(process_name, process_name)
        )
        metadata_file = os.path.join(process_model_dir, "model_metadata.json")

        if not os.path.exists(metadata_file):
            return {
                'available': False,
                'process_name': process_name,
                'message': f'No model found for {process_name}'
            }

        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            return {
                'available': True,
                'process_name': process_name,
                'model_version': metadata.get('model_version'),
                'training_timestamp': metadata.get('training_timestamp'),
                'feature_count': len(metadata.get('feature_columns', [])),
                'threshold': metadata.get('threshold'),
                'training_info': metadata.get('training_data_info'),
                'performance_metrics': metadata.get('performance_metrics'),
            }
        except Exception as e:
            return {
                'available': False,
                'process_name': process_name,
                'error': str(e)
            }

    def is_model_available(self, process_name: str) -> bool:
        """
        특정 공정의 모델이 사용 가능한지 확인

        Args:
            process_name: 공정명

        Returns:
            모델 사용 가능 여부
        """
        return process_name in self.list_available_processes()

    def clear_cache(self):
        """모델 캐시 초기화"""
        self.loaded_models.clear()
        print("모델 캐시 초기화 완료")

    def get_all_models_info(self) -> Dict[str, Dict]:
        """
        모든 사용 가능한 모델의 정보 조회

        Returns:
            {process_name: model_info} 딕셔너리
        """
        all_info = {}

        for process_name in self.list_available_processes():
            all_info[process_name] = self.get_model_info(process_name)

        return all_info


if __name__ == "__main__":
    # 테스트 코드
    print("=== ProcessModelManager 테스트 ===\n")

    manager = ProcessModelManager(base_model_dir="models")

    print("1. 사용 가능한 공정 목록:")
    available_processes = manager.list_available_processes()
    print(f"   {available_processes}")

    print("\n2. 각 공정별 모델 정보:")
    all_info = manager.get_all_models_info()
    for process_name, info in all_info.items():
        print(f"\n   {process_name}:")
        if info['available']:
            print(f"     - Version: {info['model_version']}")
            print(f"     - Features: {info['feature_count']}개")
            print(f"     - Threshold: {info['threshold']}")
        else:
            print(f"     - Status: Not available")

    # 모델 로드 테스트 (CMP가 있다고 가정)
    if 'semiconductor_cmp_001' in available_processes:
        print("\n3. CMP 모델 로드 테스트:")
        try:
            model, scaler, metadata = manager.get_model_for_process('semiconductor_cmp_001')
            print(f"   ✓ 모델 로드 성공")
            print(f"   - Feature columns: {metadata['feature_columns'][:5]}...")
        except Exception as e:
            print(f"   ✗ 모델 로드 실패: {e}")

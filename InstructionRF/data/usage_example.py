#!/usr/bin/env python3
"""
제조업 AI 데이터셋 사용 예제
"""
import json
import os

def load_and_use_dataset():
    """데이터셋 로드 및 사용 예제"""
    
    # 1. JSON 데이터셋 로드
    with open('data/manufacturing_dataset_100.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 메타데이터와 데이터셋 분리
    metadata = data.get('metadata', {})
    dataset = data.get('dataset', [])
    
    print(f"데이터셋 정보:")
    print(f"  제목: {metadata.get('title', 'Unknown')}")
    print(f"  총 데이터 수: {len(dataset)}건")
    print(f"  생성일: {metadata.get('created_at', 'Unknown')}")
    
    # 2. 의도별 데이터 분류
    intent_groups = {}
    for item in dataset:
        intent = item['expected_intent_type']
        if intent not in intent_groups:
            intent_groups[intent] = []
        intent_groups[intent].append(item)
    
    # 3. 샘플 데이터 출력
    print(f"\n각 의도별 샘플 데이터:")
    for intent, items in intent_groups.items():
        print(f"\n{intent} ({len(items)}건):")
        for item in items[:3]:  # 각 의도별로 3개씩만 출력
            print(f"  ID {item['id']}: {item['query']}")
            print(f"    → Priority: {item['expected_priority']}")

if __name__ == "__main__":
    load_and_use_dataset()

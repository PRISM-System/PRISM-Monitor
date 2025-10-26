# 🚀 이상치 탐지 빠른 테스트 가이드

## 📋 사용 가능한 공정명 (25개)

### 🔬 반도체 (Semiconductor) - 9개
- `semi_cmp_sensors` - CMP (Chemical Mechanical Polishing)
- `semi_etch_sensors` - Etch (식각)
- `semi_cvd_sensors` - CVD (Chemical Vapor Deposition)
- `semi_ion_sensors` - Ion Implantation
- `semi_photo_sensors` - Photolithography
- `semiconductor_cmp_001` - CMP (신규)
- `semiconductor_etch_002` - Etch (신규)
- `semiconductor_deposition_003` - Deposition (신규)
- `semiconductor_full_004` - Full Process (신규)

### ⚗️ 화학 (Chemical) - 4개
- `chemical_reactor_001` - Reactor (반응기)
- `chemical_distillation_002` - Distillation (증류)
- `chemical_refining_003` - Refining (정제)
- `chemical_full_004` - Full Process

### 🔋 배터리 (Battery) - 4개
- `battery_formation_001` - Formation (화성/충전)
- `battery_coating_002` - Coating (코팅)
- `battery_aging_003` - Aging (노화 테스트)
- `battery_production_004` - Production (전체 생산)

### 🚗 자동차 (Automotive) - 4개
- `automotive_welding_001` - Welding (용접)
- `automotive_painting_002` - Painting (도장)
- `automotive_press_003` - Press (프레스)
- `automotive_assembly_004` - Assembly (조립)

### 🏭 철강 (Steel) - 4개
- `steel_rolling_001` - Rolling (압연)
- `steel_converter_002` - Converter (전로)
- `steel_casting_003` - Casting (주조)
- `steel_production_004` - Production (전체 생산)

---

## 🎯 기본 테스트 명령어

### 1시간 데이터 분석
```bash
python test_event_detect_standalone.py --process [공정명] \
  --start "2025-05-01T00:00:00Z" --end "2025-05-01T01:00:00Z"
```

### 전체 하루 데이터 분석 (권장)
```bash
python test_event_detect_standalone.py --process [공정명] \
  --start "2025-05-01T00:00:00Z" --end "2025-05-01T23:59:59Z"
```

### 최근 1시간 데이터 (시간 생략)
```bash
python test_event_detect_standalone.py --process [공정명]
```

---

## 📝 산업군별 예시

### 반도체
```bash
# CMP 공정
python test_event_detect_standalone.py --process semi_cmp_sensors \
  --start "2025-05-01T00:00:00Z" --end "2025-05-01T01:00:00Z"
```

### 화학
```bash
# Reactor 공정
python test_event_detect_standalone.py --process chemical_reactor_001 \
  --start "2025-05-01T00:00:00Z" --end "2025-05-01T23:59:59Z"
```

### 배터리
```bash
# Formation 공정
python test_event_detect_standalone.py --process battery_formation_001 \
  --start "2025-05-01T00:00:00Z" --end "2025-05-01T23:59:59Z"
```

### 자동차
```bash
# Welding 공정
python test_event_detect_standalone.py --process automotive_welding_001 \
  --start "2025-05-01T00:00:00Z" --end "2025-05-01T23:59:59Z"
```

### 철강
```bash
# Rolling 공정
python test_event_detect_standalone.py --process steel_rolling_001 \
  --start "2025-05-01T00:00:00Z" --end "2025-05-01T23:59:59Z"
```

---

## 🛠️ 추가 옵션

### 모델 및 데이터 확인
```bash
python test_event_detect_standalone.py --check
```

### API 서버 지정
```bash
python test_event_detect_standalone.py --process semi_cmp_sensors \
  --url https://147.47.39.144:8000 \
  --start "2025-05-01T00:00:00Z" --end "2025-05-01T01:00:00Z"
```

### CSV 강제 사용
```bash
python test_event_detect_standalone.py --process semi_cmp_sensors --csv
```

### 도움말
```bash
python test_event_detect_standalone.py --help
```

---

## 📊 테스트 결과

### 콘솔 출력
- 실시간 진행 상황
- 탐지된 이상치 수
- 상위 5개 이상치 상세 정보

### JSON 파일
- 파일명: `test_results_[공정명].json`
- 내용: 전체 이상치 목록, 분석 요약, 메타데이터

---

## 💡 동작 원리

1. **URL 미제공**: 로컬 CSV 파일 자동 사용
2. **URL 제공**: API 시도 → 실패 시 CSV로 자동 폴백
3. **--csv 플래그**: API 시도 없이 CSV 직접 사용

---

## ✅ 테스트 검증 완료

- ✅ Semiconductor (반도체) - CMP, Etch 등
- ✅ Chemical (화학) - Reactor, Distillation 등
- ✅ Battery (배터리) - Formation, Aging 등
- ✅ Automotive (자동차) - 모든 공정
- ✅ Steel (철강) - 모든 공정

**총 25개 공정 모델 정상 작동 확인**

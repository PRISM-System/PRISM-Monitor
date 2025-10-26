FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# Pipenv가 .venv 폴더를 프로젝트 디렉토리(/app) 내부에 생성하도록 설정
ENV PIPENV_VENV_IN_PROJECT=1
# Python 버전을 3.11로 설정
ENV PIPENV_DEFAULT_PYTHON_VERSION=3.11

# 기본 패키지 설치
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    git \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    # Python 3.11 관련 패키지 설치
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    # 심볼릭 링크를 3.11로 변경
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

COPY .env-local .
RUN mv .env-local .env

# 애플리케이션 파일 나머지 복사
COPY . .


# pip 업그레이드 및 필수 패키지 설치 (3.11 사용)
RUN python3.11 -m ensurepip --upgrade \
    && python3.11 -m pip install --upgrade pip setuptools wheel

# -------------------------------------------------------------
# 👇 Pipenv 설치 및 종속성 복사
# -------------------------------------------------------------

# Pipenv 설치
RUN pip install pipenv

# Pipenv를 사용하여 가상 환경 생성 및 패키지 설치
# --deploy 플래그는 Pipfile.lock에 기반하여 정확히 설치합니다.
RUN pipenv install --deploy

# -------------------------------------------------------------
# 👇 .env-local을 .env로 복사 추가
# -------------------------------------------------------------

# 환경 변수 파일을 복사 (주의: .env-local이 프로젝트 루트에 존재해야 합니다)
# 이 파일을 복사하는 것은 빌드 시 환경 변수 설정이 필요하거나, 
# 애플리케이션 코드가 .env 파일을 읽도록 설계된 경우에 유용합니다.

# 포트 노출
EXPOSE 8001

# 시작 명령어
# pipenv run을 사용하여 가상 환경 내에서 uvicorn을 실행합니다.
CMD ["pipenv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
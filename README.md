# WatchTower-AI

교통 이상 징후 감지 시스템. YOLO 기반 AI 모델로 이미지·동영상에서 역주행, 갓길 정차, 정체, 화재 등을 감지하는 Flask 웹 애플리케이션입니다.

## 주요 기능

- **이미지 감지** — 이미지 업로드 후 AI 모델로 객체 감지 결과 확인
- **동영상 감지** — 동영상 업로드 후 백그라운드 처리, 감지 결과 영상 다운로드
- **다중 모델 지원** — 용도별 감지 모델 선택 가능
- **회원 인증** — 회원가입/로그인, 개인 업로드 관리

### 지원 감지 모델

| 모델 | 용도 |
|------|------|
| `yolo11n` | 일반 객체 감지 |
| `fire_detect_v1` | 화재 감지 |
| `shoulder_stop` | 갓길 정차 감지 |
| `wrong_way` | 역주행 감지 |
| `highway` | 고속도로 이상 징후 통합 감지 |

## 시작하기

### 요구사항

- Python 3.10 이상
- [uv](https://docs.astral.sh/uv/) 패키지 매니저 (또는 pip)

### 설치

```bash
# 저장소 클론
git clone https://github.com/smw312500-commit/vision-project.git
cd vision-project

# 가상환경 생성 및 패키지 설치 (uv 사용 시)
uv sync

# pip 사용 시
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux
pip install -e .
```

### 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성합니다.

```env
SECRET_KEY=변경하세요-랜덤-문자열
WTF_CSRF_SECRET_KEY=변경하세요-랜덤-문자열
SECURITY_PASSWORD_SALT=변경하세요-랜덤-문자열
UPLOAD_FOLDER=uploads
MODELS_FOLDER=model
```

> **참고:** `python -c "import secrets; print(secrets.token_hex(32))"` 명령으로 안전한 키를 생성할 수 있습니다.

### DB 초기화

```bash
flask --app run.py db init
flask --app run.py db migrate -m "init"
flask --app run.py db upgrade
```

> 이미 `migrations/` 폴더가 있다면 `flask --app run.py db upgrade` 만 실행합니다.

### 서버 실행

```bash
python run.py
```

브라우저에서 [http://127.0.0.1:5000](http://127.0.0.1:5000) 접속

## 사용법

1. **회원가입** — 우측 상단 메뉴 → Sign Up
2. **이미지 감지**
   - 상단 메뉴 → 이미지 감지 → 이미지 업로드
   - 업로드된 이미지 클릭 → 모델 선택 후 감지 실행
3. **동영상 감지**
   - 상단 메뉴 → 동영상 감지 → 동영상 업로드
   - 업로드된 영상 클릭 → 모델 선택 후 감지 실행 (백그라운드 처리)
   - 처리 완료 후 결과 영상 확인

## 프로젝트 구조

```
vision-project/
├── run.py                  # 앱 진입점
├── pyproject.toml          # 패키지 설정
├── .env                    # 환경 변수 (직접 생성)
├── model/                  # YOLO 모델 파일 (.pt)
├── uploads/                # 업로드 파일 저장소 (자동 생성)
└── src/
    ├── __init__.py         # Flask 앱 팩토리
    ├── config/             # 환경별 설정
    ├── templates/          # 공통 템플릿 (base.html, 인증 페이지)
    ├── static/             # 정적 파일 (CSS, 영상, 이미지)
    └── domains/
        ├── auth/           # 인증 모델
        ├── user/           # 사용자 모델
        ├── root/           # 메인 페이지
        └── detect/         # 감지 기능 (핵심)
            ├── detector/   # YOLO 모델 래퍼
            ├── models.py   # DB 모델
            ├── views.py    # 라우트
            └── templates/  # 감지 관련 템플릿
```

## 기술 스택

- **Backend** — Flask, Flask-Security, Flask-SQLAlchemy, Flask-Migrate
- **AI** — Ultralytics YOLO, OpenCV
- **Database** — SQLite
- **Frontend** — Bootstrap 5, Jinja2

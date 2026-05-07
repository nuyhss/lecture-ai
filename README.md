# lecture-ai

강의 영상 1개를 넣으면 `WhisperX`로 전사하고, timestamp가 포함된 `JSON`을 생성하는 1차 MVP입니다.

지금 단계의 목표는 딱 하나입니다.

`video -> wav -> transcript.json`

## 1순위 로드맵

1. 음성 추출
2. WhisperX 전사
3. 결과 JSON 확인
4. chunk 분리
5. 요약/태깅

현재 저장소는 `2번까지` 바로 검증할 수 있도록 구성했습니다.

## 폴더 구조

```text
lecture-ai/
├─ data/
│  ├─ input/
│  ├─ audio/
│  └─ output/
├─ src/
│  ├─ extract_audio.py
│  ├─ transcribe.py
│  └─ main.py
├─ requirements.txt
└─ README.md
```

## 환경 세팅

### 1. 가상환경

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
```

### 2. PyTorch 설치

CUDA 환경에 맞는 명령으로 설치하세요. 예시는 다음과 같습니다.

```powershell
pip install torch torchvision torchaudio
```

### 3. 프로젝트 패키지 설치

```powershell
pip install -r requirements.txt
```

### 4. CUDA 확인

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

`True`가 나오면 GPU 사용 준비가 된 상태입니다.

## FFmpeg 설치

이 프로젝트는 `ffmpeg` 실행 파일이 시스템 PATH에 있어야 합니다.

아직 이 PC에서는 `ffmpeg` 명령이 잡히지 않는 상태였습니다. 설치 후 아래 명령이 동작해야 합니다.

```powershell
ffmpeg -version
```

## 실행 방법

### 1. 입력 영상 배치

입력 영상을 아래 경로에 넣습니다.

```text
data/input/lecture.mp4
```

파일명이 다르면 실행 시 `--input`으로 직접 넘기면 됩니다.

### 2. 음성만 추출

```powershell
python src/extract_audio.py --input data/input/lecture.mp4 --output data/audio/lecture.wav
```

성공하면 `data/audio/lecture.wav`가 생성됩니다.

### 3. WhisperX 전사 + JSON 저장

```powershell
python src/transcribe.py --input data/audio/lecture.wav --output data/output/transcript.json --model base --language ko
```

이 스크립트는 다음을 수행합니다.

1. WhisperX 전사
2. 정렬(alignment) 수행
3. timestamp 포함 결과를 JSON으로 저장
4. segment별 `start end text`를 터미널에 출력

### 4. 한 번에 실행

```powershell
python src/main.py --input data/input/lecture.mp4 --audio-output data/audio/lecture.wav --json-output data/output/transcript.json --model base --language ko
```

## 출력 예시

성공하면 콘솔에 이런 형태가 나옵니다.

```text
1422.10 1490.30 CNN은 이미지 처리에 특화된 구조입니다
```

그리고 아래 파일이 생성됩니다.

```text
data/output/transcript.json
```

JSON에는 `segments`가 포함되고, alignment가 성공하면 `words` 단위 timestamp도 들어갑니다.

예시 구조:

```json
{
  "segments": [
    {
      "start": 1422.1,
      "end": 1490.3,
      "text": "CNN은 이미지 처리에 특화된 구조입니다",
      "words": [
        {
          "word": "CNN은",
          "start": 1422.1,
          "end": 1423.0
        }
      ]
    }
  ]
}
```

## 현재 단계에서 중요한 점

- `WhisperX + timestamp JSON 확보`가 최우선입니다.
- 이 단계가 끝나야 `23:42 점프`, `요약`, `태깅`, `중요 구간 추출`이 가능해집니다.
- 아직은 `Spring Boot`, `DB`, `UI`는 붙이지 않습니다.

## 다음 단계

이 단계 검증이 끝나면 바로 다음으로 넘어가면 됩니다.

1. chunk 분리
2. 구간 요약
3. 태깅
4. 중요 구간 자동 추출

## openAI API를 연결해 파인튜닝한 목소리 모델로 AI를 만들어보는 프로젝트
#### [여기](https://sesang06.tistory.com/216)에서 기본적인 내용을 공부하고 템플릿을 복사해서 사용했다.
#### GPT 학습은 [여기](https://github.com/ered1228/AI_Frieren)에서 진행.

## 개인 설정
#### 프로토타입을 만든 후 기억에 의존해 쓰는 거라 빠진게 있을 수 있고, 불필요한게 있을 수도 있습니다.
#### Python 3.10.0rc1버전 [설치](https://www.python.org/ftp/python/3.10.0/python-3.10.0rc1-amd64.exe)
#### pytorch가 gpu를 사용 가능한지 [확인](https://like-grapejuice.tistory.com/401)
``` python
nvidia-smi # CUDA 버전 확인
nvcc --version # CUDA TOOLKIT 확인
python -c "import torch; print(torch.cuda.is_available())" # False가 나오는 경우 자신의 NVIDIA가 지원하는 버전의 CUDA 설치.
# 11.8 버전의 CUDA는 둘 중 한가지 방법으로 설치
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118
```

### numpy와 tensorflow
``` python
pip install numpy==1.22.0
# tensorflow는 설치했었는지 안했었는지 생각이 안남
pip install --upgrade onnxscript
```

### ffmpeg
#### ffmpeg 프로그램 [설치](https://onlytojay.medium.com/%ED%8C%8C%EC%9D%B4%EC%8D%AC%EC%9C%BC%EB%A1%9C-ffmpeg-a0f1b3fae819)
``` python
pip install ffmpeg-python
```

### 나머지
``` python
pip install -U openai-whisper
pip install openai
pip install pyaudio wave
pip install pyaudio
pip install tts mecab-python3 cutlet unidic-lite
pip install coqui-tts
pip install subprocess
pip install -U ultralytics
pip install gruut
pip install librosa
```

## 발견한 오류 목록
- FileNotFoundError: [WinError 2] 지정된 파일을 찾을 수 없습니다
  - ffmpeg가 제대로 설치가 되지 않았거나, 파일경로가 올바르지 못한경우
- PermissionError: [WinError 32] 다른 프로세스가 파일을 사용 중이기 때문에 프로세스가 액세스할 수 없습니다.
  - num_loader_workers>0 이상인 경우 생기는 오류. 여러 일꾼이 한 파일을 수정하면서 생기는 오류
- ailed to import transformers.models.gpt2.modeling_gpt2 because of the following error (look up to see its traceback)
  - 해결했었는데 까먹음
- eval_split_size가 0.5 미만이면 생기는 오류.
- ModuleNotFoundError: No module named 'deepspeed'
  ```python
  pip install deepspeed
  ```
  - deepspeed note: This is an issue with the package mentioned above, not pip.
    ```python
    pip install https://github.com/daswer123/deepspeed-windows/releases/download/13.1/deepspeed-0.13.1+cu118-cp310-cp310-win_amd64.whl
    ```
      - ImportError: cannot import name 'log' from 'torch.distributed.elastic.agent.server.api'
        - 확인해본 결과 logger에서 log로 변수 이름이 변경되서 오류가 생김.
        -  python_path\lib\site-packages\torch\distributed\elastic\agent\server\api.py에서 모든 logger를 log로 수정
      - ImportError: cannot import name '_get_socket_with_port' from 'torch.distributed.elastic.agent.server.api'
        - 현재  
        -  python_path\Lib\site-packages\deepspeed\elasticity\elastic_agent.py에서 _get_socket_with_port import 부분을 from torch.distributed.elastic.utils.distributed import get_free_port로 변경해봄.
        - 그 결과 pytorch 2.2 버전으로 변경하라는 런타임 에러 발생.
       
## 2024-10-31
#### deepspeed 라이브러리 설치vidia-smi # CUDA 버전 확인
```
nvcc --version # CUDA TOOLKIT 확인
python -c "import torch; print(torch.cuda.is_available())" # False가 나오는 경우 자신의 NVIDIA가 지원하는 버전의 CUDA 설치.
# 11.8 버전의 CUDA는 둘 중 한가지 방법으로 설치
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118
```

### numpy와 tensorflow
``` python
pip install numpy==1.22.0
# tensorflow는 설치했었는지 안했었는지 생각이 안남
pip install --upgrade onnxscript
```

### ffmpeg
#### ffmpeg 프로그램 [설치](https://onlytojay.medium.com/%ED%8C%8C%EC%9D%B4%EC%8D%AC%EC%9C%BC%EB%A1%9C-ffmpeg-a0f1b3fae819)
``` python
pip install ffmpeg-python
```

### 나머지
``` python
pip install -U openai-whisper
pip install openai
pip install pyaudio wave
pip install pyaudio
pip install tts mecab-python3 cutlet unidic-lite
pip install coqui-tts
pip install subprocess
pip install -U ultralytics
pip install gruut
pip install librosa
```

## 발견한 오류 목록
- FileNotFoundError: [WinError 2] 지정된 파일을 찾을 수 없습니다
  - ffmpeg가 제대로 설치가 되지 않았거나, 파일경로가 올바르지 못한경우
- PermissionError: [WinError 32] 다른 프로세스가 파일을 사용 중이기 때문에 프로세스가 액세스할 수 없습니다.
  - num_loader_workers>0 이상인 경우 생기는 오류. 여러 일꾼이 한 파일을 수정하면서 생기는 오류
- ailed to import transformers.models.gpt2.modeling_gpt2 because of the following error (look up to see its traceback)
  - 해결했었는데 까먹음
- eval_split_size가 0.5 미만이면 생기는 오류.
- ModuleNotFoundError: No module named 'deepspeed'
  ```python
  pip install deepspeed
  ```
  - deepspeed note: This is an issue with the package mentioned above, not pip.
    ```python
    pip install https://github.com/daswer123/deepspeed-windows/releases/download/13.1/deepspeed-0.13.1+cu118-cp310-cp310-win_amd64.whl
    ```
      - ImportError: cannot import name 'log' from 'torch.distributed.elastic.agent.server.api'
        - 확인해본 결과 logger에서 log로 변수 이름이 변경되서 오류가 생김.
        -  python_path\lib\site-packages\torch\distributed\elastic\agent\server\api.py에서 모든 logger를 log로 수정
      - ImportError: cannot import name '_get_socket_with_port' from 'torch.distributed.elastic.agent.server.api'
        - 현재  
        -  python_path\Lib\site-packages\deepspeed\elasticity\elastic_agent.py에서 _get_socket_with_port import 부분을 from torch.distributed.elastic.utils.distributed import get_free_port로 변경해봄.
        - 그 결과 pytorch 2.2 버전으로 변경하라는 런타임 에러 발생.
       
## 2024-10-31
#### deepspeed 라이브러리 설치
#### tts를 생성할때 콘솔창에서 뜨던 FutureWarning 필터링.

## 2024-10-30 
#### wav 파일 20개로 파인튜닝으로 목소리 모델 제작하여 프로토타입 완성.
#### 학습시킨 파일이 20개밖에 되지 않아 목소리가 좀 이상했음. System Instructions가 작성이 잘 안되어있어 답변이 좀 이상했음. 답변 생성과 출력이 10초 이상 걸렸음.
#### Todo) 학습시킬 wav 파일 증가. System Instructions 수정. 답변 속도를 줄이기 위한 대책 마련.
https://github.com/user-attachments/assets/e618babd-a6eb-4dc6-a093-fac75c9f66ca

## 2024-10-29
#### ChatGpt Pro가 종료됨에 따라 지금까지 작성한 System instructions를 볼 수가 없게 됨. 3만원이나 하는 Pro를 다시 구매하고 싶진 않아 이번엔 OpenAI를 결제하여 새로 다시 쓰기 시작.

# -Image-Inpainting-using-Latent-Diffusion
 Image Inpainting using Latent Diffusion

## 사진 속 같이 있는 전 애인을 지우자.

그때 사랑하던 우리의 사진을 보며 너무 사진 속 당신이 이쁘게 나왔습니다.
이 사진을 사용하고 싶지만 사용하기에는 옆에 그,그녀가 있습니다.
이를 해결하기 위해 Image Inapinting을 시도합니다!

## 모델 설명 

Latent Diffusion의 흐름

![image](https://github.com/kong276818/-Image-Inpainting-using-Latent-Diffusion/assets/106736474/b56c3c05-6626-481b-b3bb-03767dcf2451)


Latent Diffusion 모델이란 직역하면 잠재 확산이라는 의미로 Diffusion 과정이 우리가 흔히 알고 있는 픽셀 공간이 아닌 Latent 공간에서 작동하는 방식입니다.

Latent Diffuson은 처리해야 할 수가 매우 적어진다는 장점이 있습니다.
이 모델은 VAE의 U-Net에서 아이디어를 받아 만들어지게 되었습니다.

VAE (Variational Autoencoder)

픽셀의 공간에서 Latent 공간으로 이동하기 위해서는 VAE라는 것이 필요합니다.


VAE는 인코더, 디코더 두 부분으로 구성되며 인코더는 픽셀 공간의 이미지를 Latent 공간으로 압축시키는 역할을 하며, 디코더는 Latent 공간의 이미지 표현을 픽셀 공간의 이미지로 복원하는 역할을 합니다.

![image](https://github.com/kong276818/-Image-Inpainting-using-Latent-Diffusion/assets/106736474/5e62af95-404a-4207-aef7-6354409a05fa)


U-Net은 오토인코더(autoencoder)와 같은 인코더-디코더(encoder-decoder) 기반 모델에 속한다. 보통 인코딩 단계에서는 입력 이미지의 특징을 포착할 수 있도록 채널의 수를 늘리면서 차원을 축소해 나가며, 디코딩 단계에서는 저차원으로 인코딩된 정보만 이용하여 채널의 수를 줄이고 차원을 늘려서 고차원의 이미지를 복원한다. 하지만 인코딩 단계에서 차원 축소를 거치면서 이미지 객체에 대한 자세한 위치 정보를 잃게 되고, 디코딩 단계에서도 저차원의 정보만을 이용하기 때문에 위치 정보 손실을 회복하지 못하게 됩니다.

![image](https://github.com/kong276818/-Image-Inpainting-using-Latent-Diffusion/assets/106736474/15babd84-e316-4c0e-bb56-19b484d1e7f0)

U-Net의 기본 아이디어는 저차원 뿐만 아니라 고차원 정보도 이용하여 이미지의 특징을 추출함과 동시에 정확한 위치 파악도 가능하게 하자는 것 입니다. 이를 위해서 인코딩 단계의 각 레이어에서 얻은 특징을 디코딩 단계의 각 레이어에 합치는(concatenation) 방법을 사용한다. 인코더 레이어와 디코더 레이어의 직접 연결을 스킵 연결(skip connection)이라고 합니다.

![image](https://github.com/kong276818/-Image-Inpainting-using-Latent-Diffusion/assets/106736474/a4064995-ce53-4f37-a39a-87c8d0f363ec)

![image](https://github.com/kong276818/-Image-Inpainting-using-Latent-Diffusion/assets/106736474/014f19eb-f24a-44b3-9a22-7ddfbf22d695)

U-Net 은 인코더 또는 축소경로(contracting path)와 디코더 또는 확장경로(expending path)로 구성되며 두 구조는 서로 대칭적이다. 인코더와 디코더를 연결하는 부분을 브릿지(bridge)라고 합니다.


DDPM(Denoising Diffusion Probabilistic Model)

Diffusion model에는 여러 알고리즘 방법이 있습니다. 
대표적으로 DDPM의 방식을 따르고 있습니다.
DDPM은 대표적으로 markov chain을 따르며 Xt-1작업을 하는 데 있어 Xt만 따르게 되며 다음 스탭을 밟아도 전 스탭의 값에 영향을 받게 됩니다.


![image](https://github.com/kong276818/-Image-Inpainting-using-Latent-Diffusion/assets/106736474/39d1b05a-32ea-4e82-9be8-3e9dd91ba10f)


DDIM(Denoising Diffusion Implicit Models)

DDIM같은 경우는 DDPM과 다르게 markov chain을 따르지 않게 됩니다. 
이를 통해 Xt를 생성하는 과정에 있어 전 스탭 Xt-1과 X0를 모두 사용하게 됩니다. 
이를 통해 더욱 빠른 속도를 통해 생성을 할 수 있게 됩니다.


![image](https://github.com/kong276818/-Image-Inpainting-using-Latent-Diffusion/assets/106736474/b80bbfa6-55b4-432d-bc7c-f7f8976db95b)

## 실행방법

# Pretrained model https://heibox.uni-heidelberg.de/f/4d9ac7ea40c64582b7c9/?dl=1 

위에서 학습 모델을 다운로드 하여 models안에 ldm의 Inpainting big안에 넣습니다.

다음

Inpaint.py에서 IMG_PATH에 원하는 사진을 넣고 실행합니다.

STEPS = 50
IMG_PATH = 'data/09.jpg'
OUTPUT_PATH = 'outputs'



## 요구사항 

python 3.8.5


name: ldm
channels:
  - pytorch
  - defaults
dependencies:
  - python=3.8.5
  - pip=20.3
  - cudatoolkit=11.0
  - pytorch=1.7.0
  - torchvision=0.8.1
  - numpy=1.19.2
  - pip:
    - albumentations==0.4.3
    - opencv-python==4.1.2.30
    - pudb==2019.2
    - imageio==2.9.0
    - imageio-ffmpeg==0.4.2
    - pytorch-lightning==1.6.1
    - omegaconf==2.1.1
    - test-tube>=0.7.5
    - streamlit>=0.73.1
    - einops==0.3.0
    - torch-fidelity==0.3.0
    - transformers==4.3.1
    - -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
    - -e git+https://github.com/openai/CLIP.git@main#egg=clip
    - -e .
   
    






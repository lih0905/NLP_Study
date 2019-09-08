---
title: "FastText (1) - Character n-gram vectors"
date: 2019-09-03 01:48:28 -0400
categories: NLP
tags:
  - fasttext
  - character n-gram
  - word vectors

use_math: true
toc: true
---

## 개요

<center>
<img src='https://fasttext.cc/img/fasttext-logo-color-web.png'>
<br>
[img source : https://fasttext.cc/img/fasttext-logo-color-web.png]
</center>
<br>

Facebook AI 랩에서 만든 오픈소스 라이브러리 [fastText](http://https://fasttext.cc/)는 단어 표현 및 텍스트 분류 등의 기능이 있으며, 무엇보다 굉장히 빠르고 가벼운 것으로 유명하다. 이 라이브러리의 이론적 배경이 되는 논문은 다음 세 편이다.

1. [Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606), Bojanowski et al.

2. [Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759), Joulin et al.

3. [FastText.zip: Compressing text classification models](https://arxiv.org/abs/1612.03651), Joulin et al.

이 글에서는 이 중 첫 번째 논문에 대해 이해해보려고 한다. 이 논문에서는 단어의 벡터 표현에 대해 논의하고 있으며, 기존에 소개한 [Word2Vec (Skip-gram)](https://lih0905.github.io/nlp/Word2vec/) 모델을 '부분 단어'라는 개념을 도입하여 개선한다. 이를 통해 단어의 형태적 특성 학습(worry : worries, text : texts 등 유사한 형태의 단어는 유사한 의미를 가진다고 파악하는 것) 및 기존 단어장에 등장하지 않는(Out of vocabulary) 단어의 의미 또한 파악할 수 있게 된다.

## Skip-gram 모델의 형태 복습

이 논문에서는 우선 기존 skip-gram 및 네거티브 샘플링에 대하여 복습한 후 이를 일반적인 형태로 표현한다. 단어의 갯수가 $W$인 말뭉치가 주어져있다고 하자. 단어 표현이란 단어 $i$(=$i$번째 단어와 동일시)에 대한 적절한 벡터 표현 $w_i \in \mathbb{R}^d$ 을 구하는 것이다. 이를 위해서는 말뭉치가 $w_1, \ldots, w_T$ 라고 주어져있을 때, 다음 손실 함수를 최소화하는 벡터 표현을 구하고자 한다.

$$
-\sum_{t=1}^T \sum_{c \in \mathbf{C}_t} \log p(w_c \vert w_t)
$$

여기서 $\mathbf{C}_t$는 단어 $w_t$의 근방 단어를 모은 집합이다. 여기서 확률 $p(w_c \vert w_t)$를 정의하기 위해, 두 단어에 대한 점수를 평가하는 함수 $s$가 주어져 있다고 가정하자. 이 때, 일반적으로 확률 $p(w_c \vert w_t)$는 소프트맥스 함수를 이용하여 다음과 같이 정의한다.

$$
p(w_c \vert w_t) := \frac{\exp({s(w_t, w_c)})}{\sum_{j=1}^W \exp(s(w_t, w_j))}
$$

또한 이 [포스트](https://lih0905.github.io/nlp/Word2vec_2/)에서 소개한 네거티브 샘플링을 도입하면, 고정된 중심 단어 $w_t$와 문맥 단어 $w_c$에 대한 손실 함수는 다음과 같이 표현된다.

$$
-\log \left( 1+e^{-s(w_t, w_c)} \right)  -  \sum_{n \in {\mathbf{N}_{t} } }\log \left( 1+e^{s(w_t, w_n)} \right) 
$$

여기서 ${\mathbf{N}_{t}}$는 $w_t$의 문맥에 등장하지 않는 네거티브 샘플들을 모은 집합이다. 따라서 로지스틱 손실 함수 $\ell : x \mapsto \log(1+e^{-x})$ 을 생각하면 skip-gram 모델의 손실 함수는 다음과 같이 표현할 수 있다.

$$
-\sum_{t=1}^T \left[\sum_{c \in {\mathbf{C}_{t} } }\log  \ell(s(w_t, w_c))   +  \sum_{n \in {\mathbf{N}_{t} } }\log  \ell(-s(w_t, w_n))  \right]
$$

일반적인 skip-gram 모델에서는 두 단어 벡터의 점수 함수를 내적으로 정의한다.

## 부분 단어 모델

Skip-gram 모델은 하나의 단어에 대해 하나의 벡터 표현만을 생각하므로, 단어의 내부적 구조는 포착할 수가 없다. 따라서 이 논문에서는 단어의 내부 구조까지 반영한  새로운 점수 함수 $s$를 도입하고자 한다. 이는 단어 $w$를 문자 $n$-그램(부분 단어)의 집합으로 표현한 후 각 부분 단어에 대응되는 벡터 표현을 사용하여 이루어진다.

이 과정을 단어 $w$= `where`를 통해 이해해보자. 먼저 주어진 단어의 양 끝에 < 와 > 를 붙여 `<where>`라고 쓴다. 이후 이 문자열에서 차례대로 $n$개의 문자열을 다음과 같이 골라 집합 $\mathcal{G}_w$를 구성한다.

|$n$| 부분 단어|
|:--:|:--:|
|3|`<wh`, `whe`, `her`, `ere`, `re>`|
|4|`<whe`,`wher`,`here`,`ere>`|
|5|`<wher`,`where`,`here>`|

그리고 마지막으로 원래의 문자열 `<where>` 또한 $\mathcal{G}_w$에 추가한다. 이 논문에서는 $n$이 $3$에서 $6$인 부분 단어를 모두 고려했다고 한다. 따라서 $w$=`where`인 경우 부분 단어 집합 $\mathcal{G}_w$은 다음과 같다.

$$
\mathcal{G}_w = \{ 
    \text{<wh, whe, her, ere, re>,
    <whe,wher,here,ere>,
    <wher,where,here>,where} 
    \}
$$

이후 $\mathcal{G}_w$의 각 $n$-gram $g$에 대하여 벡터 표현 $z_g$을 정의한 후, 원래 단어 $w$에 대응되는 단어 표현을 $ \sum_{g \in \mathcal{G}_w} z_g $ 라고 정의한다. 그러면 부분 단어를 이용한 두 단어 $w$와 $c$ 사이의 점수 함수를 다음과 같이 자연스럽게 정의할 수 있다.

$$
s(w,c) := \sum_{g \in \mathcal{G}_w} z_g^t w_c
$$


위와 같은 방법으로 모든 단어에 대해 부분 단어 집합을 구한 후, 앞서 정의한 skip-gram 모델의 손실 함수에서 점수 함수 부분만 위 식으로 변경한 것이 바로 부분 단어 모델이다.

이렇게 정의한 부분 단어 모델은 단어 전체의 형태 및 의미 뿐 아니라 단어를 구성하는 형태소의 의미 또한 파악할 수 있게 되므로, 단어의 형태적 특성까지 담아낼 수 있다. 또한, 훈련 텍스트에 등장하지 않는 단어(tech-rich)라 하더라도 부분 단어(tech, rich)의 의미를 통해 전체의 의미를 유추할 수 있게 된다.




## 참고 자료

1. [[Enriching Word Vectors with Subword Information], Bojanowski et al. (2017)](https://arxiv.org/pdf/1301.3781)

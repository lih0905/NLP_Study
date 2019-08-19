# W5. Word2Vec

## Distributed Representations of Words and Phrases and Their Compositionality - Mikolov et al (2014)
#### '19.8.19 edited.

### 개요
이번주 스터디에서는 단어의 벡터화의 가장 기초인 Word2Vec 모델 및 그 개선법에 대해 공부한다. Word2Vec은 `T. Mikolov` 및 그 외 연구진에 의해 쓰인 `Efficient Estimation of Word Representations in vector space (2013)` 논문에서 처음 소개되었으며, 본 논문에서는 Word2Vec 모델을 개선하는 방안에 대해 논의한다. Word2Vec 모델은 CBOW(Continuous Bag of Words)와 Skip-gram 방식으로 나뉘며, 일반적으로 Skip-gram 방식이 더 성능이 좋기에 널리 사용된다. 본 글에서는 Skip-gram 모델에 한정하여 논의하고자 한다.

단어를 벡터화하는 가장 간단한 방법은 원-핫-벡터를 이용하여 각 단어를 단위 벡터($\mathbf{e}_w$)로 표기하는 것이다. 그러나 이 방식은 단어의 갯수만큼 벡터의 차원이 커지게 되며, 동시에 벡터의 대부분이 $0$으로 채워져 메모리를 효율적으로 사용하지 못하는 단점이 있다. 또한 각 단어 벡터간 내적이 $0$이므로 단어 사이의 관계에 대한 정보를 전혀 얻지 못하게 된다. 따라서 단어를 밀도 높은 벡터(dense vector)로 나타내는 모델을 개발하고자 하는 연구가 계속 되어 왔다. 

<img src = 'https://wikidocs.net/images/page/22660/%EC%8A%A4%ED%82%B5%EA%B7%B8%EB%9E%A8.PNG'>

Skip-gram은 단어들을 정해진 차원의 벡터 공간에 임베딩하는 모델이다(일반적으로 벡터 공간의 차원 << 단어의 갯수). 먼저 임의의 값으로 벡터들을 초기화한 후, 주어진 텍스트 데이터를 토대로 특정 단어가 주어졌을 때 그 주변 단어들의 등장 확률을 증가시키는 방향으로 학습하는 알고리즘이다. 가령 `I love him but he hates me.` 라는 문장을 생각해보자. 여기서 `him`이라는 단어를 기준으로 앞 뒤 두 단어들인 `I`,`love`,`but`,`he`의 발생 확률을 증가시키는 방향으로 학습하게 된다.

### Skip-gram 모델의 구조
주어진 훈련 텍스트에 등장하는 단어들의 집합을 `Vocab`이라고 하고, 이 집합의 크기를 `K`라고 하자. 또한 이 모델의 hyperparameter로 다음 두 가지가 주어진다. 중심 단어를 기준으로 몇 번째 단어까지 고려할 지 결정해야 하며, 이를 `WINDOW SIZE`라고 한다. 또한 단어들을 임베딩할 벡터공간의 차원(`D`)을 결정해야 한다. 

먼저 크기가 ($K$, $D$) 인 행렬 $V, U$를 임의값으로 초기화하자. $i$번째 단어 $w_i \in \text{Vocab}$에 대해 먼저 입력 벡터(input vector)를 $v_{i} = [V]_i \in \mathbb{R}^D$ , 출력 벡터(output vector)를 $u_{i} = [U]_i \in \mathbb{R}^D$라고 정의한다(행렬 $A$ 에 대해 $i$번째 행/열은 각각 $[A]_i$ 와 $[A]^i$로 표기). 

임의의 중심 단어 $c$에 대해($c$번째 단어), 점수 벡터(score vector) $z = U \cdot v_c$ 를 계산한 후 이 벡터에 소프트맥스를 취해 확률 벡터 $\hat{y} = \text{softmax}(z)$ 를 얻는다. 이를 정리하면, 중심 단어 $c$에 대해 단어 $o$가 문맥에 발생할 확률은 다음과 같다.
$$
p(o|c) := \frac{\exp(u_o^t v_c)}{\sum_{i=1}^{K}\exp(u_i^t v_c)}
$$

주어진 텍스트가 $\{ w_1, w_2, \ldots, w_T\}$라고 토큰화되어 있을 때, 우리의 목표는 다음 값을 최소화하는 것이다($m$은 WINDOW SIZE).
$$
J(\theta) :=- \frac{1}{T} \sum_{t=1}^{T} \sum_{j=0, j\ne m}^{2m} \log p(w_{t+j}|w_t)
$$

중심 단어 $w_c$만을 고려할 때,
$$
\begin{array}{lcl}
 J &=& - \log p(w_{c-m}, \ldots, w_{c-1},w_{c+1},\ldots, w_{c+m})\\
 & =& - \log \prod_{j=0, j\ne m}^{2m} p(w_{c-m+j}|w_c) \\
 & =& - \log \prod_{j=0, j\ne m}^{2m} \frac{\exp(u_{c-m+j}^t v_c)}{\sum_{i=1}^{K}\exp(u_i^t v_c)} \\
 &=&-\sum_{j=0, j\ne m}^{2m} u_{c-m+j}^t v_c + 2m \log\sum_{i=1}^{K}\exp(u_i^t v_c) 
 \end{array}
$$가 되며, 이는 cross-entropy $\sum_{j=0, j\ne m}^{2m} H(\hat{y}, y_{c-m+j})$ 와 같다.

이로부터 계산을 통해 다음을 확인할 수 있다.
$$
\begin{array}{lcl}
\frac{\partial{J}}{\partial{v_c}} &=& - \sum_j {u_j} + 2m \sum_{w=1}^K \frac{\exp(u_w^t v_c)}{\sum_{i=1}^K \exp(u_i^tv_c)}u_w \\
&=&  - \sum_j {u_j} + 2m \sum_{i=1}^K p(w|c) u_w,\\
\frac{\partial{J}}{\partial{u_o}} &=& -v_c + p(o|c) v_c \quad (o \text{ is an context vector}),\\
\frac{\partial{J}}{\partial{u_w}} &=& p(w|c) v_c \quad (w \text{ is not an context vector}) .
\end{array}
$$ 이를 통해 그래디언트 업데이트, 즉 학습을 수행할 수 있다. 

### Skip-gram모델의 장단점
잘 학습된 Word2Vec 모델의 경우 각 단어에 대응되는 벡터는 단어들 사이의 형태적, 의미적 관계를 모두 보존하게 된다. 예를 들면 다음 대응들을 내포하고 있다.

* do : did = play : ?? 
	* 이 경우는 단어의 형태(현재형:과거형)에 대한 추론으로 played를 답한다. 
* king : man = queen : ?? 
	* 이 경우는 단어의 의미(권위/성별)에 대한 추론으로 woman을 답한다.

따라서 Word2Vec 모델을 사용하면 단어들을 훨씬 작은 공간에 임베딩하여 메모리를 효율화하는 것 뿐 아니라, 단어들 사이의 관계까지 모델링할 수 있다는 장점이 있다. 그러나 이런 vanilla skip-gram 모델의 경우 치명적인 단점이 있다. 실제 언어를 모델링하는 경우 단어의 갯수 $K$가  $10^5$ ~ $10^7$ 정도로 굉장히 큰 편인데, 단어 벡터 계산 및 그래디언트 계산 시마다 <b>소프트맥스 연산에 드는 계산량이 $K$에 비례</b>하기 때문에 굉장히 비효율적이다. 따라서 다음 기법들을 도입하여 계산량을 줄이고자 한다.

### Hierarchical Softmax
계층형 소프트맥스(hierarchical softmax)는 이진 트리를 사용하여 모든 단어를 표현한다. 트리의 각 잎(leaf, 트리의 말단)는 단어에 해당하며, 뿌리(root)로부터 각 잎까지는 유일한 경로(path)로 연결된다. 이 모델은 단어의 입력 벡터($v_i$)만 존재하며, 대신 트리의 각 마디에 대응되는 벡터가 존재하여 이를 훈련시키도록 한다. 이 모델에서  단어 $w$가 출력으로 선택될 확률은 뿌리에서 해당 단어에 대응되는 잎까지 랜덤 워크로 도달할 확률과 같다.
<img src ='https://shuuki4.files.wordpress.com/2016/01/hsexample.png'>
$L(w)$를 뿌리로부터 단어 $w$까지 도달하는 경로의 길이라고 하고, $n(w,i)$를 뿌리부터 단어 $w$ 사이의 경로(유일함!) 중 $i$번째 마디라고 정의하자. 따라서 $n(w,1)$은 뿌리에 해당하고, $n(w,L(w))$는 단어 $w$ 자신이다. 내부 마디 $n$마다 자식 하나를 임의로 고정하고 이를 $ch(n)$이라 표기하자. 그러면 중심 단어 $w_i$의 문맥에 단어 $w$가 등장할 확률은 다음과 같이 정의된다:
$$
p(w|w_i) =\prod_{j=1}^{L(w)-1} \sigma([n(w,j+1) = ch(n(w,j))]\cdot {v'_{n(w,j)}}^t v_{w_i}).
$$ 여기서 $[x]$는 $x$가 참일 때 $1$, 거짓일 때 $-1$로 주어지는 함수이며 $\sigma$는 시그모이드 함수이다.  그리고 $v_{w_i}$는 단어 $w_i$의 입력 벡터, $v'_{n(w,j)}$는 내부 마디의 벡터 표현이다.

시그모이드 함수는 임의의 $x$에 대해 $\sigma(x) + \sigma(-x) = 1$ 을 만족한다. 따라서 임의의 마디 $n$에서 $\sigma({v'_{n}}^t v_{w_i}) + \sigma(-{v'_{n}}^t v_{w_i}) = 1$ 이 성립한다. 이로부터 다음 식을 얻을 수 있다.
$$
\sum_{w=1}^{K} p(w|w_i) = 1
$$ 따라서 $p(w|w_i)$는 확률분포를 이루게 되며, 손실 함수는 $-\log(p(w|w_i))$ 로 주어진다. 다만 여기서는 출력 벡터를 업데이트하는 것이 아니라 트리의 내부 마디들의 벡터를 업데이트하게 된다. 이렇게 정의된 모델은 계산량이 $\log_2 (K)$에 비례하므로 기존 소프트맥스 모델에 비해 획기적으로 줄어들게 된다. 

본 논문에서는 Huffman 트리를 이용하였다고 한다. Huffman 트리는 가장 빈도가 낮은 단어부터 두개씩 묶어가면서 전체 단어를 연결하여 생성한다. 따라서 가장 빈도가 높은 단어가 가장 마지막에 묶이게 되므로 트리의 뿌리로부터 가깝게 생성된다.

### Negative Sampling
네거티브 샘플링은 계층형 소프트맥스 대신 계산량을 줄이기 위해 사용할 수 있는 방법이다. 매 훈련 스텝마다 모든 단어에 대해 연산을 수행하는 대신 일부 부정적 예제를 샘플링하여 사용하는 방법이다. 단어 $w$, $c$를 고려하자. 이제 $D$라는 문장에서  $(w,c)$가 $c$가 중심 단어, $w$가 그 문맥에 발생할 확률을 $p(D=1|w,c)$, 발생하지 않을 확율을 $p(D=0|w,c)$이라고 하자. 

먼저 시그모이드 함수를 이용하여 $p(D=1|w,c)$를 다음과 같이 정의하자:
$$
p(D=1|w,c) = \sigma(u_w^t v_c) = \frac{1}{1+\exp(-u_w^t v_c)}
$$ 그러면 모델링의 목표는 다음과 같다.

$$
\begin{array}{ll}
&\text{maximize} \log\left(\prod_{(w,c)\in D} p(D=1|w,c) \prod_{(w,c)\in \tilde{D}} p(D=0|w,c) \right) \\
= &\text{maximize} \log\left(\prod_{(w,c)\in D} p(D=1|w,c) \prod_{(w,c)\in \tilde{D}} (1-p(D=1|w,c)) \right) \\
= &\text{maximize} \left(\sum_{(w,c)\in D} \log p(D=1|w,c) +\sum_{(w,c)\in \tilde{D}} \log(1-p(D=1|w,c)) \right) \\
= &\text{maximize} \left(\sum_{(w,c)\in D} \log (\frac{1}{1+\exp(-u_w^t v_c)}) +\sum_{(w,c)\in \tilde{D}} \log(\frac{1}{1+\exp(u_w^t v_c)}) \right)
\end{array}
$$ 여기서  $\tilde{D}$는 말이 되지 않는 문장 등을 샘플링한 집합이다. 예를 들어 `stock boil fish is toy`와 같은 문장의 발생 확률은 매우 낮게 계산되어야 하기 때문에, 위 수식에서 해당 집합에 포함된 단어들은 발생 확률이 낮아지도록 설정하였다.

Skip-gram 모델에서 새로운 손실 함수는 다음과 같이 정의된다.
$$
-\log \sigma(u^t_{c-m+j}\cdot v_c) - \sum_{k=1}^{|\tilde{D}|} \log \sigma(-\tilde{u}^t_k \cdot v_c)
$$ 위 식에서 $\{\tilde{u}_k\}$들은 샘플링된 부정적 예제들이다. 이때, 각 단어들은 다음 확률 분포를 사용하여 샘플링하는 것이 효과적이라고 알려져있다.
$$
p(w_i) \sim \frac{f(w_i)^{3/4}}{\sum_j f(w_j)^{3/4}}
$$ 여기서 $f(w_j)$는 훈련 데이터 중 단어 $w_i$의 발생 빈도이다. 위와 같이 $3/4$ 승을 한 분포를 사용하는 이유는 드물게 발생하는 단어의 샘플링 비율을 높이기 위함이다.



### Subsampling on frequency
매우 큰 텍스트 데이터의 경우  `in`, `the`, `a` 등의 단어는 너무 자주 등장하지만 실제로는 별 의미를 담고 있지 않다. 예를 들어 `France`와 `Paris`가 같이 발생한 경우 모델의 훈련에 도움이 되지만 `France`와 `the`가 같이 발생한 경우는 큰 의미를 가지기 어렵다. 따라서 이런 단어의 발생 빈도에 따른 불균형을 해소하기 위해 서브샘플링 기법을 도입한다.

훈련 데이터로 주어진 단어 $w_i$의 발생 빈도가 $f(w_i)$인 경우, 훈련 데이터에서 해당 단어는 다음 확률로 제거하도록 한다:
$$
p(w_i) = 1 - \sqrt{\frac{t}{f(w_i)}}
$$ 여기서 $t$는 해당 확률을 적용하는 한계점으로, 보통 $10^{-5}$ 정도를 사용한다. 다시 말해, 발생 빈도가 $t$ 이하인 단어는 제거하지 않는다는 의미이다.

이런 공식을 적용하면 발생 빈도가 $t$ 이상인 단어는 크게 제거하면서도 발생 빈도의 순서 자체는 보존하는 장점이 있다. 실제로 이렇게 서브샘플링을 적용할 경우 훈련 속도뿐 아니라 정확성까지도 크게 향상되었다.




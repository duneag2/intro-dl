## 1. 파이썬 라이브러리에서 OS 패키지 사용하기: Q. 왜 여기서는 문법이 달라질까?

A. 기본적으로 코랩에서 사용하는 문법들, !ls, !pwd와 같은 명령어는 '쉘(shell)명령어'입니다. 주로 주피터 환경에서 많이 사용하는 명령어로 리눅스 기반의 운영 시스템 명령을 직접 수행할 수 있습니다. 
   반면, 파이썬의 os 라이브러리는 운영 체제와 관련된 여러 기능을 제공하는 모듈입니다. os 모듈을 사용하는 것은 파이썬 스크립트 내에서 운영 체제의 기능을 '프로그래밍 방식'으로 제어하고자 할 때 사용됩니다.
   주로 코드에서 반복문(for문), 조건문(if문) 등에 잘 녹아드는 os 라이브러리 함수들을 코드에 사용합니다. 
   하지만 코드가 복잡하면 쉘 별로 따로 실행할 수 있는 쉘 명령어를 통해서 조작하는 것이 편할 때도 있습니다.




## 2. 안으로는 내적 계산은 안 되는걸까?

A. dot_vector1 = vector1 @ vector2
   print(dot_vector1)

   dot_vector2 = vector1.dot(vector2)
   print(dot_vector2)

   이런 방법으로도 구할 수 있습니다.




## 3. shuffle = False: test 데이터의 경우 data shuffle하지 않음. -> Q. 왜일까?

A. 학습에서 셔플을 하는 이유를 살펴볼 필요가 있습니다. 만약 데이터를 섞지 않고 학습하게 되면 학습한 데이터가 서로 뭉쳐서 편향된 학습을 할 수도 있습니다. 
   예를 들어서 장미 1000장, 튤립 1000장의 사진을 학습 데이터로 사용한다고 가정하겠습니다. 이런 경우 셔플하지 않고 순서대로 학습하면 장미와 튤립을 제대로 구분하지 못합니다.
   즉, 무작위로 섞어서 순서 없이 학습을 진행해야 모델이 장미와 튤립을 구분하는 능력을 가지게 되는 것입니다.

   그런데 테스트의 경우에는 모델이 모두 학습된 후 모델의 파라미터를 고정시키고 진행하는 것이기 때문에 셔플을 하던 안하던 크게 의미가 없습니다. 그래서 일반적으로는 셔플을 하지 않습니다.




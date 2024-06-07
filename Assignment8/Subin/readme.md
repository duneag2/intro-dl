
Are you gonna upload your assignment after all this time?
Welcome~!

마지막 attention 모델 구현하는 문제에서 왜 그래프가 표시되지 않는 걸까요?:(

## code 수정
%matplotlib inline

output_words, attentions = evaluate(
    encoder1, attn_decoder1, "je suis trop froid .")
plt.matshow(attentions.numpy())

## Ans
plot을 하기 전에 %matplotlib inline 을 코드에 포함시켜주세요!

뒤에서 두번째 셀을 위와 같이 수정해주시면 plot결과를 확인할 수 있습니다.

일반적인 python 스크립트에서는 plot을 진행하면 별도의 창을 열어서 plot 결과물을 보여주는데, jupyter notebook에서는 별도의 창이 열리지 않습니다. 그래서 코드가 제대로 실행되어도 피규어가 생성되지 않은 것처럼 보입니다. 위 코드를 추가해서 셀 출력에 피규어를 포함시켜서 plot 결과물을 확인할 수 있도록 해주는 것입니다.

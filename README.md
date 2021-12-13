# loan-prediction
Binary classification with imbalanced data
- 고객의 채무 불이행 가능성 예측

## Description

### 분석개요
여신전문회사에서는 회사의 손실을 막기 위해 채무 불이행 고객을 사전에 예측하는 것이 매우 중요하다. 고객의 채무 불이행 가능성을 예측할 수 있다면 대출 심사 시 위험 고객을 사전에
방어할 수 있고, 기대출 고객에 대해서는 따로 채무 관리를 할 수 있기 때문이다. Kaggle 에 공유된 데이터를 사용하여 채무 불이행 고객을 예측하는 분류(Classification) 모델을 만들어보고자 한다. 데이터에는 고객을 설명해주는 11 개의 변수와 고객의 채무 불이행 여부(Risk_Flag: 0, 1)인 타겟변수가 포함되어 있다. 이를 활용하여 모델을 생성한 후, 모델의 성능을 검증해보고 모델의 결과를 바탕으로 어떤 특성이 고객 채무 불이행 가능성을 잘 설명해주는지 파악해본다. 분류 모델은 XGB 와 Random Forest 기법을 사용한다.

고객의 채무 불이행은 일반적으로 정상 납입 고객보다 매우 적은 비율이다. 해당 분석에서 사용한 데이터 또한 총 252,000 건의 데이터 중 채무 불이행 이력이 있는 고객의 수는 30,996 건(12.3%)으로 타겟변수의 비율이 매우 불균형하다. 불균형 데이터를 조정하지 않고 모델을 생성하게 되면 예측 정확도는 높지만 재현율이 낮아지게 된다. 채무 불이행 고객을 예측하는 경우에는 정상 고객을 정상(0)으로 분류하는 것 보다, 불이행 고객을 이상(1)으로 분류하는 것이 중요하기 때문에 정확도 보다는 재현율을 높이는 데 초점을 맞추어야 한다.  따라서 본 과제에서는 Undersampling 기법인 ENN 과 Oversampling 기법인 SMOTE 를 적용하여 불균형 데이터를 먼저 조정한 후에 분류 모델링을 진행한다. 

### Data 
Kaggle에 공유된 ‘Loan Prediction Based on Customer Behavior’활용
(https://www.kaggle.com/subhamjain/loan-prediction-based-on-customer-behavior

![image](https://user-images.githubusercontent.com/79688191/145768171-31a56ea2-ae16-48f1-a298-6cdd889fd05a.png)

### 불균형 데이터 조정_ENN, SMOTE

![image](https://user-images.githubusercontent.com/79688191/145768321-0414aa0e-974a-4d71-85a6-1887c63e74bd.png)



### XGB Classifier
![image](https://user-images.githubusercontent.com/79688191/145768378-3c953052-cc1c-4358-9b7d-09c2259f030e.png)


### Random Forest Classifier

![image](https://user-images.githubusercontent.com/79688191/145768413-4a378de3-6271-4109-a3b7-c442e3df593a.png)


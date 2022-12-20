#!/usr/bin/env python
# coding: utf-8

# # Modelo de Previsão de Sexo do Pacientes

# ## Test - Data Scientist (profile statistician)

# ### Amanda Cristina da Costa Guimaraes

# In[1]:


get_ipython().system('pip install graphviz')
get_ipython().system('graphviz')


# In[2]:


import pandas as pd
import numpy as np


# In[3]:


df = pd.read_csv('test_data_CANDIDATE.csv')


# Vamos ler o arquivo csv e transformar em um dataframe

# In[4]:


df['sex'] = df['sex'].replace('M', 0)


# In[5]:


df['sex'] = df['sex'].replace('m', 0)


# In[6]:


df['sex'] = df['sex'].replace('F', 1)


# In[7]:


df['sex'] = df['sex'].replace('f', 1)


# Vamos converter os dados de sexo para 0 e 1 sendo:
#     - masculino (M) = 0
#     - feminino (F) = 1

# In[8]:


df['sex'].value_counts()


# Após a transformação dos dados temos em nosso dataframe:
# - 92 pacientes do sexo masculino 
# - 196 pacientes do sexo feminino

# Vamos define os valores do dataframe "X" e "Y" sendo:
# - X = Cor do cabelo (hc) e Cor da pele (sk)
# - Y = Sexo (sex)

# In[9]:


x = df[['hc', 'sk']] 
y = df[['sex']]


# Verificando como está o x.

# In[10]:


x.head()


# Verificando como está o y.

# In[11]:


y.head()


# ### Para analisar essas classes eu vou precisar de aloritimos estimadores, e para isso existem as bibliotecas do Python como: sklearn
# 
# Queremos criar um mecanismo que consiga estimar a classe desse tipo de dado. Para isso, usaremos algorítimos de machine learning, data analysis e estatística que estão armazenados nas diversas bibliotecas do Python. A que usaremos se chama Scikit Learn.
# 
# Do módulo sklearn.svm, importaremos um estimador chamado LinearSVC. Posteriomente, entraremos em mais detalhes sobre alguns estimadores existentes e o que cada um é capaz de realizar. Por enquanto, usaremos esse estimador básico.

# In[12]:


from sklearn.svm import LinearSVC
model = LinearSVC()


# In[13]:


treino_x = x[:75]
treino_y = y[:75]


# ### Treina a IA

# In[14]:


from sklearn.svm import LinearSVC


# In[15]:


from sklearn.metrics import accuracy_score


# In[16]:


treino_x = x[:75]
treino_y = y[:75]
teste_x = x[75:]
teste_y = y[75:]
teste_y.shape

print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))


# Primeiramente, com df.shape verificaremos quantos elementos temos nos dados e o formato deles:

# In[17]:


df.shape


# Veremos que em nosso arquivo há 288 linhas e 18 colunas. Separaremos em média 25% para testar o algorítimo, e o restante (cerca de 75% dos dados) para o treinamento. Portanto, para treino_x, coletaremos os primeiros 75 elementos (treino_x = x[:75]). Podemos utilizar treino_x.shape para verificar se o número de elementos está de fato correto:

# In[18]:


treino_x = x[:75]
treino_y = y[:75]


# Para confirmarmos se as matrizes estão com a quantidade correta de elementos, acionaremos teste_y.shape. Por fim, registraremos essas informações imprimindo o tamanho (len()) de treino_x e treino_y utilizando print().

# In[30]:


treino_x = x[:75]
treino_y = y[:75]
teste_x = x[75:]
teste_y = y[75:]
teste_y.shape

print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))


# Para treinarmos e executarmos o algorítimo usaremos a motodologia  do sklearn.svm importaremos LinearSVC, e treinaremos o modelo com os dados treino_x e treino_y.

# In[31]:


from sklearn.svm import LinearSVC

modelo = LinearSVC()
modelo.fit(treino_x, treino_y)


# In[32]:


from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

modelo = LinearSVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print("A acurácia foi %.2f%%" % acuracia)


# ### Testa o Modelo

# In[22]:


previsões = modelo.predict(teste_x)


# 

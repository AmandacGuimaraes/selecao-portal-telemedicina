#!/usr/bin/env python
# coding: utf-8

#  # Test - Data Scientist (profile statistician)

# ## Análise descritiva dos dados da tabela 

# ### Amanda Cristina da Costa Guimarães

# 1) Analyzing the data

# In[14]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


dados = pd.read_csv('test_data_CANDIDATE.csv')


# In[3]:


dados


# In[12]:


dados.info()


# ### Analisando a idade minima e maxima do nosso grupo de estudo:

# In[17]:


sns.histplot(dados['age'], bins=30);


# In[4]:


dados.age.min()


# In[5]:


dados.age.max()


# In[6]:


print('De %s até %s anos' % (dados.age.min(),dados.age.max())) 


# ### Descrição dos dados da tabela:

# Na tabela podemos visualizar 288 pacientes sendo do sexo masculino e feminino, com idades variando de 29 a 77 anos, e dados medicos referentes a: tipo de dor no peito, pressão arterial de repouso, colesterol sérico, glicemia de jejum, resultados eletrocardiográficos em repouso, frequência cardíaca máxima alcançada, angina induzida por exercício, depressão do segmento ST induzida pelo exercício em relação ao repous,inclinação do pico do segmento ST do exercício. E dados descritivo de etinia do paciente: cor do cabelo do paciente e cor da pele do paciente.

# In[7]:


dados.value_counts()


# In[8]:


dados['sex'].value_counts()


# Em relação ao sexo temos: 
# Feminino (F) = 196
# Masculino (M) = 92

# ### Analisando a cor da pele do grupo de estudo:

# In[9]:


frequencia = pd.crosstab(dados.sex, dados.sk)
frequencia


# In[19]:


sns.histplot(dados['sk'], bins=30);


# ### Análise da cor da pele dos pacientes em relação ao sexo:

# Em relação aos pacientes do sexo Feminino temos:
# - 85 na cor da pele 0
# - 55 na cor da pele 1
# - 37 na cor da pele 2
# - 19 na cor da pele 3
# 
# Em relação aos pacientes do sexo Masculino temos:
# - 42 na cor da pele 0
# - 22 na cor da pele 1
# - 21 na cor da pele 2
# - 7 na cor da pele 3

# ### Analisando a cor do cabelo do grupo de estudo:

# In[10]:


cor_cabelo = pd.crosstab(dados.sex, dados.hc)
cor_cabelo


# In[20]:


sns.histplot(dados['hc'], bins=30);


# ## Análise da cor do cabelo dos pacientes em relação ao sexo:

# Em relação aos pacientes do sexo Feminino temos:
# - 121 na cor do cabelo 0
# - 55 na cor do cabelo 1
# - 20 na cor do cabelo 2
# 
# 
# Em relação aos pacientes do sexo Masculino temos:
# - 24 na cor do cabelo 0
# - 62 na cor do cabelo 1
# - 6 na cor do cabelo 2
# 

# In[18]:


glicose = pd.crosstab(dados.sex, dados.fbs)
glicose


# ## Análise da glicemia de jejum dos pacientes em relação ao sexo:

# Em relação aos pacientes do sexo Feminino temos:
# 
# - 166 apresentam glicemia igual ou abaixo de 120 mg/dl em jejum
# - 30 apresentam glicemia acima de 120 mg/dl em jejum
# 
# Em relação aos pacientes do sexo Masculino temos:
# 
# - 80 pacientes apresentam glicemia igual ou abaixo de 120 mg/dl em jejum
# - 12 pacientes apresentam glicemia acima de 120 mg/dl em jejum

# In[ ]:





# 🚔 Sistema de Análise Criminal

> Projeto acadêmico - Dashboard interativo para análise de dados criminais com Machine Learning e visualizações avançadas

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)

## 🎯 Sobre o Projeto

**Projeto Acadêmico** desenvolvido para demonstrar aplicação de técnicas de Machine Learning e visualização de dados na análise criminal. Sistema web interativo que oferece insights através de dashboards e modelos preditivos.

## Funcionalidades Principais

### 📊 **Dashboard Interativo**
- **10 abas especializadas** de análise
- Filtros dinâmicos em tempo real
- Interface responsiva e intuitiva

### 🗺️ **Análise Geográfica**
- Mapas de calor interativos
- Clustering geográfico inteligente
- Validação de coordenadas por bairro
- Visualização 3D de densidade

### 👮 **Análise Investigativa**
- **Desempenho por delegacia** - Taxa de resolução
- **Perfil dos suspeitos** - Idade, gênero, armas
- **Índice de gravidade** - Priorização de casos
- **Modus operandi** - Padrões criminais

### 🤖 **Machine Learning**
- Detecção de anomalias (Isolation Forest)
- Clustering avançado de padrões
- Previsão de risco (XGBoost/LightGBM)
- Modelos criptografados para segurança

### 🔒 **Segurança e Conformidade**
- Criptografia AES-128 para dados sensíveis
- Conformidade LGPD completa
- Logs de auditoria automáticos
- Controle de acesso por usuário

## 🛠️ Stack Tecnológica

### **Core**
- **Python 3.8+** - Linguagem principal
- **Streamlit 1.28+** - Framework web interativo

### **Data Science & ML**
- **Pandas 2.0+** - Manipulação de dados
- **NumPy 1.24+** - Computação numérica
- **Scikit-learn 1.3+** - Machine Learning
- **XGBoost 1.7+** - Gradient boosting
- **LightGBM 4.0+** - Modelos avançados

### **Visualização**
- **Plotly 5.15+** - Gráficos interativos
- **Folium 0.14+** - Mapas geográficos
- **PyDeck 0.8+** - Visualização 3D
- **Matplotlib/Seaborn** - Gráficos estatísticos

### **Segurança**
- **Cryptography 41.0+** - Criptografia AES
- **Hashlib** - Hash seguro
- **Logging** - Auditoria

## Instalação

### **Pré-requisitos**
- Python 3.8 ou superior
- 4GB RAM mínimo
- Windows 10/11, macOS, ou Linux

### **Setup Rápido**
```bash
# 1. Clonar repositório
git clone https://github.com/seu-usuario/sistema-analise-criminal.git
cd sistema-analise-criminal

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Configuração inicial
python setup.py

# 4. Executar análise
python advanced_analysis.py

# 5. Iniciar dashboard
streamlit run enhanced_app.py
```

### **Acesso ao Sistema**
- **URL:** http://localhost:8501
- **Login:** `admin`
- **Senha:** `admin123`

## Uso do Sistema

### **10 Abas de Análise Especializadas:**

1. **🌎 Geográfica** - Mapas de calor, clusters, pontos individuais
2. **🕒 Temporal** - Padrões por hora/dia/semana
3. **👥 Clusters** - Agrupamentos de crimes similares
4. **🚨 Anomalias** - Detecção de padrões atípicos
5. **🔮 Previsão** - Modelos preditivos de risco
6. **👮 Investigação** - Desempenho por delegacia
7. **🔫 Perfil Suspeitos** - Análise demográfica
8. **🧬 Gravidade** - Índice de priorização
9. **📝 Modus Operandi** - Padrões criminais
10. **📊 Estatísticas** - KPIs e insights gerais

### **Filtros Disponíveis:**
-  Período de análise
-  Bairros específicos
-  Tipos de crime
-  Faixa horária
-  Dias da semana

## 📁 Estrutura do Projeto

```
sistema-analise-criminal/
├── enhanced_app.py          #  Dashboard principal
├── dashboard_tabs.py        # Funções das abas
├── advanced_analysis.py     # Análise e ML
├── security_utils.py        # Segurança e LGPD
├── setup.py                # Configuração inicial
├── requirements.txt        # Dependências
├── data/                   # Dataset
│   └── dataset_ocorrencias_delegacia_5.csv
├── .gitignore             #  Arquivos ignorados
└── README.md              #  Documentação
```

## 🔒 Recursos de Segurança

- Criptografia AES-128 para dados sensíveis
- Validação geográfica por bairros
- Logs de auditoria automáticos
- Controle de acesso por usuário

## 🎓 Contexto Acadêmico

Este projeto foi desenvolvido como atividade acadêmica para demonstrar:
- Aplicação prática de Machine Learning
- Desenvolvimento de dashboards interativos
- Análise e visualização de dados
- Implementação de sistemas web com Python


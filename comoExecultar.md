# 🚀 Como Executar o Sistema

## 📋 Pré-requisitos
- Python 3.8 ou superior
- 4GB RAM mínimo

## ⚡ Execução Rápida

### 1. Instalar dependências
```bash
pip install -r requirements.txt
```

### 2. Configurar sistema
```bash
python setup.py
```

### 3. Executar análise
```bash
python advanced_analysis.py
```

### 4. Iniciar dashboard
```bash
streamlit run enhanced_app.py
```

## 🌐 Acesso
- **URL:** http://localhost:8501
- **Login:** admin
- **Senha:** admin123

## 📊 Funcionalidades
- 🌎 **Análise Geográfica** - Mapas interativos
- 🕒 **Análise Temporal** - Padrões por tempo
- 👥 **Clusters** - Agrupamentos de crimes
- 🚨 **Anomalias** - Detecção de padrões atípicos
- 🔮 **Previsão** - Modelos preditivos
- 📊 **Estatísticas** - KPIs e insights

## 🔧 Solução de Problemas

### Erro de módulo não encontrado
```bash
pip install nome-do-modulo
```

### Erro de porta ocupada
- Feche outros processos do Streamlit
- Ou acesse: http://localhost:8502

### Dataset não encontrado
- Coloque o arquivo CSV na pasta `data/`
- Nome: `dataset_ocorrencias_delegacia_5.csv`

"""
Análise Criminal Avançada com Modelos Melhorados
Implementa K-Prototypes, HDBSCAN, XGBoost e tratamento robusto de dados
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import silhouette_score, classification_report
# import hdbscan  # Removido devido a problemas de compilação no Windows
# from kmodes.kprototypes import KPrototypes  # Removido devido a problemas de compilação
import xgboost as xgb
import lightgbm as lgb
from security_utils import SecurityManager, LGPDCompliance, DataValidator
import warnings
warnings.filterwarnings('ignore')

class AdvancedCriminalAnalysis:
    """Classe principal para análise criminal avançada"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.security_manager = SecurityManager()
        self.lgpd = LGPDCompliance()
        self.validator = DataValidator()
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_and_preprocess_data(self):
        """Carrega e preprocessa os dados com validação robusta"""
        print("Carregando e preprocessando dados...")
        
        # Carregar dados
        df = pd.read_csv(self.data_path, sep=',')
        print(f"Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas")
        
        # Normalizar nomes de colunas
        df.columns = df.columns.str.lower().str.strip()
        
        # Validar e limpar dados
        df_clean = self.validator.clean_and_validate_data(df)
        print(f"Após limpeza: {df_clean.shape[0]} linhas válidas")
        
        # Engenharia de features temporais
        df_clean['data_ocorrencia'] = pd.to_datetime(df_clean['data_ocorrencia'], errors='coerce')
        df_clean = df_clean.dropna(subset=['data_ocorrencia'])
        
        df_clean['hora'] = df_clean['data_ocorrencia'].dt.hour
        df_clean['dia_semana'] = df_clean['data_ocorrencia'].dt.day_name()
        df_clean['mes'] = df_clean['data_ocorrencia'].dt.month
        df_clean['ano'] = df_clean['data_ocorrencia'].dt.year
        df_clean['dia_mes'] = df_clean['data_ocorrencia'].dt.day
        
        # Mapeamento de dias da semana
        day_map = {
            'Monday': 'segunda-feira', 'Tuesday': 'terça-feira',
            'Wednesday': 'quarta-feira', 'Thursday': 'quinta-feira',
            'Friday': 'sexta-feira', 'Saturday': 'sábado', 'Sunday': 'domingo'
        }
        df_clean['dia_semana'] = df_clean['dia_semana'].map(day_map)
        
        # Features adicionais
        df_clean['periodo_dia'] = df_clean['hora'].apply(self._categorize_time_period)
        df_clean['fim_semana'] = df_clean['dia_semana'].isin(['sábado', 'domingo'])
        
        # Limpeza de coordenadas
        for col in ['latitude', 'longitude']:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        df_clean = df_clean.dropna(subset=['latitude', 'longitude'])
        
        # Tratamento de valores faltantes em colunas categóricas
        categorical_cols = ['bairro', 'tipo_crime', 'descricao_modus_operandi']
        for col in categorical_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna('Não Informado')
        
        self.df = df_clean
        print("Pré-processamento concluído.")
        return df_clean
    
    def _categorize_time_period(self, hour):
        """Categoriza período do dia"""
        if 6 <= hour < 12:
            return 'Manhã'
        elif 12 <= hour < 18:
            return 'Tarde'
        elif 18 <= hour < 24:
            return 'Noite'
        else:
            return 'Madrugada'
    
    def perform_clustering(self, method='kmeans'):
        """Realiza clusterização com diferentes métodos"""
        print(f"Executando clusterização com {method}...")
        
        # Preparar dados para clusterização
        features_clustering = ['bairro', 'tipo_crime', 'hora', 'dia_semana', 'latitude', 'longitude']
        df_cluster = self.df.dropna(subset=features_clustering).copy()
        
        if method == 'dbscan':
            # DBSCAN como alternativa ao HDBSCAN
            from sklearn.cluster import DBSCAN
            features_for_dbscan = ['latitude', 'longitude', 'hora']
            X_dbscan = df_cluster[features_for_dbscan].copy()
            X_dbscan_scaled = self.scaler.fit_transform(X_dbscan)
            
            clusterer = DBSCAN(eps=0.3, min_samples=10)
            clusters = clusterer.fit_predict(X_dbscan_scaled)
            
        elif method == 'agglomerative':
            # Clustering Hierárquico Aglomerativo
            from sklearn.cluster import AgglomerativeClustering
            features_for_agg = ['latitude', 'longitude', 'hora']
            X_agg = df_cluster[features_for_agg].copy()
            X_agg_scaled = self.scaler.fit_transform(X_agg)
            
            clusterer = AgglomerativeClustering(n_clusters=5, linkage='ward')
            clusters = clusterer.fit_predict(X_agg_scaled)
            
        else:  # kmeans como método principal
            from sklearn.cluster import KMeans
            # Usar encoding para variáveis categóricas
            df_encoded = pd.get_dummies(df_cluster[features_clustering])
            df_encoded_scaled = self.scaler.fit_transform(df_encoded)
            
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(df_encoded_scaled)
        
        df_cluster['cluster'] = clusters
        
        # Calcular silhouette score se possível
        try:
            if method == 'dbscan':
                sil_score = silhouette_score(X_dbscan_scaled, clusters) if len(set(clusters)) > 1 else 0
            elif method == 'agglomerative':
                sil_score = silhouette_score(X_agg_scaled, clusters) if len(set(clusters)) > 1 else 0
            else:  # kmeans
                sil_score = silhouette_score(df_encoded_scaled, clusters)
        except:
            sil_score = 0
        
        print(f"Silhouette Score ({method}): {sil_score:.3f}")
        
        # Adicionar clusters ao dataframe principal
        self.df['cluster'] = -1  # Valor padrão
        self.df.loc[df_cluster.index, 'cluster'] = clusters
        
        return sil_score
    
    def detect_anomalies(self, method='isolation_forest'):
        """Detecta anomalias com diferentes métodos"""
        print(f"Detectando anomalias com {method}...")
        
        features_anomaly = ['latitude', 'longitude', 'hora']
        df_anomaly = self.df.dropna(subset=features_anomaly).copy()
        
        # Escalar features
        X_anomaly = self.scaler.fit_transform(df_anomaly[features_anomaly])
        
        if method == 'isolation_forest':
            model_anomaly = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
        elif method == 'lof':
            model_anomaly = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
        
        anomalies = model_anomaly.fit_predict(X_anomaly)
        df_anomaly['anomalia'] = anomalies
        
        # Adicionar anomalias ao dataframe principal
        self.df['anomalia'] = 1  # Valor padrão (normal)
        self.df.loc[df_anomaly.index, 'anomalia'] = anomalies
        
        print(f"Anomalias detectadas: {sum(anomalies == -1)} de {len(anomalies)} ({sum(anomalies == -1)/len(anomalies)*100:.1f}%)")
    
    def train_risk_prediction_model(self, model_type='xgboost'):
        """Treina modelo de previsão de risco"""
        print(f"Treinando modelo de previsão de risco com {model_type}...")
        
        # Criar variável alvo baseada em densidade de crimes por bairro
        bairro_counts = self.df['bairro'].value_counts()
        
        # Usar quantis para criar níveis de risco mais balanceados
        q33 = bairro_counts.quantile(0.33)
        q66 = bairro_counts.quantile(0.66)
        
        def classify_risk(bairro):
            count = bairro_counts.get(bairro, 0)
            if count <= q33:
                return 'Baixo Risco'
            elif count <= q66:
                return 'Médio Risco'
            else:
                return 'Alto Risco'
        
        self.df['nivel_risco'] = self.df['bairro'].apply(classify_risk)
        
        # Preparar features para o modelo
        features_pred = ['bairro', 'hora', 'dia_semana', 'mes', 'periodo_dia', 'fim_semana']
        target = 'nivel_risco'
        
        df_model = self.df.dropna(subset=features_pred + [target]).copy()
        
        # Encoding de variáveis categóricas
        X = pd.get_dummies(df_model[features_pred], drop_first=True)
        y = df_model[target]
        
        # Salvar colunas do modelo para uso posterior
        self.model_columns = X.columns.tolist()
        
        # Split dos dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Treinar modelo
        if model_type == 'xgboost':
            # Encoding do target para XGBoost
            le_target = LabelEncoder()
            y_train_encoded = le_target.fit_transform(y_train)
            y_test_encoded = le_target.transform(y_test)
            
            model = xgb.XGBClassifier(
                random_state=42,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                eval_metric='mlogloss'
            )
            model.fit(X_train, y_train_encoded)
            
            # Converter predições de volta
            pred_encoded = model.predict(X_test)
            pred = le_target.inverse_transform(pred_encoded)
            
            # Salvar label encoder
            self.target_encoder = le_target
            
        elif model_type == 'lightgbm':
            model = lgb.LGBMClassifier(
                random_state=42,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                verbose=-1
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            
        else:  # random_forest
            model = RandomForestClassifier(
                random_state=42,
                n_estimators=100,
                max_depth=10,
                class_weight='balanced'
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
        
        # Avaliar modelo
        class_report = classification_report(y_test, pred, output_dict=True, zero_division=0)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y if model_type != 'xgboost' else le_target.transform(y), cv=5)
        
        print(f"Acurácia média (CV): {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Salvar modelo e metadados
        model_data = {
            'model': model,
            'model_columns': self.model_columns,
            'bairros_list': df_model['bairro'].unique().tolist(),
            'dias_semana_list': df_model['dia_semana'].unique().tolist(),
            'feature_names': features_pred,
            'model_type': model_type,
            'cv_scores': cv_scores,
            'class_report': class_report
        }
        
        if model_type == 'xgboost':
            model_data['target_encoder'] = self.target_encoder
        
        # Salvar modelo criptografado
        self.security_manager.save_encrypted_model(model_data, 'risk_model_secure.pkl')
        
        return class_report, cv_scores.mean()
    
    def get_feature_importance(self, model_data):
        """Obtém importância das features do modelo"""
        model = model_data['model']
        model_type = model_data.get('model_type', 'random_forest')
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_names = model_data['model_columns']
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        return None
    
    def save_results(self):
        """Salva resultados da análise"""
        results = {
            'data': self.df,
            'silhouette_score': getattr(self, 'silhouette_score', 0),
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'analysis_timestamp': pd.Timestamp.now()
        }
        
        # Salvar resultados criptografados
        self.security_manager.save_encrypted_model(results, 'analysis_results_secure.pkl')
        
        print("Resultados salvos com segurança.")

def main():
    """Função principal"""
    print("=== Análise Criminal Avançada ===")
    
    # Inicializar análise
    analysis = AdvancedCriminalAnalysis('data/dataset_ocorrencias_delegacia_5.csv')
    
    # Carregar e preprocessar dados
    df = analysis.load_and_preprocess_data()
    
    # Clusterização
    sil_score = analysis.perform_clustering(method='kmeans')
    analysis.silhouette_score = sil_score
    
    # Detecção de anomalias
    analysis.detect_anomalies(method='isolation_forest')
    
    # Modelo de previsão de risco
    class_report, cv_accuracy = analysis.train_risk_prediction_model(model_type='xgboost')
    
    # Salvar resultados
    analysis.save_results()
    
    print(f"\n=== Resumo da Análise ===")
    print(f"Total de registros processados: {len(df)}")
    print(f"Silhouette Score: {sil_score:.3f}")
    print(f"Acurácia do modelo de risco: {cv_accuracy:.3f}")
    print(f"Anomalias detectadas: {sum(df['anomalia'] == -1)}")
    
    print("\nAnálise concluída! Execute o dashboard com:")
    print("streamlit run enhanced_app.py")

if __name__ == "__main__":
    main()

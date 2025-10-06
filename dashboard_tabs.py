"""
Funções das abas do dashboard de análise criminal
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import os
from security_utils import SecurityManager

def create_scatter_map(df, show_anomalies):
    """Cria mapa com pontos individuais"""
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        st.error("Coordenadas não disponíveis nos dados.")
        return
    
    
    # Filtrar apenas coordenadas válidas para Recife
    from security_utils import DataValidator
    
    # Sempre usar validação específica por bairro quando possível
    if 'bairro' in df.columns:
        # Validação individual por bairro para cada linha
        valid_coords = df.apply(
            lambda row: DataValidator.validate_coordinates_by_neighborhood(
                row['latitude'], row['longitude'], row['bairro']
            ), 
            axis=1
        )
        bairros_unicos = df['bairro'].unique()
        if len(bairros_unicos) == 1:
            st.info(f"🎯 Usando validação específica para: {bairros_unicos[0]}")
        else:
            st.info(f"🎯 Usando validação específica para {len(bairros_unicos)} bairros: {', '.join(sorted(bairros_unicos))}")
    else:
        # Fallback para validação geral mais restritiva
        valid_coords = df.apply(
            lambda row: DataValidator.validate_coordinates(row['latitude'], row['longitude']), 
            axis=1
        )
        st.info("🎯 Usando validação geral (sem informação de bairro)")
    
    df_valid = df[valid_coords].copy()
    
    if len(df_valid) == 0:
        st.error("❌ Nenhuma coordenada válida encontrada para o Recife.")
        st.info("💡 Verifique se os dados contêm coordenadas da região metropolitana do Recife.")
        return
    
    # Mostrar estatísticas de limpeza
    removed_count = len(df) - len(df_valid)
    if removed_count > 0:
        st.warning(f"⚠️ {removed_count} pontos com coordenadas inválidas foram removidos do mapa.")
    
    center_lat = df_valid['latitude'].mean()
    center_lon = df_valid['longitude'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
    # Adicionar pontos normais (usando apenas coordenadas válidas)
    normal_data = df_valid[df_valid['anomalia'] != -1] if 'anomalia' in df_valid.columns else df_valid
    
    for _, row in normal_data.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=3,
            popup=f"{row.get('tipo_crime', 'N/A')} - {row.get('bairro', 'N/A')}",
            color='blue',
            fill=True,
            fillColor='blue'
        ).add_to(m)
    
    # Adicionar anomalias se solicitado (usando apenas coordenadas válidas)
    if show_anomalies and 'anomalia' in df_valid.columns:
        anomalies = df_valid[df_valid['anomalia'] == -1]
        for _, row in anomalies.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=6,
                popup=f"ANOMALIA: {row.get('tipo_crime', 'N/A')} - {row.get('bairro', 'N/A')}",
                color='red',
                fill=True,
                fillColor='red'
            ).add_to(m)
    
    st_folium(m, use_container_width=True, height=500)

def show_temporal_analysis(df):
    """Aba de análise temporal"""
    st.header("🕒 Análise Temporal - Padrões de Criminalidade")
    
    if len(df) == 0:
        st.warning("Nenhum dado disponível para análise temporal.")
        return
    
    # Análise por hora do dia
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Distribuição por Hora do Dia")
        if 'hora' in df.columns:
            hourly_counts = df['hora'].value_counts().sort_index()
            
            fig_hour = px.bar(
                x=hourly_counts.index,
                y=hourly_counts.values,
                labels={'x': 'Hora do Dia', 'y': 'Número de Ocorrências'},
                title="Ocorrências por Hora"
            )
            fig_hour.update_layout(showlegend=False)
            st.plotly_chart(fig_hour, use_container_width=True)
            
            # Identificar pico
            peak_hour = hourly_counts.idxmax()
            st.info(f"🕐 **Pico de criminalidade:** {peak_hour}:00 com {hourly_counts[peak_hour]} ocorrências")
    
    with col2:
        st.subheader("📅 Distribuição por Dia da Semana")
        if 'dia_semana' in df.columns:
            day_order = ['segunda-feira', 'terça-feira', 'quarta-feira', 'quinta-feira', 'sexta-feira', 'sábado', 'domingo']
            daily_counts = df['dia_semana'].value_counts().reindex(day_order).fillna(0)
            
            fig_day = px.bar(
                x=daily_counts.index,
                y=daily_counts.values,
                labels={'x': 'Dia da Semana', 'y': 'Número de Ocorrências'},
                title="Ocorrências por Dia da Semana"
            )
            fig_day.update_xaxes(tickangle=45)
            st.plotly_chart(fig_day, use_container_width=True)
            
            # Identificar dia mais crítico
            peak_day = daily_counts.idxmax()
            st.info(f"📅 **Dia mais crítico:** {peak_day} com {daily_counts[peak_day]} ocorrências")
    
    # Análise mensal
    if 'data_ocorrencia' in df.columns:
        st.subheader("📈 Evolução Temporal")
        
        df_temp = df.copy()
        df_temp['mes_ano'] = df_temp['data_ocorrencia'].dt.to_period('M')
        monthly_counts = df_temp['mes_ano'].value_counts().sort_index()
        
        fig_monthly = px.line(
            x=monthly_counts.index.astype(str),
            y=monthly_counts.values,
            labels={'x': 'Mês/Ano', 'y': 'Número de Ocorrências'},
            title="Evolução Mensal das Ocorrências"
        )
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    # Heatmap de hora vs dia da semana
    if 'hora' in df.columns and 'dia_semana' in df.columns:
        st.subheader("🔥 Mapa de Calor: Hora vs Dia da Semana")
        
        pivot_data = df.pivot_table(
            values='tipo_crime', 
            index='hora', 
            columns='dia_semana', 
            aggfunc='count', 
            fill_value=0
        )
        
        # Reordenar colunas
        day_order = ['segunda-feira', 'terça-feira', 'quarta-feira', 'quinta-feira', 'sexta-feira', 'sábado', 'domingo']
        pivot_data = pivot_data.reindex(columns=[d for d in day_order if d in pivot_data.columns])
        
        fig_heatmap = px.imshow(
            pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            aspect="auto",
            color_continuous_scale="Reds",
            title="Intensidade de Crimes por Hora e Dia da Semana"
        )
        fig_heatmap.update_xaxes(tickangle=45)
        st.plotly_chart(fig_heatmap, use_container_width=True)

def show_cluster_analysis(df, sil_score):
    """Aba de análise de clusters"""
    st.header("👥 Clusters de Ocorrências - Modus Operandi")
    
    if 'cluster' not in df.columns:
        st.warning("Dados de clusterização não disponíveis.")
        return
    
    # Informações gerais sobre clustering
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_clusters = df['cluster'].nunique()
        st.metric("🎯 Número de Clusters", n_clusters)
    
    with col2:
        st.metric("📊 Silhouette Score", f"{sil_score:.3f}")
    
    with col3:
        noise_points = len(df[df['cluster'] == -1]) if -1 in df['cluster'].values else 0
        st.metric("🔍 Pontos de Ruído", noise_points)
    
    # Seletor de cluster
    clusters_available = sorted([c for c in df['cluster'].unique() if c != -1])
    
    if clusters_available:
        selected_cluster = st.selectbox(
            "Selecione um Cluster para Análise Detalhada",
            options=clusters_available
        )
        
        # Análise do cluster selecionado
        cluster_data = df[df['cluster'] == selected_cluster]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"📋 Perfil do Cluster {selected_cluster}")
            st.write(f"**Total de Ocorrências:** {len(cluster_data)}")
            
            # Top crimes
            if 'tipo_crime' in cluster_data.columns:
                top_crimes = cluster_data['tipo_crime'].value_counts().head(5)
                st.write("**Principais Tipos de Crime:**")
                for crime, count in top_crimes.items():
                    percentage = (count / len(cluster_data)) * 100
                    st.write(f"• {crime}: {count} ({percentage:.1f}%)")
        
        with col2:
            st.subheader(f"🏘️ Localização do Cluster {selected_cluster}")
            
            # Top bairros
            if 'bairro' in cluster_data.columns:
                top_bairros = cluster_data['bairro'].value_counts().head(5)
                st.write("**Principais Bairros:**")
                for bairro, count in top_bairros.items():
                    percentage = (count / len(cluster_data)) * 100
                    st.write(f"• {bairro}: {count} ({percentage:.1f}%)")
        
        # Tabela de exemplos
        st.subheader("📄 Exemplos de Ocorrências do Cluster")
        display_cols = ['data_ocorrencia', 'tipo_crime', 'bairro']
        if 'descricao_modus_operandi' in cluster_data.columns:
            display_cols.append('descricao_modus_operandi')
        
        st.dataframe(
            cluster_data[display_cols].head(10),
            use_container_width=True
        )
    
    # Visualização de todos os clusters
    st.subheader("📊 Comparação entre Clusters")
    
    # Gráfico de barras por cluster
    cluster_counts = df[df['cluster'] != -1]['cluster'].value_counts().sort_index()
    
    fig_clusters = px.bar(
        x=cluster_counts.index,
        y=cluster_counts.values,
        labels={'x': 'Cluster ID', 'y': 'Número de Ocorrências'},
        title="Distribuição de Ocorrências por Cluster"
    )
    st.plotly_chart(fig_clusters, use_container_width=True)

def show_anomaly_analysis(df):
    """Aba de análise de anomalias"""
    st.header("🚨 Ocorrências Atípicas - Alertas de Segurança")
    
    if 'anomalia' not in df.columns:
        st.warning("Dados de detecção de anomalias não disponíveis.")
        return
    
    anomalies = df[df['anomalia'] == -1]
    normal_cases = df[df['anomalia'] == 1]
    
    # KPIs de anomalias
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🚨 Total de Anomalias", len(anomalies))
    
    with col2:
        percentage = (len(anomalies) / len(df)) * 100 if len(df) > 0 else 0
        st.metric("📊 Percentual", f"{percentage:.2f}%")
    
    with col3:
        if len(anomalies) > 0:
            most_common_crime = anomalies['tipo_crime'].mode().iloc[0] if 'tipo_crime' in anomalies.columns else "N/A"
            st.metric("🎯 Crime Mais Comum", most_common_crime)
    
    with col4:
        if len(anomalies) > 0:
            most_affected_area = anomalies['bairro'].mode().iloc[0] if 'bairro' in anomalies.columns else "N/A"
            st.metric("🏘️ Área Mais Afetada", most_affected_area)
    
    if len(anomalies) == 0:
        st.info("✅ Nenhuma anomalia detectada nos dados filtrados.")
        return
    
    # Análise detalhada das anomalias
    st.subheader("🔍 Análise Detalhada das Anomalias")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribuição por tipo de crime
        if 'tipo_crime' in anomalies.columns:
            crime_dist = anomalies['tipo_crime'].value_counts()
            fig_crime = px.pie(
                values=crime_dist.values,
                names=crime_dist.index,
                title="Anomalias por Tipo de Crime"
            )
            st.plotly_chart(fig_crime, use_container_width=True)
    
    with col2:
        # Distribuição por bairro
        if 'bairro' in anomalies.columns:
            bairro_dist = anomalies['bairro'].value_counts().head(10)
            fig_bairro = px.bar(
                x=bairro_dist.values,
                y=bairro_dist.index,
                orientation='h',
                title="Top 10 Bairros com Anomalias"
            )
            st.plotly_chart(fig_bairro, use_container_width=True)
    
    # Tabela de anomalias
    st.subheader("📋 Lista de Anomalias Detectadas")
    
    # Filtros para anomalias
    col1, col2 = st.columns(2)
    
    with col1:
        if 'tipo_crime' in anomalies.columns:
            crime_filter = st.multiselect(
                "Filtrar por Tipo de Crime",
                options=anomalies['tipo_crime'].unique(),
                default=anomalies['tipo_crime'].unique()[:3]
            )
            anomalies_filtered = anomalies[anomalies['tipo_crime'].isin(crime_filter)]
        else:
            anomalies_filtered = anomalies
    
    with col2:
        if 'bairro' in anomalies.columns:
            bairro_filter = st.multiselect(
                "Filtrar por Bairro",
                options=anomalies['bairro'].unique(),
                default=anomalies['bairro'].unique()[:5]
            )
            anomalies_filtered = anomalies_filtered[anomalies_filtered['bairro'].isin(bairro_filter)]
    
    # Exibir tabela
    display_cols = ['data_ocorrencia', 'tipo_crime', 'bairro']
    if 'descricao_modus_operandi' in anomalies_filtered.columns:
        display_cols.append('descricao_modus_operandi')
    
    st.dataframe(
        anomalies_filtered[display_cols].sort_values('data_ocorrencia', ascending=False),
        use_container_width=True
    )
    
    # Insights sobre anomalias
    st.subheader("💡 Insights sobre Anomalias")
    
    insights = []
    
    if len(anomalies) > 0:
        # Análise temporal
        if 'hora' in anomalies.columns:
            peak_hour = anomalies['hora'].mode().iloc[0]
            insights.append(f"🕐 Horário mais comum para anomalias: {peak_hour}:00")
        
        # Análise geográfica
        if 'bairro' in anomalies.columns:
            top_bairro = anomalies['bairro'].value_counts().index[0]
            count_bairro = anomalies['bairro'].value_counts().iloc[0]
            insights.append(f"🏘️ Bairro com mais anomalias: {top_bairro} ({count_bairro} casos)")
        
        # Comparação com casos normais
        if len(normal_cases) > 0 and 'tipo_crime' in df.columns:
            anomaly_crimes = set(anomalies['tipo_crime'].unique())
            normal_crimes = set(normal_cases['tipo_crime'].unique())
            unique_anomaly_crimes = anomaly_crimes - normal_crimes
            
            if unique_anomaly_crimes:
                insights.append(f"🎯 Crimes únicos em anomalias: {', '.join(unique_anomaly_crimes)}")
    
    for insight in insights:
        st.info(insight)

def show_risk_prediction():
    """Aba de previsão de risco"""
    st.header("🔮 Previsão de Risco de Criminalidade")
    
    # Tentar carregar modelo
    security_manager = SecurityManager()
    
    try:
        if os.path.exists('risk_model_secure.pkl'):
            model_data = security_manager.load_encrypted_model('risk_model_secure.pkl')
        elif os.path.exists('risk_model.pkl'):
            import pickle
            with open('risk_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
                # Converter formato antigo se necessário
                if isinstance(model_data, tuple):
                    model, model_columns, bairros_list, dias_semana_list = model_data
                    model_data = {
                        'model': model,
                        'model_columns': model_columns,
                        'bairros_list': bairros_list,
                        'dias_semana_list': dias_semana_list
                    }
        else:
            st.error("❌ Modelo de previsão não encontrado.")
            return
            
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {str(e)}")
        return
    
    # Interface de previsão
    st.subheader("🎯 Fazer Previsão de Risco")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_bairro = st.selectbox(
            "Selecione o Bairro",
            options=sorted(model_data['bairros_list'])
        )
    
    with col2:
        selected_dia = st.selectbox(
            "Selecione o Dia da Semana",
            options=model_data['dias_semana_list']
        )
    
    with col3:
        selected_hora = st.slider("Selecione a Hora", 0, 23, 19)
    
    if st.button("🔮 Prever Nível de Risco", type="primary"):
        try:
            # Preparar dados para previsão
            input_data = pd.DataFrame(0, index=[0], columns=model_data['model_columns'])
            
            # Preencher features
            bairro_col = f"bairro_{selected_bairro}"
            if bairro_col in input_data.columns:
                input_data[bairro_col] = 1
            
            dia_col = f"dia_semana_{selected_dia}"
            if dia_col in input_data.columns:
                input_data[dia_col] = 1
            
            if 'hora' in input_data.columns:
                input_data['hora'] = selected_hora
            
            # Fazer previsão
            model = model_data['model']
            
            if 'target_encoder' in model_data:
                # XGBoost com encoder
                prediction_encoded = model.predict(input_data)[0]
                prediction = model_data['target_encoder'].inverse_transform([prediction_encoded])[0]
                proba = model.predict_proba(input_data).max() * 100
            else:
                # Outros modelos
                prediction = model.predict(input_data)[0]
                proba = model.predict_proba(input_data).max() * 100
            
            # Exibir resultado
            st.subheader("📊 Resultado da Previsão")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if "Alto" in prediction:
                    st.error(f"🔴 **Nível de Risco: {prediction}**")
                elif "Médio" in prediction:
                    st.warning(f"🟡 **Nível de Risco: {prediction}**")
                else:
                    st.success(f"🟢 **Nível de Risco: {prediction}**")
            
            with col2:
                st.metric("🎯 Confiança", f"{proba:.1f}%")
            
            # Recomendações
            st.subheader("💡 Recomendações")
            
            if "Alto" in prediction:
                st.error("""
                **Ações Recomendadas:**
                • Aumentar patrulhamento na área
                • Considerar rondas preventivas
                • Alertar equipes próximas
                • Monitorar atividades suspeitas
                """)
            elif "Médio" in prediction:
                st.warning("""
                **Ações Recomendadas:**
                • Manter patrulhamento regular
                • Estar atento a mudanças no padrão
                • Coordenar com outras unidades
                """)
            else:
                st.success("""
                **Situação Normal:**
                • Patrulhamento de rotina
                • Foco em outras áreas prioritárias
                """)
                
        except Exception as e:
            st.error(f"Erro na previsão: {str(e)}")
    
    # Feature Importance
    if 'model' in model_data:
        st.subheader("📈 Importância das Variáveis")
        
        try:
            model = model_data['model']
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': model_data['model_columns'],
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False).head(10)
                
                fig_importance = px.bar(
                    importance_df,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title="Top 10 Variáveis Mais Importantes"
                )
                st.plotly_chart(fig_importance, use_container_width=True)
                
        except Exception as e:
            st.warning("Não foi possível exibir a importância das variáveis.")

def show_investigative_performance(df):
    """Aba de desempenho investigativo"""
    st.header("👮 Desempenho Investigativo - Polícia Civil")
    
    if 'status_investigacao' not in df.columns or 'orgao_responsavel' not in df.columns:
        st.warning("Dados de investigação não disponíveis no dataset atual.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Taxa de Resolução por Delegacia")
        
        # Calcular taxa de resolução por delegacia
        delegacia_stats = df.groupby('orgao_responsavel').agg({
            'status_investigacao': ['count', lambda x: (x == 'Concluído').sum()]
        }).round(2)
        
        delegacia_stats.columns = ['Total_Casos', 'Casos_Concluidos']
        delegacia_stats['Taxa_Resolucao'] = (
            delegacia_stats['Casos_Concluidos'] / delegacia_stats['Total_Casos'] * 100
        ).round(1)
        
        # Gráfico de barras
        fig_resolucao = px.bar(
            delegacia_stats.reset_index(),
            x='orgao_responsavel',
            y='Taxa_Resolucao',
            title="Taxa de Resolução por Delegacia (%)",
            color='Taxa_Resolucao',
            color_continuous_scale='RdYlGn'
        )
        fig_resolucao.update_xaxes(tickangle=45)
        st.plotly_chart(fig_resolucao, use_container_width=True)
    
    with col2:
        st.subheader("📊 Volume de Casos por Delegacia")
        
        # Volume de casos
        volume_casos = df['orgao_responsavel'].value_counts()
        
        fig_volume = px.pie(
            values=volume_casos.values,
            names=volume_casos.index,
            title="Distribuição de Casos por Delegacia"
        )
        st.plotly_chart(fig_volume, use_container_width=True)
    
    # Tabela resumo
    st.subheader("📋 Resumo por Delegacia")
    delegacia_stats['Taxa_Resolucao'] = delegacia_stats['Taxa_Resolucao'].astype(str) + '%'
    st.dataframe(delegacia_stats, use_container_width=True)
    
    # Insights
    st.subheader("💡 Insights Investigativos")
    melhor_delegacia = delegacia_stats['Taxa_Resolucao'].str.rstrip('%').astype(float).idxmax()
    pior_delegacia = delegacia_stats['Taxa_Resolucao'].str.rstrip('%').astype(float).idxmin()
    
    st.success(f"🏆 **Melhor Performance:** {melhor_delegacia}")
    st.error(f"⚠️ **Necessita Atenção:** {pior_delegacia}")

def show_suspect_profile(df):
    """Aba de perfil dos suspeitos"""
    st.header("🔫 Perfil dos Suspeitos - Análise Demográfica")
    
    required_cols = ['idade_suspeito', 'genero_suspeito', 'arma_utilizada', 'tipo_crime']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.warning(f"Colunas não disponíveis: {', '.join(missing_cols)}")
        st.info("Simulando dados para demonstração...")
        
        # Simular dados para demonstração
        np.random.seed(42)
        df_demo = df.copy()
        df_demo['idade_suspeito'] = np.random.randint(16, 65, len(df))
        df_demo['genero_suspeito'] = np.random.choice(['Masculino', 'Feminino'], len(df), p=[0.7, 0.3])
        df_demo['arma_utilizada'] = np.random.choice(
            ['Arma de Fogo', 'Arma Branca', 'Sem Arma', 'Outros'], 
            len(df), p=[0.3, 0.2, 0.4, 0.1]
        )
        df = df_demo
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("👥 Distribuição por Gênero")
        genero_dist = df['genero_suspeito'].value_counts()
        
        fig_genero = px.pie(
            values=genero_dist.values,
            names=genero_dist.index,
            title="Suspeitos por Gênero",
            color_discrete_sequence=['#FF6B6B', '#4ECDC4']
        )
        st.plotly_chart(fig_genero, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Faixa Etária")
        
        # Criar faixas etárias
        df['faixa_etaria'] = pd.cut(
            df['idade_suspeito'], 
            bins=[0, 18, 25, 35, 50, 100],
            labels=['<18', '18-25', '26-35', '36-50', '>50']
        )
        
        idade_dist = df['faixa_etaria'].value_counts()
        
        fig_idade = px.bar(
            x=idade_dist.index,
            y=idade_dist.values,
            title="Suspeitos por Faixa Etária",
            color=idade_dist.values,
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_idade, use_container_width=True)
    
    with col3:
        st.subheader("🔫 Armas Utilizadas")
        arma_dist = df['arma_utilizada'].value_counts()
        
        fig_arma = px.bar(
            x=arma_dist.values,
            y=arma_dist.index,
            orientation='h',
            title="Tipos de Arma",
            color=arma_dist.values,
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_arma, use_container_width=True)
    
    # Análise cruzada
    st.subheader("🔍 Análise Cruzada: Crime vs Perfil")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Heatmap: Tipo de Crime vs Gênero
        crime_genero = pd.crosstab(df['tipo_crime'], df['genero_suspeito'])
        
        fig_heatmap1 = px.imshow(
            crime_genero.values,
            x=crime_genero.columns,
            y=crime_genero.index,
            title="Crime vs Gênero",
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_heatmap1, use_container_width=True)
    
    with col2:
        # Heatmap: Tipo de Crime vs Arma
        crime_arma = pd.crosstab(df['tipo_crime'], df['arma_utilizada'])
        
        fig_heatmap2 = px.imshow(
            crime_arma.values,
            x=crime_arma.columns,
            y=crime_arma.index,
            title="Crime vs Arma Utilizada",
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_heatmap2, use_container_width=True)
    
    # Insights
    st.subheader("💡 Insights do Perfil")
    
    genero_predominante = df['genero_suspeito'].mode()[0]
    faixa_predominante = df['faixa_etaria'].mode()[0]
    arma_predominante = df['arma_utilizada'].mode()[0]
    
    st.info(f"👤 **Perfil Típico:** {genero_predominante}, {faixa_predominante} anos, usando {arma_predominante}")
    
    # Estatísticas por tipo de crime
    crime_stats = df.groupby('tipo_crime').agg({
        'genero_suspeito': lambda x: x.mode()[0],
        'faixa_etaria': lambda x: x.mode()[0],
        'arma_utilizada': lambda x: x.mode()[0]
    })
    
    st.subheader("📊 Perfil por Tipo de Crime")
    st.dataframe(crime_stats, use_container_width=True)

def show_gravity_index(df):
    """Aba de índice de gravidade"""
    st.header("🧬 Índice de Gravidade dos Casos")
    
    st.info("💡 **Metodologia:** Índice calculado com base em: tipo de crime, arma utilizada, número de vítimas e suspeitos")
    
    # Simular dados se necessário
    if 'num_vitimas' not in df.columns:
        np.random.seed(42)
        df['num_vitimas'] = np.random.randint(1, 5, len(df))
        df['num_suspeitos'] = np.random.randint(1, 4, len(df))
        df['arma_utilizada'] = np.random.choice(
            ['Arma de Fogo', 'Arma Branca', 'Sem Arma', 'Outros'], 
            len(df), p=[0.3, 0.2, 0.4, 0.1]
        )
    
    # Calcular índice de gravidade
    def calcular_gravidade(row):
        score = 0
        
        # Peso por tipo de crime
        crime_weights = {
            'Homicídio': 10, 'Latrocínio': 9, 'Sequestro': 8, 'Estupro': 8,
            'Roubo': 6, 'Extorsão': 5, 'Furto': 3, 'Estelionato': 2, 'Ameaça': 2
        }
        score += crime_weights.get(row['tipo_crime'], 1)
        
        # Peso por arma
        arma_weights = {
            'Arma de Fogo': 5, 'Arma Branca': 3, 'Outros': 2, 'Sem Arma': 1
        }
        score += arma_weights.get(row['arma_utilizada'], 1)
        
        # Peso por número de vítimas e suspeitos
        score += row['num_vitimas'] * 2
        score += row['num_suspeitos'] * 1
        
        return score
    
    df['indice_gravidade'] = df.apply(calcular_gravidade, axis=1)
    
    # Classificar gravidade
    df['nivel_gravidade'] = pd.cut(
        df['indice_gravidade'],
        bins=[0, 10, 20, 100],
        labels=['Baixa', 'Média', 'Alta']
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🟢 Baixa Gravidade", 
                 len(df[df['nivel_gravidade'] == 'Baixa']),
                 f"{len(df[df['nivel_gravidade'] == 'Baixa'])/len(df)*100:.1f}%")
    
    with col2:
        st.metric("🟡 Média Gravidade", 
                 len(df[df['nivel_gravidade'] == 'Média']),
                 f"{len(df[df['nivel_gravidade'] == 'Média'])/len(df)*100:.1f}%")
    
    with col3:
        st.metric("🔴 Alta Gravidade", 
                 len(df[df['nivel_gravidade'] == 'Alta']),
                 f"{len(df[df['nivel_gravidade'] == 'Alta'])/len(df)*100:.1f}%")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Distribuição de Gravidade")
        
        gravidade_dist = df['nivel_gravidade'].value_counts()
        
        fig_gravidade = px.pie(
            values=gravidade_dist.values,
            names=gravidade_dist.index,
            title="Casos por Nível de Gravidade",
            color_discrete_sequence=['#2ECC71', '#F39C12', '#E74C3C']
        )
        st.plotly_chart(fig_gravidade, use_container_width=True)
    
    with col2:
        st.subheader("🗺️ Gravidade por Bairro")
        
        bairro_gravidade = df.groupby('bairro')['indice_gravidade'].mean().sort_values(ascending=False)
        
        fig_bairro = px.bar(
            x=bairro_gravidade.values[:10],
            y=bairro_gravidade.index[:10],
            orientation='h',
            title="Top 10 Bairros - Índice Médio de Gravidade",
            color=bairro_gravidade.values[:10],
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_bairro, use_container_width=True)
    
    # Casos de alta gravidade
    st.subheader("🚨 Casos de Alta Gravidade - Prioridade Investigativa")
    
    casos_alta = df[df['nivel_gravidade'] == 'Alta'].sort_values('indice_gravidade', ascending=False)
    
    if len(casos_alta) > 0:
        st.dataframe(
            casos_alta[['tipo_crime', 'bairro', 'arma_utilizada', 'num_vitimas', 'indice_gravidade']].head(10),
            use_container_width=True
        )
    else:
        st.info("Nenhum caso de alta gravidade encontrado nos filtros atuais.")

def show_modus_operandi_analysis(df):
    """Aba de análise de modus operandi"""
    st.header("📝 Análise de Modus Operandi - Padrões Criminais")
    
    if 'descricao_modus_operandi' not in df.columns:
        st.warning("Campo 'descricao_modus_operandi' não disponível no dataset.")
        st.info("Simulando análise de padrões textuais...")
        
        # Simular descrições para demonstração
        modus_exemplos = [
            "Golpe telefônico fingindo ser do banco",
            "Invasão de residência durante madrugada",
            "Furto de veículo em estacionamento",
            "Fraude online com cartão clonado",
            "Roubo em via pública com arma",
            "Estelionato por aplicativo falso",
            "Sequestro relâmpago no trânsito"
        ]
        
        np.random.seed(42)
        df['descricao_modus_operandi'] = np.random.choice(modus_exemplos, len(df))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Padrões Mais Comuns")
        
        # Análise de frequência de padrões
        modus_freq = df['descricao_modus_operandi'].value_counts()
        
        fig_modus = px.bar(
            x=modus_freq.values,
            y=modus_freq.index,
            orientation='h',
            title="Modus Operandi Mais Frequentes",
            color=modus_freq.values,
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_modus, use_container_width=True)
    
    with col2:
        st.subheader("📈 Evolução Temporal dos Padrões")
        
        if 'data_ocorrencia' in df.columns:
            # Converter para datetime e extrair ano-mês como string
            df['data_dt'] = pd.to_datetime(df['data_ocorrencia'])
            df['mes_str'] = df['data_dt'].dt.strftime('%Y-%m')
            
            # Top 3 modus operandi
            top_modus = modus_freq.head(3).index
            
            modus_temporal = df[df['descricao_modus_operandi'].isin(top_modus)].groupby(
                ['mes_str', 'descricao_modus_operandi']
            ).size().reset_index(name='count')
            
            # Ordenar por mês
            modus_temporal = modus_temporal.sort_values('mes_str')
            
            fig_temporal = px.line(
                modus_temporal,
                x='mes_str',
                y='count',
                color='descricao_modus_operandi',
                title="Evolução dos Principais Padrões"
            )
            fig_temporal.update_xaxes(tickangle=45)
            st.plotly_chart(fig_temporal, use_container_width=True)
        else:
            st.info("Campo 'data_ocorrencia' não disponível para análise temporal.")
    
    # Análise por bairro
    st.subheader("🗺️ Modus Operandi por Região")
    
    modus_bairro = pd.crosstab(df['bairro'], df['descricao_modus_operandi'])
    
    # Mostrar apenas top 5 bairros e top 5 modus operandi
    top_bairros = df['bairro'].value_counts().head(5).index
    top_modus_5 = modus_freq.head(5).index
    
    modus_bairro_filtered = modus_bairro.loc[top_bairros, top_modus_5]
    
    fig_heatmap = px.imshow(
        modus_bairro_filtered.values,
        x=modus_bairro_filtered.columns,
        y=modus_bairro_filtered.index,
        title="Modus Operandi vs Bairros (Top 5)",
        color_continuous_scale='Blues'
    )
    fig_heatmap.update_xaxes(tickangle=45)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Insights e recomendações
    st.subheader("💡 Insights e Recomendações")
    
    modus_predominante = modus_freq.index[0]
    bairro_problema = df[df['descricao_modus_operandi'] == modus_predominante]['bairro'].mode()[0]
    
    insights = [
        f"🎯 **Padrão Dominante:** {modus_predominante} ({modus_freq.iloc[0]} casos)",
        f"🗺️ **Região Crítica:** {bairro_problema} concentra mais casos deste padrão",
        f"📊 **Diversidade:** {len(modus_freq)} padrões diferentes identificados",
        f"⚠️ **Recomendação:** Focar operações preventivas no padrão dominante"
    ]
    
    for insight in insights:
        st.info(insight)

def show_statistics_insights(df):
    """Aba de estatísticas e insights"""
    st.header("📊 Estatísticas e Insights Gerais")
    
    if len(df) == 0:
        st.warning("Nenhum dado disponível para análise estatística.")
        return
    
    # Estatísticas gerais
    st.subheader("📈 Resumo Estatístico")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_crimes = len(df)
        st.metric("📊 Total de Crimes", f"{total_crimes:,}")
    
    with col2:
        unique_locations = df['bairro'].nunique() if 'bairro' in df.columns else 0
        st.metric("🏘️ Bairros Afetados", unique_locations)
    
    with col3:
        if 'data_ocorrencia' in df.columns:
            date_range = (df['data_ocorrencia'].max() - df['data_ocorrencia'].min()).days
            st.metric("📅 Período (dias)", date_range)
    
    with col4:
        crime_types = df['tipo_crime'].nunique() if 'tipo_crime' in df.columns else 0
        st.metric("🎯 Tipos de Crime", crime_types)
    
    # Análises detalhadas
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top 10 Crimes Mais Frequentes")
        if 'tipo_crime' in df.columns:
            top_crimes = df['tipo_crime'].value_counts().head(10)
            fig_top_crimes = px.bar(
                x=top_crimes.values,
                y=top_crimes.index,
                orientation='h',
                title="Crimes Mais Frequentes"
            )
            st.plotly_chart(fig_top_crimes, use_container_width=True)
    
    with col2:
        st.subheader("🏘️ Top 10 Bairros Mais Afetados")
        if 'bairro' in df.columns:
            top_bairros = df['bairro'].value_counts().head(10)
            fig_top_bairros = px.bar(
                x=top_bairros.values,
                y=top_bairros.index,
                orientation='h',
                title="Bairros Mais Afetados"
            )
            st.plotly_chart(fig_top_bairros, use_container_width=True)
    
    # Distribuição de risco
    if 'nivel_risco' in df.columns:
        st.subheader("⚠️ Distribuição por Nível de Risco")
        
        risk_dist = df['nivel_risco'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_risk_pie = px.pie(
                values=risk_dist.values,
                names=risk_dist.index,
                title="Distribuição de Níveis de Risco",
                color_discrete_map={
                    'Alto Risco': '#ff4444',
                    'Médio Risco': '#ffaa00',
                    'Baixo Risco': '#44ff44'
                }
            )
            st.plotly_chart(fig_risk_pie, use_container_width=True)
        
        with col2:
            # Tabela de estatísticas por risco
            risk_stats = df.groupby('nivel_risco').agg({
                'tipo_crime': 'count',
                'bairro': 'nunique'
            }).rename(columns={
                'tipo_crime': 'Total Ocorrências',
                'bairro': 'Bairros Únicos'
            })
            
            st.write("**Estatísticas por Nível de Risco:**")
            st.dataframe(risk_stats, use_container_width=True)
    
    # Correlações e insights
    st.subheader("🔍 Insights Avançados")
    
    insights = []
    
    # Análise temporal
    if 'hora' in df.columns:
        peak_hours = df['hora'].value_counts().head(3)
        insights.append(f"🕐 **Horários de pico:** {', '.join([f'{h}:00' for h in peak_hours.index])}")
    
    # Análise de concentração
    if 'bairro' in df.columns:
        top_3_bairros = df['bairro'].value_counts().head(3)
        total_top_3 = top_3_bairros.sum()
        percentage = (total_top_3 / len(df)) * 100
        insights.append(f"🏘️ **Concentração:** Top 3 bairros representam {percentage:.1f}% dos crimes")
    
    # Análise de diversidade criminal
    if 'tipo_crime' in df.columns and 'bairro' in df.columns:
        crime_diversity = df.groupby('bairro')['tipo_crime'].nunique().mean()
        insights.append(f"🎯 **Diversidade criminal:** Média de {crime_diversity:.1f} tipos de crime por bairro")
    
    # Análise de anomalias
    if 'anomalia' in df.columns:
        anomaly_rate = (len(df[df['anomalia'] == -1]) / len(df)) * 100
        insights.append(f"🚨 **Taxa de anomalias:** {anomaly_rate:.2f}% das ocorrências são atípicas")
    
    for insight in insights:
        st.info(insight)
    
    # Tabela de dados filtrados
    st.subheader("📋 Dados Filtrados (Amostra)")
    
    display_cols = ['data_ocorrencia', 'tipo_crime', 'bairro']
    if 'nivel_risco' in df.columns:
        display_cols.append('nivel_risco')
    if 'cluster' in df.columns:
        display_cols.append('cluster')
    
    st.dataframe(
        df[display_cols].head(20),
        use_container_width=True
    )

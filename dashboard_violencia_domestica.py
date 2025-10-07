import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta

# Configuração da página
st.set_page_config(
    page_title="Dashboard Violência Doméstica - DEAM",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado para violência doméstica
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #8B0000 0%, #DC143C 100%);
    padding: 1.5rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    border-left: 4px solid #DC143C;
    margin: 0.5rem 0;
}
.alert-box {
    background: #ffe6e6;
    border: 1px solid #ff9999;
    border-radius: 5px;
    padding: 1rem;
    margin: 1rem 0;
    color: #cc0000;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_violencia_domestica_data():
    """Carrega e filtra dados específicos de violência doméstica"""
    try:
        # Carregar dataset completo
        df = pd.read_csv('data/dataset_ocorrencias_delegacia_5.csv')
        
        # Filtrar apenas casos específicos: Estupro, Ameaça e Violência Doméstica
        tipos_violencia_domestica = [
            'Violência Doméstica',
            'Estupro', 
            'Ameaça'
        ]
        
        # Filtrar dados
        df_vd = df[df['tipo_crime'].isin(tipos_violencia_domestica)].copy()
        
        # Verificar se encontrou dados
        if len(df_vd) == 0:
            st.error("❌ Nenhum caso de violência doméstica encontrado no dataset.")
            return pd.DataFrame()
        
        # Processar dados
        df_vd['data_ocorrencia'] = pd.to_datetime(df_vd['data_ocorrencia'], errors='coerce')
        df_vd['hora'] = df_vd['data_ocorrencia'].dt.hour
        df_vd['dia_semana'] = df_vd['data_ocorrencia'].dt.day_name()
        df_vd['mes'] = df_vd['data_ocorrencia'].dt.month
        df_vd['ano'] = df_vd['data_ocorrencia'].dt.year
        
        # Mapear dias da semana para português
        day_map = {
            'Monday': 'Segunda-feira', 'Tuesday': 'Terça-feira',
            'Wednesday': 'Quarta-feira', 'Thursday': 'Quinta-feira',
            'Friday': 'Sexta-feira', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
        }
        df_vd['dia_semana'] = df_vd['dia_semana'].map(day_map)
        
        # Criar campos específicos para violência doméstica
        df_vd['genero_vitima'] = np.where(
            df_vd['sexo_suspeito'] == 'Feminino', 'Masculino', 'Feminino'
        )  # Inverter para representar vítima
        
        df_vd['faixa_etaria_vitima'] = pd.cut(
            df_vd['idade_suspeito'], 
            bins=[0, 18, 25, 35, 50, 100],
            labels=['Menor de 18', '18-25', '26-35', '36-50', 'Acima de 50']
        )
        
        # Usar status real da investigação
        df_vd['status_caso'] = df_vd['status_investigacao']
        
        return df_vd
        
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        return pd.DataFrame()

def create_sidebar_filters(df):
    """Cria filtros específicos para violência doméstica"""
    st.sidebar.markdown("## 🔍 Filtros - Violência Doméstica")
    
    # Filtro por tipo de violência
    tipos_disponiveis = sorted(df['tipo_crime'].unique())
    tipos_selecionados = st.sidebar.multiselect(
        "💔 Tipos de Violência",
        options=tipos_disponiveis,
        default=[],
        help="Selecione os tipos de violência para análise"
    )
    
    # Filtro por bairro
    bairros_disponiveis = sorted(df['bairro'].unique())
    bairros_selecionados = st.sidebar.multiselect(
        "🏘️ Bairros",
        options=bairros_disponiveis,
        default=[],
        help="Selecione os bairros para análise"
    )
    
    # Filtro por gênero da vítima
    generos_disponiveis = sorted(df['genero_vitima'].unique())
    generos_selecionados = st.sidebar.multiselect(
        "👤 Gênero da Vítima",
        options=generos_disponiveis,
        default=[]
    )
    
    # Filtro por status do caso
    status_disponiveis = sorted(df['status_caso'].unique())
    status_selecionados = st.sidebar.multiselect(
        "📋 Status do Caso",
        options=status_disponiveis,
        default=[]
    )
    
    # Filtro por período - limitado ao range do dataset
    if not df['data_ocorrencia'].isna().all():
        min_date = df['data_ocorrencia'].min().date()
        max_date = df['data_ocorrencia'].max().date()
        
        st.sidebar.markdown(f"**Período disponível:** {min_date.strftime('%d/%m/%Y')} a {max_date.strftime('%d/%m/%Y')}")
        
        data_range = st.sidebar.date_input(
            "📅 Período",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
    else:
        data_range = None
    
    return {
        'tipos': tipos_selecionados,
        'bairros': bairros_selecionados,
        'generos': generos_selecionados,
        'status': status_selecionados,
        'data_range': data_range
    }

def apply_filters(df, filters):
    """Aplica filtros aos dados"""
    df_filtered = df.copy()
    
    if filters['tipos']:
        df_filtered = df_filtered[df_filtered['tipo_crime'].isin(filters['tipos'])]
    
    if filters['bairros']:
        df_filtered = df_filtered[df_filtered['bairro'].isin(filters['bairros'])]
    
    if filters['generos']:
        df_filtered = df_filtered[df_filtered['genero_vitima'].isin(filters['generos'])]
    
    if filters['status']:
        df_filtered = df_filtered[df_filtered['status_caso'].isin(filters['status'])]
    
    if filters['data_range'] and len(filters['data_range']) == 2:
        start_date, end_date = filters['data_range']
        df_filtered = df_filtered[
            (df_filtered['data_ocorrencia'].dt.date >= start_date) &
            (df_filtered['data_ocorrencia'].dt.date <= end_date)
        ]
    
    return df_filtered

def show_kpis(df):
    """Exibe KPIs específicos de violência doméstica"""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_casos = len(df)
        st.metric("📊 Total de Casos", f"{total_casos:,}")
    
    with col2:
        casos_estupro = len(df[df['tipo_crime'] == 'Estupro'])
        st.metric("🚨 Estupro", f"{casos_estupro:,}")
    
    with col3:
        casos_ameaca = len(df[df['tipo_crime'] == 'Ameaça'])
        st.metric("⚠️ Ameaças", f"{casos_ameaca:,}")
    
    with col4:
        casos_concluidos = len(df[df['status_caso'] == 'Concluído'])
        st.metric("✅ Casos Concluídos", f"{casos_concluidos:,}")
    
    with col5:
        vitimas_femininas = len(df[df['genero_vitima'] == 'Feminino'])
        percentual = (vitimas_femininas / len(df) * 100) if len(df) > 0 else 0
        st.metric("👩 Vítimas Femininas", f"{vitimas_femininas:,} ({percentual:.1f}%)")

def show_mapa_violencia(df):
    """Exibe mapa específico de violência doméstica"""
    st.subheader("🗺️ Distribuição Geográfica dos Casos")
    
    if len(df) == 0:
        st.warning("Nenhum caso encontrado com os filtros aplicados.")
        return
    
    # Filtrar coordenadas válidas
    df_valid = df.dropna(subset=['latitude', 'longitude'])
    
    if len(df_valid) == 0:
        st.error("Nenhuma coordenada válida encontrada.")
        return
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("### 🎛️ Opções do Mapa")
        tipo_mapa = st.selectbox(
            "Tipo de Visualização",
            ["Mapa de Calor", "Pontos por Tipo", "Clusters por Bairro"]
        )
        
        destacar_graves = st.checkbox("Destacar Casos Graves", True)
    
    with col1:
        center_lat = df_valid['latitude'].mean()
        center_lon = df_valid['longitude'].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        
        if tipo_mapa == "Mapa de Calor":
            from folium.plugins import HeatMap
            heat_data = [[row['latitude'], row['longitude']] for _, row in df_valid.iterrows()]
            HeatMap(heat_data, radius=15, blur=10, gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'}).add_to(m)
        
        elif tipo_mapa == "Pontos por Tipo":
            cores_tipos = {
                'Estupro': 'red',
                'Violência Doméstica': 'darkred',
                'Ameaça': 'orange',
                'Lesão Corporal': 'purple',
                'Injúria': 'blue'
            }
            
            for _, row in df_valid.iterrows():
                cor = cores_tipos.get(row['tipo_crime'], 'gray')
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=6,
                    popup=f"<b>{row['tipo_crime']}</b><br>Bairro: {row['bairro']}<br>Data: {row['data_ocorrencia'].strftime('%d/%m/%Y') if pd.notna(row['data_ocorrencia']) else 'N/A'}",
                    color=cor,
                    fill=True,
                    fillColor=cor,
                    fillOpacity=0.7
                ).add_to(m)
        
        else:  # Clusters por Bairro
            from folium.plugins import MarkerCluster
            marker_cluster = MarkerCluster().add_to(m)
            
            for _, row in df_valid.iterrows():
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    popup=f"<b>{row['tipo_crime']}</b><br>Bairro: {row['bairro']}",
                    icon=folium.Icon(color='red', icon='exclamation-sign')
                ).add_to(marker_cluster)
        
        st_folium(m, use_container_width=True, height=500)

def show_analise_temporal(df):
    """Análise temporal específica para violência doméstica"""
    st.subheader("📈 Padrões Temporais da Violência Doméstica")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🕐 Distribuição por Hora do Dia")
        if 'hora' in df.columns:
            hourly_data = df['hora'].value_counts().sort_index()
            
            fig_hora = px.bar(
                x=hourly_data.index,
                y=hourly_data.values,
                title="Casos por Hora",
                labels={'x': 'Hora do Dia', 'y': 'Número de Casos'},
                color=hourly_data.values,
                color_continuous_scale='Reds'
            )
            fig_hora.update_layout(showlegend=False)
            st.plotly_chart(fig_hora, use_container_width=True)
            
            # Identificar horário crítico
            hora_pico = hourly_data.idxmax()
            st.info(f"🕐 **Horário mais crítico:** {hora_pico}:00 com {hourly_data[hora_pico]} casos")
    
    with col2:
        st.markdown("#### 📅 Distribuição por Dia da Semana")
        if 'dia_semana' in df.columns:
            dias_ordem = ['Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 'Sexta-feira', 'Sábado', 'Domingo']
            daily_data = df['dia_semana'].value_counts().reindex(dias_ordem).fillna(0)
            
            fig_dia = px.bar(
                x=daily_data.index,
                y=daily_data.values,
                title="Casos por Dia da Semana",
                labels={'x': 'Dia da Semana', 'y': 'Número de Casos'},
                color=daily_data.values,
                color_continuous_scale='Reds'
            )
            fig_dia.update_xaxes(tickangle=45)
            st.plotly_chart(fig_dia, use_container_width=True)
            
            # Identificar dia crítico
            dia_pico = daily_data.idxmax()
            st.info(f"📅 **Dia mais crítico:** {dia_pico} com {daily_data[dia_pico]} casos")

def show_perfil_vitimas(df):
    """Análise do perfil das vítimas"""
    st.subheader("👤 Perfil das Vítimas")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 👥 Distribuição por Gênero")
        genero_dist = df['genero_vitima'].value_counts()
        
        fig_genero = px.pie(
            values=genero_dist.values,
            names=genero_dist.index,
            title="Vítimas por Gênero",
            color_discrete_sequence=['#FF69B4', '#4169E1']
        )
        st.plotly_chart(fig_genero, use_container_width=True)
    
    with col2:
        st.markdown("#### 📊 Faixa Etária")
        if 'faixa_etaria_vitima' in df.columns:
            idade_dist = df['faixa_etaria_vitima'].value_counts()
            
            fig_idade = px.bar(
                x=idade_dist.index,
                y=idade_dist.values,
                title="Vítimas por Faixa Etária",
                color=idade_dist.values,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_idade, use_container_width=True)
    
    with col3:
        st.markdown("#### 📋 Status dos Casos")
        status_dist = df['status_caso'].value_counts()
        
        fig_status = px.pie(
            values=status_dist.values,
            names=status_dist.index,
            title="Status das Investigações",
            color_discrete_map={
                'Concluído': '#28a745',
                'Em Investigação': '#ffc107',
                'Arquivado': '#dc3545'
            }
        )
        st.plotly_chart(fig_status, use_container_width=True)

def show_estatisticas_bairros(df):
    """Estatísticas por bairro"""
    st.subheader("🏘️ Análise por Bairros")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📍 Top 10 Bairros com Mais Casos")
        bairro_stats = df['bairro'].value_counts().head(10)
        
        fig_bairros = px.bar(
            x=bairro_stats.values,
            y=bairro_stats.index,
            orientation='h',
            title="Casos por Bairro",
            color=bairro_stats.values,
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_bairros, use_container_width=True)
    
    with col2:
        st.markdown("#### 🎯 Tipos de Violência por Bairro")
        if len(df) > 0:
            # Pegar top 5 bairros e top 5 tipos
            top_bairros = df['bairro'].value_counts().head(5).index
            top_tipos = df['tipo_crime'].value_counts().head(5).index
            
            df_cross = df[df['bairro'].isin(top_bairros) & df['tipo_crime'].isin(top_tipos)]
            
            if len(df_cross) > 0:
                cross_tab = pd.crosstab(df_cross['bairro'], df_cross['tipo_crime'])
                
                fig_heatmap = px.imshow(
                    cross_tab.values,
                    x=cross_tab.columns,
                    y=cross_tab.index,
                    title="Heatmap: Bairro vs Tipo de Violência",
                    color_continuous_scale='Reds'
                )
                fig_heatmap.update_xaxes(tickangle=45)
                st.plotly_chart(fig_heatmap, use_container_width=True)

def main():
    """Função principal do dashboard"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Dashboard de Violência Doméstica</h1>  
        <p><strong>Alunos:</strong> Sabrina Vidal, Mario Beltrão, Gabriel Vidal, Matheus Eduardo, Beatriz, Mylena Lucena, Leonardo</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Carregar dados
    df = load_violencia_domestica_data()
    
    if df.empty:
        st.error("❌ Não foi possível carregar os dados.")
        return
    
    # Mostrar informações sobre os dados
    tipos_encontrados = df['tipo_crime'].value_counts()
    st.info(f"📊 **Dataset carregado:** {len(df)} casos encontrados | Estupro: {tipos_encontrados.get('Estupro', 0)} | Ameaça: {tipos_encontrados.get('Ameaça', 0)} | Violência Doméstica: {tipos_encontrados.get('Violência Doméstica', 0)}")
    
    # Filtros
    filters = create_sidebar_filters(df)
    df_filtered = apply_filters(df, filters)
    
    # Verificar se há dados após filtros
    if len(df_filtered) == 0:
        st.warning("⚠️ Nenhum caso encontrado com os filtros aplicados. Ajuste os filtros.")
        return
    
    # KPIs
    show_kpis(df_filtered)
    
    # Abas principais
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗺️ Mapa Geográfico",
        "📈 Análise Temporal",
        "👤 Perfil das Vítimas",
        "🏘️ Análise por Bairros",
        "📋 Dados Detalhados"
    ])
    
    with tab1:
        show_mapa_violencia(df_filtered)
    
    with tab2:
        show_analise_temporal(df_filtered)
    
    with tab3:
        show_perfil_vitimas(df_filtered)
    
    with tab4:
        show_estatisticas_bairros(df_filtered)
    
    with tab5:
        st.subheader("📋 Dados Detalhados")
        
        # Mostrar estatísticas resumidas
        st.markdown("### 📊 Resumo Estatístico")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total de Registros", len(df_filtered))
        with col2:
            st.metric("Bairros Únicos", df_filtered['bairro'].nunique())
        with col3:
            st.metric("Tipos de Violência", df_filtered['tipo_crime'].nunique())
        
        # Tabela de dados
        st.markdown("### 📄 Tabela de Casos")
        colunas_exibir = [
            'data_ocorrencia', 'tipo_crime', 'bairro', 'genero_vitima', 
            'status_caso', 'orgao_responsavel'
        ]
        
        df_display = df_filtered[colunas_exibir].copy()
        df_display['data_ocorrencia'] = df_display['data_ocorrencia'].dt.strftime('%d/%m/%Y %H:%M')
        
        st.dataframe(
            df_display,
            use_container_width=True,
            height=400
        )
        
        # Opção de download
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            label="📥 Baixar dados filtrados (CSV)",
            data=csv,
            file_name=f"violencia_domestica_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Rodapé
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #666;'>Dashboard de Violência Doméstica - DEAM | Projeto Acadêmico</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

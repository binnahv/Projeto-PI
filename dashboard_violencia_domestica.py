import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
from security_utils import DataValidator

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

def clean_geographic_data(df, show_info=False):
    """Limpa dados geográficos removendo coordenadas inválidas"""
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        return df
    
    # Aplicar validação geográfica
    if 'bairro' in df.columns:
        # Validação individual por bairro para cada linha
        valid_coords = df.apply(
            lambda row: DataValidator.validate_coordinates_by_neighborhood(
                row['latitude'], row['longitude'], row['bairro']
            ), 
            axis=1
        )
        
        removed_count = len(df) - valid_coords.sum()
        if removed_count > 0 and show_info:
            st.info(f"🧹 **Limpeza geográfica:** {removed_count} pontos com coordenadas inválidas foram removidos (fora dos limites dos bairros)")
    else:
        # Fallback para validação geral mais restritiva
        valid_coords = df.apply(
            lambda row: DataValidator.validate_coordinates(row['latitude'], row['longitude']), 
            axis=1
        )
        
        removed_count = len(df) - valid_coords.sum()
        if removed_count > 0 and show_info:
            st.info(f"🧹 **Limpeza geográfica:** {removed_count} pontos com coordenadas inválidas foram removidos")
    
    return df[valid_coords].copy()

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
    st.sidebar.markdown("## Filtros - Violência Doméstica")
    
    # Filtro por tipo de violência
    tipos_disponiveis = sorted(df['tipo_crime'].unique())
    tipos_selecionados = st.sidebar.multiselect(
        "Tipos de Violência",
        options=tipos_disponiveis,
        default=[],
        help="Selecione os tipos de violência para análise"
    )
    
    # Filtro por bairro
    bairros_disponiveis = sorted(df['bairro'].unique())
    bairros_selecionados = st.sidebar.multiselect(
        "Bairros",
        options=bairros_disponiveis,
        default=[],
        help="Selecione os bairros para análise"
    )
    
    # Filtro por gênero da vítima
    generos_disponiveis = sorted(df['genero_vitima'].unique())
    generos_selecionados = st.sidebar.multiselect(
        "Gênero da Vítima",
        options=generos_disponiveis,
        default=[]
    )
    
    # Filtro por status do caso
    status_disponiveis = sorted(df['status_caso'].unique())
    status_selecionados = st.sidebar.multiselect(
        "Status do Caso",
        options=status_disponiveis,
        default=[]
    )
    
    # Filtro por período - limitado ao range do dataset
    if not df['data_ocorrencia'].isna().all():
        min_date = df['data_ocorrencia'].min().date()
        max_date = df['data_ocorrencia'].max().date()
        
        st.sidebar.markdown(f"**Período disponível:** {min_date.strftime('%d/%m/%Y')} a {max_date.strftime('%d/%m/%Y')}")
        
        data_range = st.sidebar.date_input(
            "Período",
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
    """Exibe KPIs específicos para Polícia Civil"""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_casos = len(df)
        st.metric("Total de Casos", f"{total_casos:,}")
    
    with col2:
        # Usar status_caso que é o campo mapeado corretamente
        casos_concluidos = len(df[df['status_caso'] == 'Concluído']) if 'status_caso' in df.columns else 0
        taxa_resolucao = (casos_concluidos / total_casos * 100) if total_casos > 0 else 0
        st.metric("Taxa de Resolução", f"{taxa_resolucao:.1f}%")
    
    with col3:
        casos_arma_fogo = len(df[df['arma_utilizada'] == 'Arma de Fogo']) if 'arma_utilizada' in df.columns else 0
        st.metric("Casos c/ Arma de Fogo", f"{casos_arma_fogo:,}")
    
    with col4:
        casos_multiplas_vitimas = len(df[df['quantidade_vitimas'] > 1]) if 'quantidade_vitimas' in df.columns else 0
        st.metric("Múltiplas Vítimas", f"{casos_multiplas_vitimas:,}")
    
    with col5:
        # Garantir que atualiza com os filtros aplicados
        delegacias_ativas = df['orgao_responsavel'].nunique() if 'orgao_responsavel' in df.columns and len(df) > 0 else 0
        st.metric("Delegacias Envolvidas", f"{delegacias_ativas:,}")

def show_mapa_violencia(df):
    """Exibe mapa específico de violência doméstica"""
    st.subheader("Distribuição Geográfica dos Casos")
    
    if len(df) == 0:
        st.warning("Nenhum caso encontrado com os filtros aplicados.")
        return
    
    # Aplicar limpeza geográfica
    df_clean = clean_geographic_data(df, show_info=True)
    
    if len(df_clean) == 0:
        st.error("Nenhuma coordenada válida encontrada após limpeza geográfica.")
        return
    
    # Informações dos dados filtrados
    if 'bairro' in df_clean.columns:
        bairros_unicos = df_clean['bairro'].unique()
        st.info(f"**Dados no mapa:** {len(df_clean)} ocorrências em {len(bairros_unicos)} bairro(s): {', '.join(sorted(bairros_unicos))}")
        
        # Verificar se há inconsistências geográficas
        if len(bairros_unicos) == 1:
            bairro_selecionado = bairros_unicos[0]
            coords_range = f"Lat: {df_clean['latitude'].min():.4f} a {df_clean['latitude'].max():.4f}, Lon: {df_clean['longitude'].min():.4f} a {df_clean['longitude'].max():.4f}"
            st.write(f"**Range de coordenadas para {bairro_selecionado}:** {coords_range}")
    else:
        st.info(f"**Dados no mapa:** {len(df_clean)} ocorrências")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("### Opções do Mapa")
        tipo_mapa = st.selectbox(
            "Tipo de Visualização",
            ["Mapa de Calor", "Pontos por Tipo", "Clusters por Bairro"]
        )
        
        destacar_graves = st.checkbox("Destacar Casos Graves", True)
        
        st.markdown("### Controles")
        if st.button("Atualizar Mapa", help="Força atualização do mapa com filtros atuais"):
            st.cache_data.clear()
            st.rerun()
        
        # Mostrar estatísticas de limpeza
        total_original = len(df)
        total_limpo = len(df_clean)
        removidos = total_original - total_limpo
        
        if removidos > 0:
            st.warning(f"**Dados removidos:** {removidos} de {total_original} ({(removidos/total_original*100):.1f}%)")
        else:
            st.success(f"**Todos os dados válidos:** {total_limpo} ocorrências")
    
    with col1:
        center_lat = df_clean['latitude'].mean()
        center_lon = df_clean['longitude'].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        
        if tipo_mapa == "Mapa de Calor":
            from folium.plugins import HeatMap
            heat_data = [[row['latitude'], row['longitude']] for _, row in df_clean.iterrows()]
            HeatMap(heat_data, radius=15, blur=10, gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'}).add_to(m)
        
        elif tipo_mapa == "Pontos por Tipo":
            cores_tipos = {
                'Estupro': 'red',
                'Violência Doméstica': 'darkred',
                'Ameaça': 'orange',
                'Lesão Corporal': 'purple',
                'Injúria': 'blue'
            }
            
            for _, row in df_clean.iterrows():
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
            
            for _, row in df_clean.iterrows():
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    popup=f"<b>{row['tipo_crime']}</b><br>Bairro: {row['bairro']}",
                    icon=folium.Icon(color='red', icon='exclamation-sign')
                ).add_to(marker_cluster)
        
        st_folium(m, use_container_width=True, height=500)

def show_analise_temporal(df):
    """Análise temporal específica para violência doméstica"""
    st.subheader("Padrões Temporais da Violência Doméstica")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Distribuição por Hora do Dia")
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
            st.info(f"**Horário mais crítico:** {hora_pico}:00 com {hourly_data[hora_pico]} casos")
    
    with col2:
        st.markdown("#### Distribuição por Dia da Semana")
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
            st.info(f"**Dia mais crítico:** {dia_pico} com {daily_data[dia_pico]} casos")

def show_perfil_suspeitos(df):
    """Análise do perfil dos suspeitos para Polícia Civil"""
    st.subheader("Perfil dos Suspeitos - Inteligência Policial")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Gênero dos Suspeitos")
        if 'sexo_suspeito' in df.columns:
            genero_suspeito = df['sexo_suspeito'].value_counts()
            
            fig_genero = px.pie(
                values=genero_suspeito.values,
                names=genero_suspeito.index,
                title="Suspeitos por Gênero",
                color_discrete_sequence=['#1f77b4', '#ff7f0e', '#d62728']
            )
            st.plotly_chart(fig_genero, use_container_width=True)
            
            # Insight policial
            if 'Masculino' in genero_suspeito.index:
                perc_masc = (genero_suspeito['Masculino'] / genero_suspeito.sum() * 100)
                st.info(f"**Insight:** {perc_masc:.1f}% dos suspeitos são do sexo masculino")
    
    with col2:
        st.markdown("#### Faixa Etária dos Suspeitos")
        if 'idade_suspeito' in df.columns:
            # Criar faixas etárias
            df_idade = df.dropna(subset=['idade_suspeito'])
            if len(df_idade) > 0:
                df_idade['faixa_etaria'] = pd.cut(
                    df_idade['idade_suspeito'], 
                    bins=[0, 18, 25, 35, 50, 100],
                    labels=['<18', '18-25', '26-35', '36-50', '>50']
                )
                
                idade_dist = df_idade['faixa_etaria'].value_counts()
                
                fig_idade = px.bar(
                    x=idade_dist.index,
                    y=idade_dist.values,
                    title="Suspeitos por Faixa Etária",
                    color=idade_dist.values,
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_idade, use_container_width=True)
                
                # Insight policial
                faixa_principal = idade_dist.idxmax()
                st.info(f"**Perfil predominante:** {faixa_principal} anos ({idade_dist[faixa_principal]} casos)")
    
    with col3:
        st.markdown("#### Quantidade de Suspeitos")
        if 'quantidade_suspeitos' in df.columns:
            qtd_suspeitos = df['quantidade_suspeitos'].value_counts().sort_index()
            
            fig_qtd = px.bar(
                x=qtd_suspeitos.index,
                y=qtd_suspeitos.values,
                title="Casos por Nº de Suspeitos",
                color=qtd_suspeitos.values,
                color_continuous_scale='Oranges'
            )
            st.plotly_chart(fig_qtd, use_container_width=True)
            
            # Insight policial
            casos_multiplos = len(df[df['quantidade_suspeitos'] > 1])
            perc_multiplos = (casos_multiplos / len(df) * 100) if len(df) > 0 else 0
            st.info(f"**Casos com múltiplos suspeitos:** {perc_multiplos:.1f}%")
    
    # Análise cruzada por tipo de crime
    st.markdown("---")
    st.markdown("### Análise por Tipo de Crime")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Perfil por Crime")
        if 'sexo_suspeito' in df.columns and 'tipo_crime' in df.columns:
            cross_tab = pd.crosstab(df['tipo_crime'], df['sexo_suspeito'], normalize='index') * 100
            
            fig_cross = px.imshow(
                cross_tab.values,
                x=cross_tab.columns,
                y=cross_tab.index,
                title="% Gênero do Suspeito por Tipo de Crime",
                color_continuous_scale='RdYlBu',
                text_auto='.1f'
            )
            st.plotly_chart(fig_cross, use_container_width=True)
    
    with col2:
        st.markdown("#### Idade Média por Crime")
        if 'idade_suspeito' in df.columns and 'tipo_crime' in df.columns:
            idade_media = df.groupby('tipo_crime')['idade_suspeito'].mean().sort_values(ascending=False)
            
            fig_idade_media = px.bar(
                x=idade_media.values,
                y=idade_media.index,
                orientation='h',
                title="Idade Média dos Suspeitos",
                color=idade_media.values,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_idade_media, use_container_width=True)

def show_efetividade_policial(df):
    """Análise da efetividade das investigações policiais"""
    st.subheader("Efetividade Policial - Indicadores de Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Taxa de Resolução Geral")
        if 'status_caso' in df.columns:
            status_counts = df['status_caso'].value_counts()
            total_casos = len(df)
            concluidos = status_counts.get('Concluído', 0)
            taxa_resolucao = (concluidos / total_casos * 100) if total_casos > 0 else 0
            
            st.metric("Taxa de Resolução", f"{taxa_resolucao:.1f}%")
            
            # Gráfico de status
            fig_status = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Status das Investigações",
                color_discrete_map={
                    'Concluído': '#28a745',
                    'Em Investigação': '#ffc107', 
                    'Arquivado': '#dc3545'
                }
            )
            st.plotly_chart(fig_status, use_container_width=True)
    
    with col2:
        st.markdown("#### Performance por Delegacia")
        if 'orgao_responsavel' in df.columns:
            delegacia_stats = df.groupby('orgao_responsavel').agg({
                'status_caso': lambda x: (x == 'Concluído').sum() / len(x) * 100
            }).round(1)
            delegacia_stats.columns = ['Taxa_Resolucao']
            delegacia_stats = delegacia_stats.sort_values('Taxa_Resolucao', ascending=False)
            
            fig_delegacia = px.bar(
                x=delegacia_stats.values.flatten(),
                y=delegacia_stats.index,
                orientation='h',
                title="Taxa de Resolução por Delegacia (%)",
                color=delegacia_stats.values.flatten(),
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_delegacia, use_container_width=True)
            
            # Melhor e pior delegacia
            melhor = delegacia_stats.index[0]
            pior = delegacia_stats.index[-1]
            st.info(f"**Melhor:** {melhor} ({delegacia_stats.iloc[0,0]:.1f}%)")
            st.warning(f"**Precisa melhorar:** {pior} ({delegacia_stats.iloc[-1,0]:.1f}%)")
    
    with col3:
        st.markdown("#### Efetividade por Tipo de Crime")
        if 'tipo_crime' in df.columns:
            crime_stats = df.groupby('tipo_crime').agg({
                'status_caso': lambda x: (x == 'Concluído').sum() / len(x) * 100
            }).round(1)
            crime_stats.columns = ['Taxa_Resolucao']
            crime_stats = crime_stats.sort_values('Taxa_Resolucao', ascending=False)
            
            fig_crime = px.bar(
                x=crime_stats.index,
                y=crime_stats.values.flatten(),
                title="Taxa de Resolução por Crime (%)",
                color=crime_stats.values.flatten(),
                color_continuous_scale='RdYlGn'
            )
            fig_crime.update_xaxes(tickangle=45)
            st.plotly_chart(fig_crime, use_container_width=True)

def show_modus_operandi(df):
    """Análise dos padrões de modus operandi"""
    st.subheader("Modus Operandi - Padrões Criminais")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Armas Utilizadas")
        if 'arma_utilizada' in df.columns:
            armas = df['arma_utilizada'].value_counts().head(8)
            
            fig_armas = px.bar(
                x=armas.values,
                y=armas.index,
                orientation='h',
                title="Tipos de Arma Mais Utilizados",
                color=armas.values,
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_armas, use_container_width=True)
            
            # Insight sobre armas
            arma_principal = armas.index[0] if len(armas) > 0 else "N/A"
            casos_arma = armas.iloc[0] if len(armas) > 0 else 0
            st.info(f"**Arma predominante:** {arma_principal} ({casos_arma} casos)")
    
    with col2:
        st.markdown("#### Descrição do Modus Operandi")
        if 'descricao_modus_operandi' in df.columns:
            modus = df['descricao_modus_operandi'].value_counts().head(8)
            
            fig_modus = px.bar(
                x=modus.values,
                y=modus.index,
                orientation='h',
                title="Padrões de Modus Operandi",
                color=modus.values,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_modus, use_container_width=True)
            
            # Insight sobre modus
            modus_principal = modus.index[0] if len(modus) > 0 else "N/A"
            casos_modus = modus.iloc[0] if len(modus) > 0 else 0
            st.info(f"**Padrão predominante:** {modus_principal} ({casos_modus} casos)")
    
    # Análise cruzada
    st.markdown("---")
    st.markdown("### Correlação: Arma vs Tipo de Crime")
    
    if 'arma_utilizada' in df.columns and 'tipo_crime' in df.columns:
        # Criar tabela cruzada
        cross_arma_crime = pd.crosstab(df['tipo_crime'], df['arma_utilizada'])
        
        # Mostrar apenas as 5 armas mais comuns
        top_armas = df['arma_utilizada'].value_counts().head(5).index
        cross_filtered = cross_arma_crime[top_armas]
        
        fig_cross = px.imshow(
            cross_filtered.values,
            x=cross_filtered.columns,
            y=cross_filtered.index,
            title="Heatmap: Tipo de Crime vs Arma Utilizada",
            color_continuous_scale='Reds',
            text_auto=True
        )
        st.plotly_chart(fig_cross, use_container_width=True)
        
        # Insights operacionais
        st.markdown("#### Insights Operacionais:")
        for crime in df['tipo_crime'].unique():
            crime_data = df[df['tipo_crime'] == crime]
            if len(crime_data) > 0:
                arma_mais_usada = crime_data['arma_utilizada'].mode()
                if len(arma_mais_usada) > 0:
                    st.markdown(f"- **{crime}:** Arma predominante = {arma_mais_usada.iloc[0]}")

def show_correlacoes(df):
    """Análise de correlações e padrões"""
    st.subheader("Correlações e Padrões Criminais")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Vítimas vs Gravidade")
        if 'quantidade_vitimas' in df.columns and 'arma_utilizada' in df.columns:
            # Análise de múltiplas vítimas
            df_analise = df.copy()
            df_analise['multiplas_vitimas'] = df_analise['quantidade_vitimas'] > 1
            df_analise['arma_letal'] = df_analise['arma_utilizada'].isin(['Arma de Fogo', 'Faca'])
            
            cross_vitimas = pd.crosstab(df_analise['multiplas_vitimas'], df_analise['arma_letal'], normalize='index') * 100
            
            fig_vitimas = px.imshow(
                cross_vitimas.values,
                x=['Arma Não Letal', 'Arma Letal'],
                y=['Vítima Única', 'Múltiplas Vítimas'],
                title="% Uso de Arma Letal vs Nº de Vítimas",
                color_continuous_scale='Reds',
                text_auto='.1f'
            )
            st.plotly_chart(fig_vitimas, use_container_width=True)
    
    with col2:
        st.markdown("#### Sazonalidade por Crime")
        if 'data_ocorrencia' in df.columns:
            df_temp = df.copy()
            df_temp['mes'] = df_temp['data_ocorrencia'].dt.month
            
            sazonalidade = df_temp.groupby(['mes', 'tipo_crime']).size().unstack(fill_value=0)
            
            fig_sazon = px.line(
                sazonalidade,
                title="Sazonalidade dos Crimes por Mês",
                labels={'index': 'Mês', 'value': 'Número de Casos'}
            )
            st.plotly_chart(fig_sazon, use_container_width=True)
    
    # Indicadores de risco
    st.markdown("---")
    st.markdown("### Indicadores de Risco")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Casos com arma de fogo
        if 'arma_utilizada' in df.columns:
            casos_arma_fogo = len(df[df['arma_utilizada'] == 'Arma de Fogo'])
            perc_arma_fogo = (casos_arma_fogo / len(df) * 100) if len(df) > 0 else 0
            st.metric("Casos com Arma de Fogo", f"{casos_arma_fogo}", f"{perc_arma_fogo:.1f}%")
    
    with col2:
        # Casos com múltiplas vítimas
        if 'quantidade_vitimas' in df.columns:
            casos_multiplas = len(df[df['quantidade_vitimas'] > 1])
            perc_multiplas = (casos_multiplas / len(df) * 100) if len(df) > 0 else 0
            st.metric("Múltiplas Vítimas", f"{casos_multiplas}", f"{perc_multiplas:.1f}%")
    
    with col3:
        # Casos com múltiplos suspeitos
        if 'quantidade_suspeitos' in df.columns:
            casos_mult_susp = len(df[df['quantidade_suspeitos'] > 1])
            perc_mult_susp = (casos_mult_susp / len(df) * 100) if len(df) > 0 else 0
            st.metric("Múltiplos Suspeitos", f"{casos_mult_susp}", f"{perc_mult_susp:.1f}%")

def show_estatisticas_bairros(df):
    """Estatísticas por bairro"""
    st.subheader("Análise por Bairros")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Top 10 Bairros com Mais Casos")
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
        st.markdown("#### Tipos de Violência por Bairro")
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
        st.error("Não foi possível carregar os dados.")
        return
    
    # Mostrar informações sobre os dados
    tipos_encontrados = df['tipo_crime'].value_counts()
    st.info(f"**Dataset carregado:** {len(df)} casos encontrados | Estupro: {tipos_encontrados.get('Estupro', 0)} | Ameaça: {tipos_encontrados.get('Ameaça', 0)} | Violência Doméstica: {tipos_encontrados.get('Violência Doméstica', 0)}")
    
    # Aplicar limpeza geográfica inicial para estatísticas
    df_clean_stats = clean_geographic_data(df)
    if len(df_clean_stats) < len(df):
        removidos = len(df) - len(df_clean_stats)
        st.warning(f"**Limpeza geográfica aplicada:** {removidos} casos com coordenadas inválidas foram identificados ({(removidos/len(df)*100):.1f}% do total)")
    
    # Filtros
    filters = create_sidebar_filters(df)
    df_filtered = apply_filters(df, filters)
    
    # Verificar se há dados após filtros
    if len(df_filtered) == 0:
        st.warning("Nenhum caso encontrado com os filtros aplicados. Ajuste os filtros.")
        return
    
    # KPIs
    show_kpis(df_filtered)
    
    # Abas principais
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Mapa Geográfico",
        "Análise Temporal",
        "Perfil dos Suspeitos",
        "Efetividade Policial",
        "Modus Operandi",
        "Correlações",
        "Dados Detalhados"
    ])
    
    with tab1:
        show_mapa_violencia(df_filtered)
    
    with tab2:
        show_analise_temporal(df_filtered)
    
    with tab3:
        show_perfil_suspeitos(df_filtered)
    
    with tab4:
        show_efetividade_policial(df_filtered)
    
    with tab5:
        show_modus_operandi(df_filtered)
    
    with tab6:
        show_correlacoes(df_filtered)
    
    with tab7:
        st.subheader("Dados Detalhados")
        
        # Mostrar estatísticas resumidas
        st.markdown("### Resumo Estatístico")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total de Registros", len(df_filtered))
        with col2:
            st.metric("Bairros Únicos", df_filtered['bairro'].nunique())
        with col3:
            st.metric("Tipos de Violência", df_filtered['tipo_crime'].nunique())
        
        # Tabela de dados
        st.markdown("### Tabela de Casos")
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
            label="Baixar dados filtrados (CSV)",
            data=csv,
            file_name=f"violencia_domestica_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Rodapé
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #666;'>Dashboard de Violência Doméstica - DEAM | Projeto Acadêmico</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

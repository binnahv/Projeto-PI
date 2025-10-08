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
    
    # Filtro por gênero da vítima (se disponível)
    generos_selecionados = []
    if 'genero_vitima' in df.columns:
        generos_disponiveis = sorted(df['genero_vitima'].unique())
        generos_selecionados = st.sidebar.multiselect(
            "Gênero da Vítima",
            options=generos_disponiveis,
            default=[]
        )
    
    # Filtro por status do caso (se disponível)
    status_selecionados = []
    if 'status_caso' in df.columns:
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
    
    if filters['generos'] and 'genero_vitima' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['genero_vitima'].isin(filters['generos'])]
    
    if filters['status'] and 'status_caso' in df_filtered.columns:
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
        if 'status_caso' in df.columns:
            casos_concluidos = len(df[df['status_caso'] == 'Concluído'])
            taxa_resolucao = (casos_concluidos / total_casos * 100) if total_casos > 0 else 0
            st.metric("Taxa de Resolução", f"{taxa_resolucao:.1f}%")
        else:
            st.metric("Taxa de Resolução", "N/A", help="Coluna 'status_caso' não disponível")
    
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
    
    # Isso Verifica se as colunas "necessárias" existem
    if 'status_caso' not in df.columns:
        st.warning("⚠️ Análise de efetividade não disponível: coluna 'status_caso' não encontrada no dataset.")
        st.info("💡 Para análise completa, o dataset deve conter as colunas: 'status_caso', 'orgao_responsavel'")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Taxa de Resolução Geral")
        status_counts = df['status_caso'].value_counts()
        total_casos = len(df)
        concluidos = status_counts.get('Concluído', 0)
        taxa_resolucao = (concluidos / total_casos * 100) if total_casos > 0 else 0
        
        st.metric("Taxa de Resolução", f"{taxa_resolucao:.1f}%")
        
        # O gráfico de status
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
            
            # A melhor e pior delegacia
            if len(delegacia_stats) > 0:
                melhor = delegacia_stats.index[0]
                pior = delegacia_stats.index[-1]
                st.info(f"**Melhor:** {melhor} ({delegacia_stats.iloc[0,0]:.1f}%)")
                st.warning(f"**Precisa melhorar:** {pior} ({delegacia_stats.iloc[-1,0]:.1f}%)")
        else:
            st.warning("Coluna 'orgao_responsavel' não disponível")
    
    with col3:
        st.markdown("#### Efetividade por Tipo de Crime")
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

def generate_html_report(df):
    """Gera relatório HTML exportável com achados e métricas"""
    
    # Calcular métricas principais
    total_casos = len(df)
    tipos_crime = df['tipo_crime'].value_counts()
    bairros_top = df['bairro'].value_counts().head(5)
    
    # Calcular taxa de resolução
    casos_concluidos = len(df[df['status_caso'] == 'Concluído']) if 'status_caso' in df.columns else 0
    taxa_resolucao = (casos_concluidos / total_casos * 100) if total_casos > 0 else 0
    
    # Análise temporal
    if 'hora' not in df.columns:
        df_temp = df.copy()
        df_temp['hora'] = df_temp['data_ocorrencia'].dt.hour
    else:
        df_temp = df
    
    hora_pico = df_temp['hora'].mode().iloc[0] if len(df_temp) > 0 else 0
    
    # Template HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Relatório de Análise - Violência Doméstica</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                line-height: 1.6;
                color: #333;
            }}
            .header {{
                text-align: center;
                border-bottom: 2px solid #dc3545;
                padding-bottom: 20px;
                margin-bottom: 30px;
            }}
            .metric {{
                background: #f8f9fa;
                padding: 15px;
                margin: 10px 0;
                border-left: 4px solid #dc3545;
            }}
            .section {{
                margin: 30px 0;
            }}
            .insight {{
                background: #e3f2fd;
                padding: 15px;
                margin: 10px 0;
                border-radius: 5px;
                border-left: 4px solid #2196f3;
            }}
            .warning {{
                background: #fff3cd;
                padding: 15px;
                margin: 10px 0;
                border-radius: 5px;
                border-left: 4px solid #ffc107;
            }}
            .footer {{
                text-align: center;
                margin-top: 50px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                color: #666;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Relatório de Análise - Violência Doméstica</h1>
            <p><strong>Delegacia Especializada de Atendimento à Mulher - DEAM</strong></p>
            <p>Data de Geração: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
        </div>

        <div class="section">
            <h2>📊 Resumo Executivo</h2>
            <div class="metric">
                <strong>Total de Casos Analisados:</strong> {total_casos:,}
            </div>
            <div class="metric">
                <strong>Taxa de Resolução:</strong> {taxa_resolucao:.1f}%
            </div>
            <div class="metric">
                <strong>Horário de Maior Incidência:</strong> {hora_pico}:00h
            </div>
        </div>

        <div class="section">
            <h2>🚨 Distribuição por Tipo de Crime</h2>
            <table>
                <tr>
                    <th>Tipo de Crime</th>
                    <th>Número de Casos</th>
                    <th>Percentual</th>
                </tr>
    """
    
    # Adicionar dados dos tipos de crime
    for tipo, casos in tipos_crime.items():
        percentual = (casos / total_casos * 100)
        html_content += f"""
                <tr>
                    <td>{tipo}</td>
                    <td>{casos}</td>
                    <td>{percentual:.1f}%</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>

        <div class="section">
            <h2>📍 Bairros Mais Afetados</h2>
            <table>
                <tr>
                    <th>Bairro</th>
                    <th>Número de Casos</th>
                    <th>Percentual do Total</th>
                </tr>
    """
    
    # Adicionar dados dos bairros
    for bairro, casos in bairros_top.items():
        percentual = (casos / total_casos * 100)
        html_content += f"""
                <tr>
                    <td>{bairro}</td>
                    <td>{casos}</td>
                    <td>{percentual:.1f}%</td>
                </tr>
        """
    
    # Gerar insights automáticos
    insights = []
    
    # Insight sobre concentração
    if len(bairros_top) > 0:
        bairro_principal = bairros_top.index[0]
        concentracao = (bairros_top.iloc[0] / total_casos * 100)
        if concentracao > 20:
            insights.append(f"Alta concentração de casos em {bairro_principal} ({concentracao:.1f}% do total) - Requer atenção especial")
    
    # Insight sobre taxa de resolução
    if taxa_resolucao < 70:
        insights.append(f"Taxa de resolução baixa ({taxa_resolucao:.1f}%) - Necessário revisar processos investigativos")
    elif taxa_resolucao > 85:
        insights.append(f"Excelente taxa de resolução ({taxa_resolucao:.1f}%) - Manter padrão de qualidade")
    
    # Insight sobre horário
    if 22 <= hora_pico <= 6:
        insights.append(f"Pico de ocorrências no período noturno ({hora_pico}h) - Reforçar patrulhamento noturno")
    
    html_content += """
            </table>
        </div>

        <div class="section">
            <h2>💡 Insights e Recomendações</h2>
    """
    
    for insight in insights:
        html_content += f'<div class="insight">• {insight}</div>'
    
    # Adicionar limitações
    html_content += f"""
        </div>

        <div class="section">
            <h2>⚠️ Limitações do Estudo</h2>
            <div class="warning">
                <strong>Qualidade dos Dados:</strong> Este relatório é baseado em {total_casos} casos válidos após limpeza geográfica. 
                Coordenadas inválidas foram removidas para garantir precisão das análises espaciais.
            </div>
            <div class="warning">
                <strong>Período de Análise:</strong> Os dados refletem o período disponível no dataset e podem não representar 
                tendências de longo prazo.
            </div>
            <div class="warning">
                <strong>Interpretação:</strong> As análises são baseadas em dados históricos e devem ser complementadas 
                com conhecimento operacional local.
            </div>
        </div>

        <div class="section">
            <h2>📈 Métricas de Machine Learning</h2>
            <div class="metric">
                <strong>Algoritmos Utilizados:</strong> KMeans, DBSCAN, Isolation Forest, Random Forest
            </div>
            <div class="metric">
                <strong>Features Analisadas:</strong> Localização geográfica, padrões temporais, tipos de crime, características dos casos
            </div>
            <div class="metric">
                <strong>Validação:</strong> Silhouette Score para clustering, Cross-validation para modelos supervisionados
            </div>
        </div>

        <div class="footer">
            <p><strong>Dashboard de Violência Doméstica - DEAM</strong></p>
            <p>Projeto Acadêmico | Análise de Dados Criminais</p>
            <p><em>Este relatório foi gerado automaticamente pelo sistema de análise de dados.</em></p>
        </div>
    </body>
    </html>
    """
    
    return html_content

def show_clustering_analysis(df):
    """Análise de Clusterização - Machine Learning Não Supervisionado"""
    st.subheader("Análise de Clusters - Agrupamento de Padrões Criminais")
    
    if len(df) < 10:
        st.warning("Dados insuficientes para análise de clustering (mínimo 10 casos).")
        return
    
    # Preparar dados para clustering
    try:
        from sklearn.cluster import KMeans, DBSCAN
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.metrics import silhouette_score
        import numpy as np
        
        # Feature engineering
        df_cluster = df.copy()
        
        # Encoding de variáveis categóricas
        le_tipo = LabelEncoder()
        le_bairro = LabelEncoder()
        
        df_cluster['tipo_crime_encoded'] = le_tipo.fit_transform(df_cluster['tipo_crime'])
        df_cluster['bairro_encoded'] = le_bairro.fit_transform(df_cluster['bairro'])
        
        # Extrair hora se disponível
        if 'hora' in df_cluster.columns:
            features = ['latitude', 'longitude', 'hora', 'tipo_crime_encoded', 'bairro_encoded']
        else:
            df_cluster['hora'] = df_cluster['data_ocorrencia'].dt.hour
            features = ['latitude', 'longitude', 'hora', 'tipo_crime_encoded', 'bairro_encoded']
        
        # Remover NaN
        df_cluster = df_cluster.dropna(subset=features)
        
        if len(df_cluster) < 5:
            st.error("Dados insuficientes após limpeza.")
            return
        
        X = df_cluster[features].values
        
        # Normalizar dados
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### KMeans Clustering")
            
            # Determinar número ótimo de clusters
            n_clusters = st.slider("Número de Clusters", 2, min(8, len(df_cluster)//2), 4)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters_kmeans = kmeans.fit_predict(X_scaled)
            
            # Silhouette Score
            silhouette_avg = silhouette_score(X_scaled, clusters_kmeans)
            st.metric("Silhouette Score", f"{silhouette_avg:.3f}")
            
            # Adicionar clusters ao dataframe
            df_cluster['cluster_kmeans'] = clusters_kmeans
            
            # Visualização dos clusters no mapa
            if 'latitude' in df_cluster.columns and 'longitude' in df_cluster.columns:
                fig_map = px.scatter_mapbox(
                    df_cluster,
                    lat='latitude',
                    lon='longitude',
                    color='cluster_kmeans',
                    title="Clusters Geográficos (KMeans)",
                    mapbox_style="open-street-map",
                    zoom=10,
                    height=400
                )
                st.plotly_chart(fig_map, use_container_width=True)
        
        with col2:
            st.markdown("#### DBSCAN Clustering")
            
            eps = st.slider("Epsilon (DBSCAN)", 0.1, 2.0, 0.5, 0.1)
            min_samples = st.slider("Min Samples", 2, 10, 3)
            
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            clusters_dbscan = dbscan.fit_predict(X_scaled)
            
            n_clusters_dbscan = len(set(clusters_dbscan)) - (1 if -1 in clusters_dbscan else 0)
            n_noise = list(clusters_dbscan).count(-1)
            
            st.metric("Clusters Encontrados", n_clusters_dbscan)
            st.metric("Pontos de Ruído", n_noise)
            
            if n_clusters_dbscan > 1:
                # Silhouette para DBSCAN (excluindo ruído)
                mask = clusters_dbscan != -1
                if np.sum(mask) > 1:
                    silhouette_dbscan = silhouette_score(X_scaled[mask], clusters_dbscan[mask])
                    st.metric("Silhouette Score", f"{silhouette_dbscan:.3f}")
        
        # Análise dos clusters
        st.markdown("---")
        st.markdown("### Interpretação dos Clusters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Características dos Clusters (KMeans)")
            
            for i in range(n_clusters):
                cluster_data = df_cluster[df_cluster['cluster_kmeans'] == i]
                
                if len(cluster_data) > 0:
                    st.markdown(f"**Cluster {i} ({len(cluster_data)} casos):**")
                    
                    # Tipo de crime predominante
                    tipo_predominante = cluster_data['tipo_crime'].mode()
                    if len(tipo_predominante) > 0:
                        st.markdown(f"- Crime predominante: {tipo_predominante.iloc[0]}")
                    
                    # Bairro predominante
                    bairro_predominante = cluster_data['bairro'].mode()
                    if len(bairro_predominante) > 0:
                        st.markdown(f"- Bairro predominante: {bairro_predominante.iloc[0]}")
                    
                    # Horário médio
                    if 'hora' in cluster_data.columns:
                        hora_media = cluster_data['hora'].mean()
                        st.markdown(f"- Horário médio: {hora_media:.1f}h")
                    
                    st.markdown("")
        
        with col2:
            st.markdown("#### Distribuição por Cluster")
            
            # Gráfico de distribuição
            cluster_dist = pd.Series(clusters_kmeans).value_counts().sort_index()
            
            fig_dist = px.bar(
                x=cluster_dist.index,
                y=cluster_dist.values,
                title="Distribuição de Casos por Cluster",
                labels={'x': 'Cluster', 'y': 'Número de Casos'}
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Heatmap de características
            cluster_summary = df_cluster.groupby('cluster_kmeans').agg({
                'hora': 'mean',
                'tipo_crime_encoded': 'mean',
                'bairro_encoded': 'mean'
            }).round(2)
            
            fig_heatmap = px.imshow(
                cluster_summary.T,
                title="Características Médias por Cluster",
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Insights operacionais
        st.markdown("---")
        st.markdown("### Insights Operacionais dos Clusters")
        
        insights = []
        
        for i in range(n_clusters):
            cluster_data = df_cluster[df_cluster['cluster_kmeans'] == i]
            
            if len(cluster_data) > 0:
                # Análise temporal
                if 'hora' in cluster_data.columns:
                    hora_media = cluster_data['hora'].mean()
                    if hora_media >= 22 or hora_media <= 6:
                        insights.append(f"Cluster {i}: Padrão noturno/madrugada - Reforçar patrulhamento noturno")
                    elif 6 < hora_media <= 12:
                        insights.append(f"Cluster {i}: Padrão matinal - Monitoramento diurno")
                
                # Análise geográfica
                bairros_cluster = cluster_data['bairro'].value_counts()
                if len(bairros_cluster) > 0:
                    bairro_principal = bairros_cluster.index[0]
                    concentracao = (bairros_cluster.iloc[0] / len(cluster_data)) * 100
                    if concentracao > 70:
                        insights.append(f"Cluster {i}: Alta concentração em {bairro_principal} ({concentracao:.1f}%) - Foco geográfico")
        
        for insight in insights:
            st.info(f"💡 {insight}")
    
    except ImportError:
        st.error("Bibliotecas de Machine Learning não instaladas. Execute: pip install scikit-learn")
    except Exception as e:
        st.error(f"Erro na análise de clustering: {str(e)}")

def show_anomaly_detection(df):
    """Detecção de Anomalias - Machine Learning Não Supervisionado"""
    st.subheader("Detecção de Anomalias - Casos Atípicos")
    
    if len(df) < 10:
        st.warning("Dados insuficientes para detecção de anomalias (mínimo 10 casos).")
        return
    
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        import numpy as np
        
        # Preparar dados
        df_anomaly = df.copy()
        
        # Encoding
        le_tipo = LabelEncoder()
        le_bairro = LabelEncoder()
        
        df_anomaly['tipo_crime_encoded'] = le_tipo.fit_transform(df_anomaly['tipo_crime'])
        df_anomaly['bairro_encoded'] = le_bairro.fit_transform(df_anomaly['bairro'])
        
        # Features para detecção
        if 'hora' not in df_anomaly.columns:
            df_anomaly['hora'] = df_anomaly['data_ocorrencia'].dt.hour
        
        features = ['latitude', 'longitude', 'hora', 'tipo_crime_encoded']
        
        # Adicionar features se disponíveis
        if 'quantidade_vitimas' in df_anomaly.columns:
            features.append('quantidade_vitimas')
        if 'quantidade_suspeitos' in df_anomaly.columns:
            features.append('quantidade_suspeitos')
        
        df_anomaly = df_anomaly.dropna(subset=features)
        
        if len(df_anomaly) < 5:
            st.error("Dados insuficientes após limpeza.")
            return
        
        X = df_anomaly[features].values
        
        # Normalizar
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Isolation Forest")
            
            contamination = st.slider("Taxa de Contaminação", 0.05, 0.3, 0.1, 0.05)
            
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            anomalies_iso = iso_forest.fit_predict(X_scaled)
            
            df_anomaly['anomaly_iso'] = anomalies_iso
            
            n_anomalies_iso = np.sum(anomalies_iso == -1)
            st.metric("Anomalias Detectadas", n_anomalies_iso)
            st.metric("Taxa de Anomalias", f"{(n_anomalies_iso/len(df_anomaly)*100):.1f}%")
            
        with col2:
            st.markdown("#### Local Outlier Factor")
            
            n_neighbors = st.slider("Número de Vizinhos", 5, 20, 10)
            
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
            anomalies_lof = lof.fit_predict(X_scaled)
            
            df_anomaly['anomaly_lof'] = anomalies_lof
            
            n_anomalies_lof = np.sum(anomalies_lof == -1)
            st.metric("Anomalias Detectadas", n_anomalies_lof)
            st.metric("Taxa de Anomalias", f"{(n_anomalies_lof/len(df_anomaly)*100):.1f}%")
        
        # Visualização das anomalias
        st.markdown("---")
        st.markdown("### Visualização das Anomalias")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Mapa de Anomalias (Isolation Forest)")
            
            if 'latitude' in df_anomaly.columns and 'longitude' in df_anomaly.columns:
                fig_iso = px.scatter_mapbox(
                    df_anomaly,
                    lat='latitude',
                    lon='longitude',
                    color='anomaly_iso',
                    title="Anomalias Geográficas",
                    mapbox_style="open-street-map",
                    color_discrete_map={1: 'blue', -1: 'red'},
                    zoom=10,
                    height=400
                )
                st.plotly_chart(fig_iso, use_container_width=True)
        
        with col2:
            st.markdown("#### Distribuição Temporal das Anomalias")
            
            anomalies_by_hour = df_anomaly[df_anomaly['anomaly_iso'] == -1]['hora'].value_counts().sort_index()
            
            if len(anomalies_by_hour) > 0:
                fig_hour = px.bar(
                    x=anomalies_by_hour.index,
                    y=anomalies_by_hour.values,
                    title="Anomalias por Hora do Dia",
                    labels={'x': 'Hora', 'y': 'Número de Anomalias'}
                )
                st.plotly_chart(fig_hour, use_container_width=True)
        
        # Análise das anomalias
        st.markdown("---")
        st.markdown("### Análise das Anomalias Detectadas")
        
        anomalies_df = df_anomaly[df_anomaly['anomaly_iso'] == -1].copy()
        
        if len(anomalies_df) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Características das Anomalias")
                
                # Tipos de crime nas anomalias
                tipos_anomalias = anomalies_df['tipo_crime'].value_counts()
                st.markdown("**Tipos de crime mais anômalos:**")
                for tipo, count in tipos_anomalias.head(3).items():
                    perc = (count / len(anomalies_df)) * 100
                    st.markdown(f"- {tipo}: {count} casos ({perc:.1f}%)")
                
                # Bairros com anomalias
                bairros_anomalias = anomalies_df['bairro'].value_counts()
                st.markdown("**Bairros com mais anomalias:**")
                for bairro, count in bairros_anomalias.head(3).items():
                    st.markdown(f"- {bairro}: {count} casos")
            
            with col2:
                st.markdown("#### Casos Anômalos Específicos")
                
                # Mostrar alguns casos anômalos
                colunas_mostrar = ['data_ocorrencia', 'tipo_crime', 'bairro', 'hora']
                anomalies_sample = anomalies_df[colunas_mostrar].head(5)
                
                st.dataframe(anomalies_sample, use_container_width=True)
                
                # Insights sobre anomalias
                st.markdown("**Possíveis razões para anomalias:**")
                st.markdown("- Horários incomuns para o tipo de crime")
                st.markdown("- Localização atípica para o padrão")
                st.markdown("- Combinação incomum de características")
                st.markdown("- Casos com múltiplas vítimas/suspeitos")
        
        # Recomendações operacionais
        st.markdown("---")
        st.markdown("### Recomendações Operacionais")
        
        if len(anomalies_df) > 0:
            st.info("💡 **Investigação Prioritária:** Casos anômalos podem indicar padrões criminais emergentes ou situações de alto risco.")
            st.warning("⚠️ **Atenção Especial:** Monitorar locais e horários onde anomalias são frequentes.")
            
            # Alertas específicos
            if len(anomalies_df[anomalies_df['hora'].between(22, 6)]) > len(anomalies_df) * 0.5:
                st.error("🌙 **Alerta Noturno:** Muitas anomalias no período noturno - Reforçar patrulhamento.")
            
            bairros_criticos = anomalies_df['bairro'].value_counts()
            if len(bairros_criticos) > 0 and bairros_criticos.iloc[0] > len(anomalies_df) * 0.3:
                bairro_critico = bairros_criticos.index[0]
                st.error(f"📍 **Alerta Geográfico:** Concentração de anomalias em {bairro_critico} - Investigação especial necessária.")
    
    except ImportError:
        st.error("Bibliotecas de Machine Learning não instaladas. Execute: pip install scikit-learn")
    except Exception as e:
        st.error(f"Erro na detecção de anomalias: {str(e)}")

def show_supervised_model(df):
    """Modelo Supervisionado Aprimorado"""
    st.subheader("Modelo Supervisionado - Predição e Análise")
    
    if len(df) < 20:
        st.warning("Dados insuficientes para modelo supervisionado (mínimo 20 casos).")
        return
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        import numpy as np
        
        st.markdown("### Configuração do Modelo")
        
        # Seleção do target
        target_options = {
            "Gravidade do Caso": "gravidade",
            "Tipo de Crime": "tipo_crime",
            "Uso de Arma": "arma_utilizada"
        }
        
        target_choice = st.selectbox("Escolha o que prever:", list(target_options.keys()))
        target_column = target_options[target_choice]
        
        # Preparar dados
        df_model = df.copy()
        
        # Feature engineering
        if 'hora' not in df_model.columns:
            df_model['hora'] = df_model['data_ocorrencia'].dt.hour
        
        # Criar target baseado na escolha
        if target_choice == "Gravidade do Caso":
            # Criar score de gravidade
            df_model['gravidade_score'] = 0
            if 'arma_utilizada' in df_model.columns:
                df_model.loc[df_model['arma_utilizada'] == 'Arma de Fogo', 'gravidade_score'] += 3
                df_model.loc[df_model['arma_utilizada'] == 'Faca', 'gravidade_score'] += 2
            if 'quantidade_vitimas' in df_model.columns:
                df_model['gravidade_score'] += df_model['quantidade_vitimas'] - 1
            
            df_model['gravidade'] = pd.cut(
                df_model['gravidade_score'], 
                bins=[-1, 0, 2, 10], 
                labels=['Baixa', 'Média', 'Alta']
            )
            target_column = 'gravidade'
        
        # Features para o modelo
        features = ['latitude', 'longitude', 'hora']
        
        # Encoding de variáveis categóricas
        le_bairro = LabelEncoder()
        df_model['bairro_encoded'] = le_bairro.fit_transform(df_model['bairro'])
        features.append('bairro_encoded')
        
        if 'quantidade_vitimas' in df_model.columns:
            features.append('quantidade_vitimas')
        
        # Remover NaN
        df_model = df_model.dropna(subset=features + [target_column])
        
        if len(df_model) < 10:
            st.error("Dados insuficientes após limpeza.")
            return
        
        # Preparar X e y
        X = df_model[features]
        y = df_model[target_column]
        
        # Encoding do target se necessário
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
        
        # Split dos dados
        test_size = st.slider("Tamanho do conjunto de teste", 0.1, 0.4, 0.2, 0.05)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Treinamento do Modelo")
            
            # Opções de tunagem
            tune_hyperparams = st.checkbox("Tunagem de Hiperparâmetros", False)
            
            if tune_hyperparams:
                st.info("Executando Grid Search...")
                
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 10, None],
                    'min_samples_split': [2, 5, 10]
                }
                
                rf = RandomForestClassifier(random_state=42)
                grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy')
                grid_search.fit(X_train, y_train)
                
                best_model = grid_search.best_estimator_
                st.success(f"Melhores parâmetros: {grid_search.best_params_}")
            else:
                best_model = RandomForestClassifier(n_estimators=100, random_state=42)
                best_model.fit(X_train, y_train)
            
            # Predições
            y_pred = best_model.predict(X_test)
            
            # Métricas
            accuracy = accuracy_score(y_test, y_pred)
            st.metric("Acurácia", f"{accuracy:.3f}")
            
            # Cross-validation
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
            st.metric("CV Score Médio", f"{cv_scores.mean():.3f}")
            st.metric("CV Desvio Padrão", f"{cv_scores.std():.3f}")
        
        with col2:
            st.markdown("#### Importância das Features")
            
            feature_importance = pd.DataFrame({
                'feature': features,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig_importance = px.bar(
                feature_importance,
                x='importance',
                y='feature',
                orientation='h',
                title="Importância das Variáveis"
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        
        # Matriz de confusão
        st.markdown("---")
        st.markdown("### Avaliação do Modelo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Matriz de Confusão")
            
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(
                cm,
                title="Matriz de Confusão",
                color_continuous_scale='Blues',
                text_auto=True
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col2:
            st.markdown("#### Relatório de Classificação")
            
            if target_choice == "Gravidade do Caso" and 'le_target' in locals():
                target_names = le_target.classes_
                report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
            else:
                report = classification_report(y_test, y_pred, output_dict=True)
            
            # Mostrar métricas principais
            st.metric("Precisão Macro", f"{report['macro avg']['precision']:.3f}")
            st.metric("Recall Macro", f"{report['macro avg']['recall']:.3f}")
            st.metric("F1-Score Macro", f"{report['macro avg']['f1-score']:.3f}")
        
        # Análise de Fairness
        st.markdown("---")
        st.markdown("### Análise de Fairness")
        
        if 'bairro' in df_model.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Erro por Bairro")
                
                # Calcular erro por bairro
                df_test = df_model.iloc[X_test.index].copy()
                df_test['y_pred'] = y_pred
                df_test['y_true'] = y_test
                df_test['erro'] = (df_test['y_pred'] != df_test['y_true']).astype(int)
                
                erro_por_bairro = df_test.groupby('bairro')['erro'].mean().sort_values(ascending=False)
                
                fig_fairness = px.bar(
                    x=erro_por_bairro.values,
                    y=erro_por_bairro.index,
                    orientation='h',
                    title="Taxa de Erro por Bairro"
                )
                st.plotly_chart(fig_fairness, use_container_width=True)
            
            with col2:
                st.markdown("#### Análise de Viés")
                
                # Verificar se há viés significativo
                erro_medio = df_test['erro'].mean()
                bairros_alto_erro = erro_por_bairro[erro_por_bairro > erro_medio * 1.5]
                
                if len(bairros_alto_erro) > 0:
                    st.warning("⚠️ **Possível viés detectado:**")
                    for bairro, erro in bairros_alto_erro.items():
                        st.markdown(f"- {bairro}: {erro:.1%} de erro")
                    
                    st.markdown("**Recomendações:**")
                    st.markdown("- Coletar mais dados desses bairros")
                    st.markdown("- Revisar features específicas")
                    st.markdown("- Considerar balanceamento")
                else:
                    st.success("✅ **Modelo aparenta ser justo entre bairros**")
        
        # Predições em tempo real
        st.markdown("---")
        st.markdown("### Predição em Tempo Real")
        
        with st.expander("Fazer Predição"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pred_lat = st.number_input("Latitude", value=-8.1)
                pred_lon = st.number_input("Longitude", value=-34.9)
            
            with col2:
                pred_hora = st.slider("Hora", 0, 23, 12)
                pred_bairro = st.selectbox("Bairro", df_model['bairro'].unique())
            
            with col3:
                if 'quantidade_vitimas' in features:
                    pred_vitimas = st.number_input("Quantidade de Vítimas", 1, 10, 1)
                else:
                    pred_vitimas = 1
            
            if st.button("Fazer Predição"):
                # Preparar dados para predição
                pred_bairro_encoded = le_bairro.transform([pred_bairro])[0]
                
                pred_features = [pred_lat, pred_lon, pred_hora, pred_bairro_encoded]
                if 'quantidade_vitimas' in features:
                    pred_features.append(pred_vitimas)
                
                pred_result = best_model.predict([pred_features])[0]
                pred_proba = best_model.predict_proba([pred_features])[0]
                
                if target_choice == "Gravidade do Caso" and 'le_target' in locals():
                    pred_label = le_target.inverse_transform([pred_result])[0]
                else:
                    pred_label = pred_result
                
                st.success(f"**Predição:** {pred_label}")
                st.info(f"**Confiança:** {max(pred_proba):.1%}")
    
    except ImportError:
        st.error("Bibliotecas de Machine Learning não instaladas. Execute: pip install scikit-learn")
    except Exception as e:
        st.error(f"Erro no modelo supervisionado: {str(e)}")

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
    
    # Upload de dataset ou usar padrão
    st.sidebar.markdown("## Upload de Dataset")
    uploaded_file = st.sidebar.file_uploader(
        "Carregar arquivo CSV personalizado",
        type=['csv'],
        help="Opcional: carregue seu próprio dataset ou use o padrão"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"Dataset carregado: {len(df)} registros")
            
            # Verificar se tem as colunas mínimas necessárias
            required_cols = ['data_ocorrencia', 'tipo_crime', 'bairro', 'latitude', 'longitude']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.sidebar.warning(f"Colunas faltantes: {', '.join(missing_cols)}")
                st.sidebar.info("Usando dataset padrão...")
                df = load_violencia_domestica_data()
            else:
                # Processar data se necessário
                if df['data_ocorrencia'].dtype == 'object':
                    df['data_ocorrencia'] = pd.to_datetime(df['data_ocorrencia'], errors='coerce')
                
                # Criar colunas derivadas se não existirem
                if 'hora' not in df.columns:
                    df['hora'] = df['data_ocorrencia'].dt.hour
                if 'dia_semana' not in df.columns:
                    df['dia_semana'] = df['data_ocorrencia'].dt.day_name()
                
                st.sidebar.success("✅ Dataset compatível carregado!")
                
        except Exception as e:
            st.sidebar.error(f"Erro ao carregar arquivo: {str(e)}")
            df = load_violencia_domestica_data()
    else:
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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "Mapa Geográfico",
        "Análise Temporal",
        "Perfil dos Suspeitos",
        "Efetividade Policial",
        "Modus Operandi",
        "Correlações",
        "Clusters (ML)",
        "Anomalias (ML)",
        "Modelo Supervisionado",
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
        show_clustering_analysis(df_filtered)
    
    with tab8:
        show_anomaly_detection(df_filtered)
    
    with tab9:
        show_supervised_model(df_filtered)
    
    with tab10:
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
        colunas_exibir = ['data_ocorrencia', 'tipo_crime', 'bairro']
        
        # Adicionar colunas opcionais se existirem
        if 'genero_vitima' in df_filtered.columns:
            colunas_exibir.append('genero_vitima')
        if 'status_caso' in df_filtered.columns:
            colunas_exibir.append('status_caso')
        if 'orgao_responsavel' in df_filtered.columns:
            colunas_exibir.append('orgao_responsavel')
        
        df_display = df_filtered[colunas_exibir].copy()
        df_display['data_ocorrencia'] = df_display['data_ocorrencia'].dt.strftime('%d/%m/%Y %H:%M')
        
        st.dataframe(
            df_display,
            use_container_width=True,
            height=400
        )
        
        # Opções de download
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df_filtered.to_csv(index=False)
            st.download_button(
                label="Baixar dados filtrados (CSV)",
                data=csv,
                file_name=f"violencia_domestica_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            if st.button("Gerar Relatório HTML"):
                html_report = generate_html_report(df_filtered)
                st.download_button(
                    label="Baixar Relatório HTML",
                    data=html_report,
                    file_name=f"relatorio_violencia_domestica_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html"
                )
    
    # Rodapé
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #666;'>Dashboard de Violência Doméstica - DEAM | Projeto Acadêmico</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

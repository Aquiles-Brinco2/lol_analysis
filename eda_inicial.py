# 1. Importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo de visualizaciones
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 2. Cargar todos los datasets
def load_all_data(data_path="archive"):
    """Cargar todos los archivos CSV desde la carpeta"""
    data_path = Path(data_path)
    
    print("📂 Cargando datasets...")
    
    # Lista todos los archivos CSV
    csv_files = list(data_path.glob("*.csv"))
    print(f"Encontrados {len(csv_files)} archivos CSV:")
    for f in csv_files:
        print(f"  - {f.name}")
    
    # Cargar cada archivo
    data = {}
    
    # Cargar archivos con nombres específicos
    file_mapping = {
        'champion': ['ChampionTbl', 'champion'],
        'item': ['ItemTbl', 'item'],
        'match_stats': ['MatchStatsTbl', 'matchstats'],
        'match': ['MatchTbl', 'match'],
        'rank': ['RankTbl', 'rank'],
        'summoner_match': ['SummonerMatchTbl', 'summoner'],
        'team_match': ['TeamMatchStatsTbl', 'teammatch']
    }
    
    for csv_file in csv_files:
        filename = csv_file.stem.lower()  # Nombre sin extensión en minúsculas
        
        # Buscar coincidencias
        loaded = False
        for key, patterns in file_mapping.items():
            for pattern in patterns:
                if pattern in filename:
                    try:
                        print(f"  Cargando {csv_file.name} como '{key}'...")
                        data[key] = pd.read_csv(csv_file)
                        print(f"    ✓ Filas: {data[key].shape[0]}, Columnas: {data[key].shape[1]}")
                        loaded = True
                        break
                    except Exception as e:
                        print(f"    ✗ Error cargando {csv_file.name}: {e}")
                if loaded:
                    break
            if loaded:
                break
        
        if not loaded:
            print(f"  ⚠️ Archivo no reconocido: {csv_file.name}")
    
    return data

# 3. Cargar los datos
print("=" * 60)
print("ANÁLISIS EXPLORATORIO - LEAGUE OF LEGENDS DATASET")
print("=" * 60)

data = load_all_data("archive")

# Verificar qué datasets se cargaron
print("\n📊 Datasets cargados:")
for key, df in data.items():
    print(f"  {key}: {df.shape[0]} filas × {df.shape[1]} columnas")

# 4. Análisis básico de cada dataset
def basic_dataset_analysis(data_dict):
    """Realizar análisis básico de cada dataset"""
    
    results = {}
    
    for name, df in data_dict.items():
        print(f"\n{'='*40}")
        print(f"📈 ANÁLISIS: {name.upper()}")
        print(f"{'='*40}")
        
        # Info básica
        print(f"Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
        
        # Primeras filas
        print("\n🔍 Primeras 3 filas:")
        print(df.head(3))
        
        # Columnas y tipos
        print("\n📋 Columnas y tipos de datos:")
        print(df.dtypes.to_string())
        
        # Valores nulos
        null_counts = df.isnull().sum()
        null_percent = (null_counts / len(df) * 100).round(2)
        null_info = pd.DataFrame({
            'null_count': null_counts,
            'null_percent': null_percent
        })
        
        print("\n❓ Valores nulos por columna:")
        if null_counts.sum() > 0:
            print(null_info[null_info['null_count'] > 0].sort_values('null_count', ascending=False))
        else:
            print("¡No hay valores nulos!")
        
        # Valores únicos para columnas categóricas
        print("\n🎯 Valores únicos en columnas categóricas:")
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            unique_vals = df[col].nunique()
            if unique_vals <= 20:  # Mostrar solo si no son demasiados
                print(f"  {col}: {unique_vals} valores -> {df[col].unique()[:10]}")
            else:
                print(f"  {col}: {unique_vals} valores (mostrando primeros 5) -> {df[col].unique()[:5]}")
        
        # Estadísticas para columnas numéricas
        print("\n📊 Estadísticas descriptivas (columnas numéricas):")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(df[numeric_cols].describe().round(2))
        
        results[name] = {
            'shape': df.shape,
            'null_info': null_info,
            'categorical_cols': list(categorical_cols),
            'numeric_cols': list(numeric_cols)
        }
        
        # Guardar memoria eliminando duplicados si los hay
        dupes = df.duplicated().sum()
        if dupes > 0:
            print(f"\n⚠️  Advertencia: {dupes} filas duplicadas encontradas")
    
    return results

# 5. Ejecutar análisis básico
print("\n" + "="*60)
print("ANÁLISIS BÁSICO DE CADA DATASET")
print("="*60)

dataset_info = basic_dataset_analysis(data)

# 6. Análisis específico para predicción de victoria
print("\n" + "="*60)
print("🎯 ANÁLISIS PARA PREDICCIÓN DE VICTORIA")
print("="*60)

# Verificar si tenemos los datos clave
required_keys = ['team_match', 'match_stats', 'match']
missing_keys = [key for key in required_keys if key not in data]

if missing_keys:
    print(f"⚠️  Faltan datasets importantes: {missing_keys}")
else:
    # Análisis de TeamMatchStatsTbl
    print("\n1. DISTRIBUCIÓN DE VICTORIAS (TeamMatchStatsTbl):")
    team_df = data['team_match']
    
    # Verificar columnas disponibles
    print(f"Columnas disponibles: {list(team_df.columns)}")
    
    # Buscar columnas relacionadas con victoria
    win_cols = [col for col in team_df.columns if 'win' in col.lower() or 'result' in col.lower()]
    print(f"Columnas de victoria encontradas: {win_cols}")
    
    if win_cols:
        for win_col in win_cols:
            print(f"\nDistribución de '{win_col}':")
            print(team_df[win_col].value_counts(normalize=True).round(3))
            
            # Gráfico de distribución
            plt.figure(figsize=(10, 6))
            if team_df[win_col].dtype == 'object':
                # Para valores categóricos como 'BlueWin', 'RedWin'
                value_counts = team_df[win_col].value_counts()
                bars = plt.bar(value_counts.index.astype(str), value_counts.values)
                plt.title(f'Distribución de {win_col}', fontsize=16, fontweight='bold')
                plt.xlabel(win_col, fontsize=12)
                plt.ylabel('Frecuencia', fontsize=12)
                
                # Añadir porcentajes en las barras
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{height:,}', ha='center', va='bottom', fontsize=10)
                
                plt.tight_layout()
                plt.savefig(f'distribucion_{win_col}.png', dpi=100, bbox_inches='tight')
                plt.show()
            else:
                # Para valores numéricos
                team_df[win_col].value_counts().plot(kind='bar')
                plt.title(f'Distribución de {win_col}', fontsize=16, fontweight='bold')
                plt.xlabel(win_col, fontsize=12)
                plt.ylabel('Frecuencia', fontsize=12)
                plt.tight_layout()
                plt.savefig(f'distribucion_{win_col}.png', dpi=100, bbox_inches='tight')
                plt.show()
    else:
        print("No se encontraron columnas de victoria explícitas")
        print("Posiblemente necesitemos crear la variable objetivo")
    
    # 7. Análisis de correlación entre estadísticas y victoria
    print("\n2. ESTADÍSTICAS POR EQUIPO:")
    
    # Columnas de equipo azul y rojo
    blue_cols = [col for col in team_df.columns if 'blue' in col.lower()]
    red_cols = [col for col in team_df.columns if 'red' in col.lower()]
    
    print(f"Columnas del equipo azul: {len(blue_cols)}")
    print(f"Columnas del equipo rojo: {len(red_cols)}")
    
    # Mostrar algunas estadísticas clave
    stat_cols = ['kills', 'tower', 'dragon', 'baron', 'rift']
    for stat in stat_cols:
        blue_stat_cols = [col for col in blue_cols if stat in col.lower()]
        red_stat_cols = [col for col in red_cols if stat in col.lower()]
        
        if blue_stat_cols and red_stat_cols:
            print(f"\n{stat.upper()} - Equipo Azul vs Rojo:")
            for b_col, r_col in zip(blue_stat_cols[:3], red_stat_cols[:3]):
                print(f"  {b_col}: {team_df[b_col].mean():.2f} | {r_col}: {team_df[r_col].mean():.2f}")
    
    # 8. Análisis de MatchStatsTbl
    print("\n3. ESTADÍSTICAS DE JUGADORES (MatchStatsTbl):")
    if 'match_stats' in data:
        stats_df = data['match_stats']
        print(f"Total de estadísticas de jugadores: {stats_df.shape[0]:,}")
        
        # Verificar columnas de victoria
        stats_win_cols = [col for col in stats_df.columns if 'win' in col.lower()]
        if stats_win_cols:
            print(f"\nDistribución de victorias por jugador:")
            for win_col in stats_win_cols:
                print(stats_df[win_col].value_counts(normalize=True))
        
        # Estadísticas clave
        key_stats = ['kills', 'deaths', 'assists', 'TotalGold', 'MinionsKilled', 'visionScore']
        available_stats = [col for col in key_stats if col in stats_df.columns]
        
        if available_stats:
            print(f"\nEstadísticas promedio por jugador:")
            for stat in available_stats:
                print(f"  {stat}: {stats_df[stat].mean():.2f} (std: {stats_df[stat].std():.2f})")
    
    # 9. Análisis de MatchTbl
    print("\n4. METADATOS DE PARTIDAS (MatchTbl):")
    if 'match' in data:
        match_df = data['match']
        
        # Duración de partidas
        if 'GameDuration' in match_df.columns:
            print(f"Duración promedio: {match_df['GameDuration'].mean():.2f} segundos")
            print(f"  ({match_df['GameDuration'].mean()/60:.2f} minutos)")
            
            # Distribución de duraciones
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            match_df['GameDuration'].hist(bins=50, edgecolor='black')
            plt.title('Distribución de Duración de Partidas', fontsize=14, fontweight='bold')
            plt.xlabel('Duración (segundos)', fontsize=12)
            plt.ylabel('Frecuencia', fontsize=12)
            
            plt.subplot(1, 2, 2)
            match_df['GameDuration'].plot(kind='box')
            plt.title('Boxplot de Duración', fontsize=14, fontweight='bold')
            plt.ylabel('Segundos', fontsize=12)
            
            plt.tight_layout()
            plt.savefig('duracion_partidas.png', dpi=100, bbox_inches='tight')
            plt.show()
        
        # Tipos de cola
        if 'QueueType' in match_df.columns:
            print(f"\nTipos de cola:")
            queue_counts = match_df['QueueType'].value_counts()
            print(queue_counts)
            
            plt.figure(figsize=(10, 6))
            queue_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90)
            plt.title('Distribución de Tipos de Cola', fontsize=16, fontweight='bold')
            plt.ylabel('')
            plt.tight_layout()
            plt.savefig('tipos_cola.png', dpi=100, bbox_inches='tight')
            plt.show()
    
    # 10. Análisis de rangos
    print("\n5. DISTRIBUCIÓN DE RANGOS:")
    if 'rank' in data:
        rank_df = data['rank']
        print(rank_df)
    
    if 'match' in data and 'RankFk' in data['match'].columns:
        rank_dist = data['match']['RankFk'].value_counts().sort_index()
        print(f"\nDistribución de partidas por rango:")
        print(rank_dist)
        
        # Mapear códigos de rango a nombres si tenemos la tabla
        if 'rank' in data:
            rank_names = dict(zip(data['rank']['RankId'], data['rank']['RankName']))
            for rank_id, count in rank_dist.items():
                rank_name = rank_names.get(rank_id, f'Rango {rank_id}')
                print(f"  {rank_name}: {count} partidas ({count/len(data['match'])*100:.1f}%)")

# 11. Resumen del dataset completo
print("\n" + "="*60)
print("📋 RESUMEN GENERAL DEL DATASET")
print("="*60)

total_rows = sum([df.shape[0] for df in data.values()])
total_columns = sum([df.shape[1] for df in data.values()])

print(f"Total de filas en todos los datasets: {total_rows:,}")
print(f"Total de columnas en todos los datasets: {total_columns}")
print(f"Número de datasets: {len(data)}")

# Identificar columnas clave para la unión
print("\n🔗 Columnas clave para unir datasets:")
for name, df in data.items():
    id_cols = [col for col in df.columns if 'id' in col.lower() or 'fk' in col.lower()]
    if id_cols:
        print(f"  {name}: {id_cols}")

# 12. Recomendaciones iniciales para el modelo
print("\n" + "="*60)
print("💡 RECOMENDACIONES INICIALES PARA EL MODELO")
print("="*60)

if 'team_match' in data:
    team_df = data['team_match']
    
    # Verificar balance de clases
    win_cols = [col for col in team_df.columns if 'win' in col.lower()]
    if win_cols:
        for win_col in win_cols:
            if team_df[win_col].dtype in ['int64', 'float64']:
                class_dist = team_df[win_col].value_counts(normalize=True)
                print(f"Distribución de clases para '{win_col}':")
                print(f"  Clase 0: {class_dist.get(0, 0)*100:.1f}%")
                print(f"  Clase 1: {class_dist.get(1, 0)*100:.1f}%")
                
                if abs(class_dist.get(0, 0) - class_dist.get(1, 0)) > 0.2:
                    print("  ⚠️  Posible desbalance de clases - considerar técnicas de balanceo")
                else:
                    print("  ✓ Clases relativamente balanceadas")
    
    # Identificar características potenciales
    print("\nCaracterísticas potenciales para el modelo:")
    
    # Características de composición
    champ_cols = [col for col in team_df.columns if 'champ' in col.lower()]
    if champ_cols:
        print(f"  • Composición de campeones ({len(champ_cols)} características)")
    
    # Características de objetivos
    objective_cols = [col for col in team_df.columns if any(obj in col.lower() for obj in ['kill', 'tower', 'dragon', 'baron', 'rift'])]
    if objective_cols:
        print(f"  • Objetivos del juego ({len(objective_cols)} características)")

print("\n" + "="*60)
print("🎮 LISTO PARA EL SIGUIENTE PASO: UNIR DATASETS")
print("="*60)
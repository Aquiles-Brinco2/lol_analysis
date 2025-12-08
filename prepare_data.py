# prepare_data.py
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("PREPARACIÓN DE DATOS PARA PREDICCIÓN DE VICTORIA")
print("=" * 60)

# 1. Cargar datos correctamente
def load_data_fixed(data_path="archive"):
    """Cargar datos con nombres correctos"""
    data_path = Path(data_path)
    
    # Mapeo exacto de nombres de archivo
    files = {
        'champions': 'ChampionTbl.csv',
        'items': 'ItemTbl.csv',
        'match_stats': 'MatchStatsTbl.csv',
        'matches': 'MatchTbl.csv',  # El original pequeño
        'ranks': 'RankTbl.csv',
        'summoner_matches': 'SummonerMatchTbl.csv',
        'team_matches': 'TeamMatchTbl.csv'  # El principal
    }
    
    data = {}
    for key, filename in files.items():
        filepath = data_path / filename
        if filepath.exists():
            print(f"📂 Cargando {filename}...")
            data[key] = pd.read_csv(filepath)
            print(f"   ✓ {data[key].shape[0]} filas, {data[key].shape[1]} columnas")
        else:
            print(f"⚠️  No encontrado: {filename}")
    
    return data

# Cargar datos
print("\n1. CARGANDO DATOS...")
data = load_data_fixed()

# Verificar que tenemos los datos clave
if 'team_matches' not in data:
    print("❌ ERROR: No se encontró TeamMatchTbl.csv")
    exit()

print("\n✅ Datos cargados correctamente")

# 2. Análisis inicial del dataset principal
print("\n" + "="*60)
print("2. ANÁLISIS DEL DATASET PRINCIPAL (TeamMatchTbl)")
print("="*60)

team_df = data['team_matches']
print(f"Partidas totales: {len(team_df):,}")
print(f"\nDistribución de victorias:")
print(f"  BlueWin = 1: {team_df['BlueWin'].sum():,} partidas ({team_df['BlueWin'].mean()*100:.1f}%)")
print(f"  BlueWin = 0: {(team_df['BlueWin'] == 0).sum():,} partidas ({100 - team_df['BlueWin'].mean()*100:.1f}%)")

# 3. Crear características de composición de equipo
print("\n" + "="*60)
print("3. CREANDO CARACTERÍSTICAS DE COMPOSICIÓN")
print("="*60)

# Lista de columnas de campeones
blue_champ_cols = ['B1Champ', 'B2Champ', 'B3Champ', 'B4Champ', 'B5Champ']
red_champ_cols = ['R1Champ', 'R2Champ', 'R3Champ', 'R4Champ', 'R5Champ']

# Características básicas de composición
features_df = pd.DataFrame()
features_df['MatchFk'] = team_df['MatchFk']

# 3.1. Características simples de campeones
print("Calculando características básicas...")

# ID promedio de campeones (proxy de "antigüedad")
features_df['Blue_AvgChampId'] = team_df[blue_champ_cols].mean(axis=1)
features_df['Red_AvgChampId'] = team_df[red_champ_cols].mean(axis=1)

# Diversidad de campeones (cuántos son únicos)
features_df['Blue_UniqueChamps'] = team_df[blue_champ_cols].nunique(axis=1)
features_df['Red_UniqueChamps'] = team_df[red_champ_cols].nunique(axis=1)

# Diferencia en características
features_df['ChampId_Diff'] = features_df['Blue_AvgChampId'] - features_df['Red_AvgChampId']
features_df['UniqueChamps_Diff'] = features_df['Blue_UniqueChamps'] - features_df['Red_UniqueChamps']

print(f"✓ Características de composición creadas: {len([col for col in features_df.columns if 'Champ' in col])}")

# 3.2. Cargar datos de win rates de campeones (si existen)
# Como no los tenemos, calcularemos win rates básicos del dataset
print("\nCalculando win rates básicos de campeones...")

# Primero necesitamos una lista de todas las partidas con campeones y resultado
# Vamos a "derretir" (melt) los datos para tener una lista de campeones por partida
blue_melted = pd.melt(
    team_df[['MatchFk', 'BlueWin'] + blue_champ_cols], 
    id_vars=['MatchFk', 'BlueWin'],
    value_vars=blue_champ_cols,
    var_name='Position',
    value_name='ChampionId'
)
blue_melted['Team'] = 'Blue'

red_melted = pd.melt(
    team_df[['MatchFk', 'BlueWin'] + red_champ_cols], 
    id_vars=['MatchFk', 'BlueWin'],
    value_vars=red_champ_cols,
    var_name='Position',
    value_name='ChampionId'
)
red_melted['Team'] = 'Red'
# Para equipo rojo, BlueWin=0 significa que rojo gana
red_melted['TeamWin'] = (red_melted['BlueWin'] == 0).astype(int)
blue_melted['TeamWin'] = blue_melted['BlueWin']

# Combinar
all_champ_games = pd.concat([blue_melted, red_melted], ignore_index=True)

# Calcular win rate por campeón
champ_win_rates = all_champ_games.groupby('ChampionId')['TeamWin'].agg(['mean', 'count']).reset_index()
champ_win_rates.columns = ['ChampionId', 'ChampWinRate', 'ChampGameCount']

# Solo considerar campeones con suficientes partidas
min_games = 50  # Ajustar según necesidad
champ_win_rates_filtered = champ_win_rates[champ_win_rates['ChampGameCount'] >= min_games]
print(f"✓ Win rates calculados para {len(champ_win_rates_filtered)} campeones (con ≥{min_games} partidas)")

# 3.3. Añadir win rate promedio por equipo
def calculate_team_win_rate(row, champ_cols, champ_win_rates):
    """Calcular win rate promedio de los campeones de un equipo"""
    champ_ids = row[champ_cols].values
    win_rates = []
    for champ_id in champ_ids:
        wr = champ_win_rates[champ_win_rates['ChampionId'] == champ_id]['ChampWinRate']
        if len(wr) > 0:
            win_rates.append(wr.values[0])
    
    return np.mean(win_rates) if win_rates else 0.5  # 0.5 si no hay datos

# Aplicar a cada equipo
print("Calculando win rates por equipo...")
features_df['Blue_AvgChampWinRate'] = team_df.apply(
    lambda row: calculate_team_win_rate(row, blue_champ_cols, champ_win_rates_filtered),
    axis=1
)
features_df['Red_AvgChampWinRate'] = team_df.apply(
    lambda row: calculate_team_win_rate(row, red_champ_cols, champ_win_rates_filtered),
    axis=1
)
features_df['ChampWinRate_Diff'] = features_df['Blue_AvgChampWinRate'] - features_df['Red_AvgChampWinRate']

print(f"✓ Win rates por equipo calculados")

# 4. Características de estadísticas tempranas (objetivos)
print("\n" + "="*60)
print("4. CREANDO CARACTERÍSTICAS DE OBJETIVOS TEMPRANOS")
print("="*60)

# Usar datos de TeamMatchTbl para objetivos
# Ya tenemos kills, torretas, dragones, barones
# Podemos crear características de "primero en X"

# Para esto, necesitaríamos datos por minuto, pero podemos usar
# las estadísticas totales como proxy, o asumir que si un equipo
# tiene más de algo, probablemente lo consiguió primero

# Características basadas en TeamMatchTbl
features_df['Blue_Kills'] = team_df['BlueKills']
features_df['Red_Kills'] = team_df['RedKills']
features_df['Kill_Diff'] = team_df['BlueKills'] - team_df['RedKills']

features_df['Blue_Towers'] = team_df['BlueTowerKills']
features_df['Red_Towers'] = team_df['RedTowerKills']
features_df['Tower_Diff'] = team_df['BlueTowerKills'] - team_df['RedTowerKills']

features_df['Blue_Dragons'] = team_df['BlueDragonKills']
features_df['Red_Dragons'] = team_df['RedDragonKills']
features_df['Dragon_Diff'] = team_df['BlueDragonKills'] - team_df['RedDragonKills']

features_df['Blue_Barons'] = team_df['BlueBaronKills']
features_df['Red_Barons'] = team_df['RedBaronKills']
features_df['Baron_Diff'] = team_df['BlueBaronKills'] - team_df['RedBaronKills']

features_df['Blue_Heralds'] = team_df['BlueRiftHeraldKills']
features_df['Red_Heralds'] = team_df['RedRiftHeraldKills']
features_df['Herald_Diff'] = team_df['BlueRiftHeraldKills'] - team_df['RedRiftHeraldKills']

print(f"✓ {len([col for col in features_df.columns if 'Diff' in col])} características de diferencia creadas")

# 5. Características de contexto
print("\n" + "="*60)
print("5. CREANDO CARACTERÍSTICAS DE CONTEXTO")
print("="*60)

# Extraer región del MatchFk
features_df['Region'] = team_df['MatchFk'].str.extract(r'^([A-Z]+)')[0]

# Mapear regiones a códigos
region_mapping = {
    'EUW': 0, 'NA': 1, 'EUN': 2, 'KR': 3, 
    'BR': 4, 'TR': 5, 'RU': 6, 'OC': 7,
    'JP': 8, 'LA': 9, 'PH': 10, 'SG': 11,
    'TW': 12, 'TH': 13, 'VN': 14
}

# Crear variable dummy para región principal
features_df['Region_Code'] = features_df['Region'].map(region_mapping).fillna(-1)

# Si tenemos datos de MatchTbl original, añadir duración
if 'matches' in data and 'GameDuration' in data['matches'].columns:
    # Unir por MatchFk
    match_info = data['matches'][['MatchId', 'GameDuration']].copy()
    match_info = match_info.rename(columns={'MatchId': 'MatchFk'})
    features_df = pd.merge(features_df, match_info, on='MatchFk', how='left')
    print("✓ Duración de partida añadida")
else:
    print("ℹ️  No se encontró información de duración")

# 6. Preparar dataset final para ML
print("\n" + "="*60)
print("6. PREPARANDO DATASET FINAL PARA MACHINE LEARNING")
print("="*60)

# Añadir variable objetivo
features_df['BlueWin'] = team_df['BlueWin'].astype(int)

# Eliminar columnas no numéricas para ML (excepto MatchFk para referencia)
ml_columns = [col for col in features_df.columns 
              if col not in ['MatchFk', 'Region'] and features_df[col].dtype in [np.int64, np.float64]]

ml_df = features_df[ml_columns].copy()

print(f"Dataset final para ML:")
print(f"  - Partidas: {len(ml_df):,}")
print(f"  - Características: {len(ml_df.columns) - 1}")  # Restamos BlueWin
print(f"  - Variable objetivo: BlueWin")

print("\n📋 Columnas disponibles:")
for i, col in enumerate(ml_df.columns, 1):
    if col == 'BlueWin':
        print(f"  {i:2d}. {col} ← OBJETIVO")
    else:
        print(f"  {i:2d}. {col}")

# 7. Análisis de correlación con la victoria
print("\n" + "="*60)
print("7. ANÁLISIS DE CORRELACIÓN CON LA VICTORIA")
print("="*60)

# Calcular correlaciones
correlations = ml_df.corr()['BlueWin'].sort_values(ascending=False)

print("Correlación con BlueWin (de mayor a menor):")
print("-" * 50)
for feature, corr in correlations.items():
    if feature != 'BlueWin':
        print(f"{feature:30s}: {corr:+.4f}")

# Identificar características más importantes
top_features = correlations.abs().sort_values(ascending=False).index[1:11]  # Excluir BlueWin
print(f"\n🔝 Top 10 características más correlacionadas (absoluto):")
for i, feature in enumerate(top_features, 1):
    corr = correlations[feature]
    print(f"{i:2d}. {feature:30s}: {corr:+.4f}")

# 8. Guardar dataset procesado
print("\n" + "="*60)
print("8. GUARDANDO DATOS PROCESADOS")
print("="*60)

# Crear carpeta para datos procesados
output_dir = Path("processed_data")
output_dir.mkdir(exist_ok=True)

# Guardar dataset completo
output_path = output_dir / "lol_processed_data.csv"
ml_df.to_csv(output_path, index=False)
print(f"✅ Dataset guardado en: {output_path}")
print(f"   - Tamaño: {len(ml_df):,} filas × {len(ml_df.columns)} columnas")

# Guardar información de características
features_info = pd.DataFrame({
    'feature': ml_df.columns,
    'type': ml_df.dtypes.astype(str),
    'non_null': ml_df.notnull().sum(),
    'null_pct': (ml_df.isnull().sum() / len(ml_df) * 100).round(2),
    'mean': ml_df.mean().round(4),
    'std': ml_df.std().round(4)
})
features_info.to_csv(output_dir / "features_info.csv", index=False)
print(f"✅ Información de características guardada")

# 9. Resumen final
print("\n" + "="*60)
print("🎮 RESUMEN FINAL - LISTO PARA ENTRENAR MODELO")
print("="*60)

print(f"""
📊 DATOS PROCESADOS:
• Partidas totales: {len(ml_df):,}
• Características creadas: {len(ml_df.columns) - 1}
• Distribución de victorias:
  - Azul gana: {ml_df['BlueWin'].sum():,} ({ml_df['BlueWin'].mean()*100:.1f}%)
  - Rojo gana: {(ml_df['BlueWin'] == 0).sum():,} ({100 - ml_df['BlueWin'].mean()*100:.1f}%)

🎯 PRÓXIMOS PASOS:
1. Dividir en entrenamiento/prueba (80/20)
2. Balancear clases si es necesario
3. Entrenar modelos (Random Forest, XGBoost, etc.)
4. Evaluar con métricas: Accuracy, Precision, Recall, F1, AUC

💡 RECOMENDACIONES:
• Las características con mayor correlación son buenos candidatos
• Considerar técnicas para manejar el desbalance (57% vs 43%)
• Empezar con Random Forest para entender importancia de características
""")

# Mostrar primeros registros
print("\n🔍 Primeras 3 filas del dataset procesado:")
print(ml_df.head(3).to_string())
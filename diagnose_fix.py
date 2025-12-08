# diagnose_fix.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🔍 DIAGNÓSTICO Y CORRECCIÓN DE PROBLEMAS")
print("=" * 70)

# 1. Cargar datos y verificar
print("\n1. 📂 CARGANDO Y ANALIZANDO DATOS ORIGINALES...")
data_path = Path("processed_data/lol_processed_data.csv")

if not data_path.exists():
    print("❌ No se encontraron datos procesados")
    exit()

df = pd.read_csv(data_path)
print(f"✅ Datos cargados: {df.shape[0]:,} filas × {df.shape[1]} columnas")

# 2. Análisis detallado de los datos
print("\n2. 📊 ANÁLISIS DETALLADO DEL DATASET")
print("-" * 40)

print("📋 Primeras 5 filas:")
print(df.head())

print(f"\n🔢 Tipos de datos:")
print(df.dtypes.value_counts())

print(f"\n❓ Valores nulos:")
print(df.isnull().sum().sort_values(ascending=False).head(10))

print(f"\n🎯 Distribución de BlueWin:")
print(df['BlueWin'].value_counts(normalize=True))

# 3. Buscar problemas específicos
print("\n3. 🔍 BUSCANDO PROBLEMAS COMUNES")
print("-" * 40)

# 3.1. Verificar si hay columnas constantes
constant_cols = []
for col in df.columns:
    if df[col].nunique() == 1:
        constant_cols.append(col)

if constant_cols:
    print(f"⚠️  Columnas constantes encontradas: {constant_cols}")
else:
    print("✅ No hay columnas constantes")

# 3.2. Verificar correlaciones extremas
print(f"\n📈 Matriz de correlaciones (solo primeras 10x10):")
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix.iloc[:10, :10], annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Matriz de Correlación (primeras 10 características)')
plt.tight_layout()
plt.savefig('processed_data/correlation_heatmap.png', dpi=150)
plt.show()

# Buscar correlaciones perfectas (>0.95 o <-0.95)
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.95:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

if high_corr_pairs:
    print(f"⚠️  Correlaciones extremas encontradas:")
    for col1, col2, corr in high_corr_pairs[:5]:  # Mostrar solo 5
        print(f"  {col1} ↔ {col2}: {corr:.3f}")
else:
    print("✅ No hay correlaciones extremas")

# 3.3. Verificar si BlueWin está en los datos de alguna manera
print(f"\n🎯 Verificando variable objetivo BlueWin:")
print(f"Valores únicos: {df['BlueWin'].unique()}")
print(f"Distribución: {df['BlueWin'].value_counts()}")

# 4. Verificar si hay data leakage
print("\n4. 🚨 BUSCANDO DATA LEAKAGE")
print("-" * 40)

# Verificar si hay características que contienen información del futuro
leakage_suspects = []
for col in df.columns:
    if col != 'BlueWin':
        # Si la correlación es casi perfecta, sospechoso
        corr = abs(df[col].corr(df['BlueWin']))
        if corr > 0.9:
            leakage_suspects.append((col, corr))

if leakage_suspects:
    print("⚠️  POSIBLE DATA LEAKAGE DETECTADO!")
    print("Características altamente correlacionadas con BlueWin:")
    for col, corr in leakage_suspects:
        print(f"  {col}: correlación = {corr:.4f}")
        
    # Mostrar ejemplos
    suspect_col = leakage_suspects[0][0]
    print(f"\n🔍 Analizando {suspect_col}:")
    print(f"  Valores únicos: {df[suspect_col].nunique()}")
    print(f"  Min: {df[suspect_col].min()}, Max: {df[suspect_col].max()}")
    print(f"  Ejemplos cuando BlueWin=1: {df[df['BlueWin']==1][suspect_col].iloc[:5].values}")
    print(f"  Ejemplos cuando BlueWin=0: {df[df['BlueWin']==0][suspect_col].iloc[:5].values}")
else:
    print("✅ No se detectó data leakage obvio")

# 5. Crear dataset corregido
print("\n5. 🛠️  CREANDO DATASET CORREGIDO")
print("-" * 40)

# Primero, vamos a analizar las características que tenemos
print("📋 Lista completa de características:")
for i, col in enumerate(df.columns, 1):
    print(f"{i:3d}. {col}")

# Identificar características problemáticas
# Basado en los resultados, probablemente hay características que son
# derivadas directamente del resultado

# Vamos a crear un dataset más simple y seguro
print("\n🔧 Creando características seguras...")

# Separar el objetivo
y = df['BlueWin'].copy()

# Crear X con características que deberían ser seguras
# Eliminamos cualquier cosa que parezca un conteo o resultado
safe_features = []

for col in df.columns:
    if col == 'BlueWin':
        continue
    
    col_lower = col.lower()
    
    # Excluir características sospechosas
    exclude_keywords = ['diff', 'kills', 'tower', 'dragon', 'baron', 'herald', 'win']
    
    is_safe = True
    for keyword in exclude_keywords:
        if keyword in col_lower:
            is_safe = False
            print(f"  Excluyendo {col} (contiene '{keyword}')")
            break
    
    if is_safe:
        safe_features.append(col)

print(f"\n✅ Características seguras seleccionadas: {len(safe_features)}")
print(f"Características: {safe_features}")

X_safe = df[safe_features].copy()

# Verificar correlaciones nuevamente
print(f"\n📊 Correlaciones con BlueWin (características seguras):")
correlations = X_safe.apply(lambda x: x.corr(y))
print(correlations.sort_values(ascending=False).head(10))

# 6. Probar modelo simple para verificar
print("\n6. 🧪 PRUEBA RÁPIDA CON MODELO SIMPLE")
print("-" * 40)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X_safe, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Entrenamiento: {X_train.shape[0]:,} muestras")
print(f"Prueba: {X_test.shape[0]:,} muestras")

# Modelo simple
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n📈 Resultado con características seguras:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Precisión esperada si predijéramos siempre azul: {y.mean():.4f}")

if accuracy > y.mean() + 0.05:
    print("  ✅ El modelo aprende algo útil")
elif accuracy > y.mean():
    print("  ⚠️  El modelo es marginalmente mejor que predecir siempre azul")
else:
    print("  ❌ El modelo es peor que predecir siempre azul")

# 7. Crear dataset con mejores características
print("\n7. 🏗️  CREANDO DATASET MEJORADO")
print("-" * 40)

# Vamos a recrear características desde cero, evitando data leakage
print("🔨 Reconstruyendo características desde los datos originales...")

# Primero necesitamos cargar los datos originales
original_data_path = Path("archive")
if not original_data_path.exists():
    print("❌ No se encuentra la carpeta 'archives' con datos originales")
else:
    print("✅ Carpeta 'archives' encontrada")
    
    # Cargar TeamMatchTbl para características de composición
    team_df_path = original_data_path / "TeamMatchTbl.csv"
    if team_df_path.exists():
        team_df = pd.read_csv(team_df_path)
        print(f"✅ TeamMatchTbl cargado: {team_df.shape[0]:,} filas")
        
        # Crear características seguras de composición
        print("\n🎯 Creando características de composición seguras:")
        
        # 1. Diversidad de campeones por equipo
        blue_champs = ['B1Champ', 'B2Champ', 'B3Champ', 'B4Champ', 'B5Champ']
        red_champs = ['R1Champ', 'R2Champ', 'R3Champ', 'R4Champ', 'R5Champ']
        
        features_new = pd.DataFrame()
        features_new['MatchFk'] = team_df['MatchFk']
        
        # Diversidad (cuántos campeones únicos)
        features_new['Blue_UniqueChamps'] = team_df[blue_champs].nunique(axis=1)
        features_new['Red_UniqueChamps'] = team_df[red_champs].nunique(axis=1)
        
        # ID promedio de campeones (proxy de antigüedad)
        features_new['Blue_AvgChampId'] = team_df[blue_champs].mean(axis=1)
        features_new['Red_AvgChampId'] = team_df[red_champs].mean(axis=1)
        
        # Diferencia en diversidad
        features_new['UniqueChamps_Diff'] = features_new['Blue_UniqueChamps'] - features_new['Red_UniqueChamps']
        features_new['AvgChampId_Diff'] = features_new['Blue_AvgChampId'] - features_new['Red_AvgChampId']
        
        # 2. Información de región (extraer del MatchFk)
        features_new['Region'] = team_df['MatchFk'].str.extract(r'^([A-Z]+)')[0]
        # Codificar regiones principales
        regions_to_code = {'EUW': 0, 'NA': 1, 'EUN': 2, 'KR': 3, 'BR': 4}
        features_new['Region_Code'] = features_new['Region'].map(regions_to_code).fillna(5)
        
        # 3. Variable objetivo
        features_new['BlueWin'] = team_df['BlueWin']
        
        print(f"✅ Características creadas: {features_new.shape[1]} columnas")
        print(f"📋 Características: {list(features_new.columns)}")
        
        # Eliminar columnas no numéricas para ML
        features_ml = features_new.drop(['MatchFk', 'Region'], axis=1)
        
        # Verificar correlaciones
        print(f"\n📊 Correlaciones con BlueWin:")
        correlations_new = features_ml.corr()['BlueWin'].drop('BlueWin').sort_values(ascending=False)
        print(correlations_new.head(10))
        
        # Guardar nuevo dataset
        output_path = Path("processed_data/lol_safe_features.csv")
        features_ml.to_csv(output_path, index=False)
        print(f"\n💾 Dataset seguro guardado en: {output_path}")
        
        # Probar modelo rápido
        print("\n🧪 Probando modelo con características seguras...")
        
        X_new = features_ml.drop('BlueWin', axis=1)
        y_new = features_ml['BlueWin']
        
        X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
            X_new, y_new, test_size=0.2, random_state=42, stratify=y_new
        )
        
        model_new = LogisticRegression(max_iter=1000, random_state=42)
        model_new.fit(X_train_new, y_train_new)
        
        y_pred_new = model_new.predict(X_test_new)
        accuracy_new = accuracy_score(y_test_new, y_pred_new)
        
        print(f"📈 Resultados:")
        print(f"  Accuracy: {accuracy_new:.4f}")
        print(f"  Mejora sobre predecir siempre azul: {accuracy_new - y_new.mean():.4f}")
        print(f"  Precisión baseline (siempre azul): {y_new.mean():.4f}")
        
        # 8. Entrenar varios modelos con datos seguros
        print("\n8. 🤖 ENTRENANDO MODELOS CON DATOS SEGUROS")
        print("-" * 40)
        
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        from sklearn.metrics import classification_report
        
        models_to_test = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
        }
        
        print("📊 Resultados de modelos con características seguras:")
        print("-" * 60)
        
        results_safe = []
        for name, model in models_to_test.items():
            model.fit(X_train_new, y_train_new)
            y_pred = model.predict(X_test_new)
            accuracy = accuracy_score(y_test_new, y_pred)
            results_safe.append((name, accuracy))
            
            if name == 'Random Forest' and hasattr(model, 'feature_importances_'):
                # Mostrar importancia de características
                print(f"\n🔍 Importancia de características ({name}):")
                importances = pd.DataFrame({
                    'feature': X_new.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                for i, row in importances.head().iterrows():
                    print(f"  {row['feature']}: {row['importance']:.4f}")
        
        print("\n" + "-" * 60)
        print("🏆 COMPARACIÓN DE MODELOS (características seguras):")
        for name, accuracy in results_safe:
            print(f"  {name:20s}: Accuracy = {accuracy:.4f}")
        
        # 9. Recomendaciones finales
        print("\n9. 💡 RECOMENDACIONES Y PRÓXIMOS PASOS")
        print("-" * 40)
        
        print("""
        📌 PROBLEMAS IDENTIFICADOS:
        1. Data leakage probable: características como *Diff pueden contener
           información del resultado final
        2. Overfitting severo en algunos modelos
        3. Necesitamos características que estén disponibles ANTES del resultado
        
        🎯 ESTRATEGIA CORREGIDA:
        1. Usar solo características disponibles al inicio de la partida:
           - Composición de campeones
           - Información de región
           - Stats históricos (si los calculamos correctamente)
        
        2. NO usar:
           - Kills, torretas, dragones, barones (son resultados, no predictores)
           - Diferencias (*_Diff) que se calculan del resultado
        
        🔧 MEJORAS SUGERIDAS:
        1. Calcular win rates históricos de campeones (con cuidado temporal)
        2. Añadir información de roles/lanes
        3. Incluir sinergias entre campeones
        4. Usar información de temporada/patch
        
        🚀 PRÓXIMOS PASOS INMEDIATOS:
        1. Entrenar con el dataset seguro (lol_safe_features.csv)
        2. Añadir características históricas calculadas correctamente
        3. Implementar validación temporal (no aleatoria)
        4. Probar modelos más complejos con datos seguros
        """)
        
        # 10. Script para el siguiente paso
        print("\n10. 📜 SCRIPT PARA EL SIGUIENTE PASO")
        print("-" * 40)
        
        next_script = """
        # next_step.py - Mejorar características y entrenar modelos robustos
        
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split, TimeSeriesSplit
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        from sklearn.metrics import accuracy_score, classification_report
        import joblib
        
        # 1. Cargar datos seguros
        df = pd.read_csv("processed_data/lol_safe_features.csv")
        
        # 2. Añadir características históricas (sin data leakage)
        # Esto requiere procesar los datos originales en orden temporal
        
        # 3. Dividir temporalmente (no aleatoriamente)
        # Ordenar por MatchFk si contiene timestamp
        # df = df.sort_values('MatchFk')
        
        # 4. Entrenar con validación temporal
        X = df.drop('BlueWin', axis=1)
        y = df['BlueWin']
        
        # Dividir 80/20 temporalmente
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        # 5. Entrenar modelo robusto
        model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        model.fit(X_train, y_train)
        
        # 6. Evaluar
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy (validación temporal): {accuracy:.4f}")
         
        # 7. Guardar modelo
        joblib.dump(model, "processed_data/robust_model.pkl")
        """
        
        print(next_script)
        
    else:
        print("❌ No se encontró TeamMatchTbl.csv")

print("\n" + "="*70)
print("✅ DIAGNÓSTICO COMPLETADO")
print("="*70)
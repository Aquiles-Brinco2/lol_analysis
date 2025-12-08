# train_models.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Modelos de ML
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)

# Algoritmos
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier

# Manejo de desbalance
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

print("=" * 70)
print("🤖 ENTRENAMIENTO DE MODELOS DE MACHINE LEARNING")
print("=" * 70)

# 1. Cargar datos procesados
print("\n1. 📂 CARGANDO DATOS PROCESADOS...")
data_path = Path("processed_data/lol_processed_data.csv")

if not data_path.exists():
    print(f"❌ ERROR: No se encontró el archivo {data_path}")
    print("💡 Ejecuta primero prepare_data.py para procesar los datos")
    exit()

df = pd.read_csv(data_path)
print(f"✅ Datos cargados: {df.shape[0]:,} filas × {df.shape[1]} columnas")

# 2. Análisis inicial del dataset
print("\n2. 📊 ANÁLISIS INICIAL DEL DATASET")
print("-" * 40)

# Separar características y objetivo
X = df.drop('BlueWin', axis=1)
y = df['BlueWin']

print(f"Características (X): {X.shape[1]} variables")
print(f"Objetivo (y): {len(y)} muestras")
print(f"\nDistribución de clases:")
class_dist = y.value_counts(normalize=True)
for cls, pct in class_dist.items():
    print(f"  Clase {cls}: {pct*100:.2f}% ({y.value_counts()[cls]:,} muestras)")

# Verificar si hay valores nulos
null_counts = df.isnull().sum()
if null_counts.any():
    print(f"\n⚠️  Valores nulos encontrados:")
    for col, count in null_counts[null_counts > 0].items():
        print(f"  {col}: {count} valores nulos ({count/len(df)*100:.2f}%)")
    # Rellenar con la mediana
    X = X.fillna(X.median())
    print("✅ Valores nulos rellenados con la mediana")
else:
    print("✅ No hay valores nulos")

# 3. Dividir datos en entrenamiento y prueba
print("\n3. 🎯 DIVIDIENDO DATOS (80% entrenamiento, 20% prueba)")
print("-" * 40)

# Usar stratified split para mantener proporción de clases
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # Mantener proporción de clases
)

print(f"Conjunto de entrenamiento: {X_train.shape[0]:,} muestras")
print(f"Conjunto de prueba: {X_test.shape[0]:,} muestras")
print(f"\nDistribución en entrenamiento: {y_train.mean()*100:.1f}% victorias azul")
print(f"Distribución en prueba: {y_test.mean()*100:.1f}% victorias azul")

# 4. Escalar características
print("\n4. ⚖️ ESCALANDO CARACTERÍSTICAS...")
print("-" * 40)

# Usar RobustScaler (menos sensible a outliers)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Características escaladas con RobustScaler")

# Convertir de nuevo a DataFrame para mejor visualización
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

# 5. Manejar desbalance de clases (opcional)
print("\n5. ⚖️ MANEJANDO DESBALANCE DE CLASES...")
print("-" * 40)

print(f"Distribución original en entrenamiento:")
print(f"  Clase 0 (Rojo gana): {(y_train == 0).sum():,} muestras")
print(f"  Clase 1 (Azul gana): {(y_train == 1).sum():,} muestras")

# Opción 1: SMOTE (crear muestras sintéticas de la clase minoritaria)
use_smote = True  # Cambiar a False para no usar SMOTE

if use_smote:
    print("\n🔧 Aplicando SMOTE para balancear clases...")
    smote = SMOTE(random_state=42, sampling_strategy=0.8)  # Balancear a 80/20
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"Distribución después de SMOTE:")
    print(f"  Clase 0 (Rojo gana): {(y_train_balanced == 0).sum():,} muestras")
    print(f"  Clase 1 (Azul gana): {(y_train_balanced == 1).sum():,} muestras")
    
    X_train_final, y_train_final = X_train_balanced, y_train_balanced
else:
    print("⚠️  No se aplica balanceo (usar si modelos manejan class_weight)")
    X_train_final, y_train_final = X_train_scaled, y_train

# 6. Definir modelos a evaluar
print("\n6. 🤖 DEFINICIÓN DE MODELOS")
print("-" * 40)

# Configurar modelos con parámetros iniciales
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, 
        random_state=42,
        class_weight='balanced'  # Manejar desbalance
    ),
    
    'Decision Tree': DecisionTreeClassifier(
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    ),
    
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1  # Usar todos los cores
    ),
    
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    ),
    
    'XGBoost': XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1])  # Para desbalance
    ),
    
    'LightGBM': LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        class_weight='balanced'
    ),
    
    'Neural Network': MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        random_state=42,
        early_stopping=True
    )
}

print(f"✓ {len(models)} modelos definidos para evaluación")

# 7. Entrenar y evaluar modelos
print("\n7. 🚀 ENTRENANDO Y EVALUANDO MODELOS")
print("-" * 40)

results = []
feature_importances = {}

# Configurar validación cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for model_name, model in models.items():
    print(f"\n▶️  Entrenando {model_name}...")
    
    try:
        # Entrenar modelo
        model.fit(X_train_final, y_train_final)
        
        # Predecir en conjunto de prueba
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else np.nan
        
        # Validación cruzada
        cv_scores = cross_val_score(model, X_train_final, y_train_final, cv=cv, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Guardar resultados
        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'CV Mean': cv_mean,
            'CV Std': cv_std
        })
        
        # Guardar importancia de características si el modelo la tiene
        if hasattr(model, 'feature_importances_'):
            feature_importances[model_name] = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        print(f"   ✅ Accuracy: {accuracy:.4f} | F1-Score: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
        print(f"   📊 CV Accuracy: {cv_mean:.4f} (±{cv_std:.4f})")
        
    except Exception as e:
        print(f"   ❌ Error entrenando {model_name}: {str(e)}")
        continue

# 8. Comparar resultados
print("\n8. 📈 COMPARACIÓN DE MODELOS")
print("-" * 40)

if results:
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('F1-Score', ascending=False)
    
    print("\n📋 RESULTADOS (ordenados por F1-Score):")
    print("-" * 60)
    print(results_df.to_string(index=False))
    
    # Guardar resultados
    results_df.to_csv("processed_data/model_results.csv", index=False)
    print(f"\n✅ Resultados guardados en processed_data/model_results.csv")
    
    # Visualizar comparación
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'CV Mean']
    colors = plt.cm.Set3(np.linspace(0, 1, len(results_df)))
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        bars = ax.barh(results_df['Model'], results_df[metric], color=colors)
        ax.set_xlabel(metric)
        ax.set_title(f'Comparación de {metric}')
        ax.invert_yaxis()
        
        # Añadir valores en las barras
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{width:.3f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('processed_data/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 9. Análisis del mejor modelo
    print("\n9. 🏆 ANÁLISIS DEL MEJOR MODELO")
    print("-" * 40)
    
    best_model_row = results_df.iloc[0]
    best_model_name = best_model_row['Model']
    print(f"🏅 Mejor modelo: {best_model_name}")
    print(f"   F1-Score: {best_model_row['F1-Score']:.4f}")
    print(f"   Accuracy: {best_model_row['Accuracy']:.4f}")
    print(f"   ROC-AUC: {best_model_row['ROC-AUC']:.4f}")
    
    # Reentrenar el mejor modelo
    print(f"\n🔄 Reentrenando {best_model_name} en todos los datos de entrenamiento...")
    best_model = models[best_model_name]
    
    # Ajustar hiperparámetros si es necesario
    if best_model_name == 'Random Forest':
        print("🔧 Ajustando hiperparámetros para Random Forest...")
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }
        
        grid_search = GridSearchCV(
            best_model, param_grid, cv=3, scoring='f1',
            n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train_final, y_train_final)
        best_model = grid_search.best_estimator_
        print(f"   Mejores parámetros: {grid_search.best_params_}")
    
    elif best_model_name in ['XGBoost', 'LightGBM']:
        print(f"🔧 Ajustando hiperparámetros para {best_model_name}...")
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1]
        }
        
        grid_search = GridSearchCV(
            best_model, param_grid, cv=3, scoring='f1',
            n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train_final, y_train_final)
        best_model = grid_search.best_estimator_
        print(f"   Mejores parámetros: {grid_search.best_params_}")
    
    # Entrenar modelo final
    best_model.fit(X_train_final, y_train_final)
    
    # Evaluación detallada
    print("\n📊 EVALUACIÓN DETALLADA DEL MEJOR MODELO:")
    print("-" * 50)
    
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, 
                        index=['Rojo Gana (Real)', 'Azul Gana (Real)'],
                        columns=['Rojo Gana (Pred)', 'Azul Gana (Pred)'])
    
    print("\n🔢 MATRIZ DE CONFUSIÓN:")
    print(cm_df)
    
    # Reporte de clasificación
    print("\n📄 REPORTE DE CLASIFICACIÓN:")
    print(classification_report(y_test, y_pred, 
                               target_names=['Rojo Gana', 'Azul Gana']))
    
    # Curva ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Curva ROC - {best_model_name}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig('processed_data/roc_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 10. Importancia de características
    print("\n10. 🔍 IMPORTANCIA DE CARACTERÍSTICAS")
    print("-" * 40)
    
    if best_model_name in feature_importances:
        importances = feature_importances[best_model_name]
        
        print(f"\nTop 10 características más importantes ({best_model_name}):")
        print("-" * 50)
        for i, row in importances.head(10).iterrows():
            print(f"{i+1:2d}. {row['feature']:30s}: {row['importance']:.4f}")
        
        # Visualizar importancia
        plt.figure(figsize=(12, 8))
        top_features = importances.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importancia')
        plt.title(f'Top 15 Características Importantes - {best_model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('processed_data/feature_importance.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Guardar importancia
        importances.to_csv("processed_data/feature_importance.csv", index=False)
        print(f"\n✅ Importancia de características guardada")
    
    # 11. Guardar modelo entrenado
    print("\n11. 💾 GUARDANDO MODELO ENTRENADO")
    print("-" * 40)
    
    import joblib
    import pickle
    
    # Guardar con joblib (mejor para sklearn)
    model_path = "processed_data/best_model.pkl"
    joblib.dump(best_model, model_path)
    
    # Guardar scaler
    scaler_path = "processed_data/scaler.pkl"
    joblib.dump(scaler, scaler_path)
    
    # Guardar nombres de características
    features_path = "processed_data/feature_names.pkl"
    with open(features_path, 'wb') as f:
        pickle.dump(list(X.columns), f)
    
    print(f"✅ Modelo guardado en: {model_path}")
    print(f"✅ Scaler guardado en: {scaler_path}")
    print(f"✅ Nombres de características guardados en: {features_path}")
    
    # 12. Prueba con ejemplos específicos
    print("\n12. 🧪 PRUEBA CON EJEMPLOS ESPECÍFICOS")
    print("-" * 40)
    
    # Seleccionar 5 ejemplos aleatorios del conjunto de prueba
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_test), 5, replace=False)
    
    print("\nPredicciones para 5 partidas aleatorias:")
    print("-" * 60)
    print(f"{'Real':<10} {'Predicción':<15} {'Prob. Azul':<12} {'Correcto':<10}")
    print("-" * 60)
    
    for idx in sample_indices:
        real = y_test.iloc[idx]
        features = X_test_scaled[idx].reshape(1, -1)
        pred = best_model.predict(features)[0]
        prob = best_model.predict_proba(features)[0, 1]
        correct = "✓" if real == pred else "✗"
        
        print(f"{'Azul' if real == 1 else 'Rojo':<10} "
              f"{'Azul' if pred == 1 else 'Rojo':<15} "
              f"{prob:.3f}{'':<9} {correct:<10}")
    
    # 13. Resumen final
    print("\n" + "="*70)
    print("🎮 RESUMEN FINAL DEL PROYECTO")
    print("="*70)
    
    print(f"""
    📊 RESULTADOS OBTENIDOS:
    • Mejor modelo: {best_model_name}
    • Accuracy: {best_model_row['Accuracy']:.4f}
    • F1-Score: {best_model_row['F1-Score']:.4f}
    • ROC-AUC: {best_model_row['ROC-AUC']:.4f}
    
    🎯 INTERPRETACIÓN:
    • El modelo puede predecir correctamente el {best_model_row['Accuracy']*100:.1f}% de las partidas
    • El AUC de {best_model_row['ROC-AUC']:.3f} indica buena capacidad discriminativa
    • Las características más importantes varían según el modelo
    
    💡 POSIBLES MEJORAS:
    1. Añadir más características (maestría específica, sinergias entre campeones)
    2. Usar datos por minuto para predicciones en tiempo real
    3. Probar modelos de deep learning con más datos
    4. Añadir información de matchups específicos
    
    🚀 SIGUIENTES PASOS:
    1. El modelo está listo para hacer predicciones nuevas
    2. Puedes crear una API web para servir predicciones
    3. Implementar un dashboard para visualizar resultados
    4. Probar con datos más recientes del juego
    
    📁 ARCHIVOS GENERADOS:
    • processed_data/model_results.csv - Resultados de todos los modelos
    • processed_data/feature_importance.csv - Importancia de características
    • processed_data/best_model.pkl - Modelo entrenado listo para usar
    • processed_data/scaler.pkl - Scaler para nuevos datos
    • Varias visualizaciones en PNG
    """)
    
else:
    print("❌ No se pudieron entrenar modelos. Revisa los datos.")

print("\n" + "="*70)
print("✅ PROCESO COMPLETADO - ¡MODELOS ENTRENADOS!")
print("="*70)
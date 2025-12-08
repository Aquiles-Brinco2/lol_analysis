# League of Legends Match Predictor

Una aplicación de escritorio para predecir qué equipo ganará en una partida de League of Legends basada en la composición de campeones.

## Características

- 🎮 **Interfaz de escritorio con Tkinter** - Aplicación local sin necesidad de servidor
- 📊 **Modelo ML entrenado** - Usa Random Forest con 59.1% de precisión
- ⭐ **Predicciones basadas en Win Rates** - Analiza el desempeño histórico de cada campeón en su rol
- 📈 **Estadísticas en vivo** - Muestra Win Rate y Pick Rate de cada campeón
- 👥 **170+ campeones** - Carga automáticamente la base de datos completa de campeones
- 🛡️ **Validación de composiciones** - Previene campeones repetidos y campos vacíos
- 📁 **Cargar datos personalizados** - Opción para importar listas de campeones desde CSV

## Archivos principales

- `lol_predictor_tkinter.py` - Aplicación principal (interfaz gráfica)
- `train_winrate_model.py` - Script para entrenar el modelo ML
- `archive/` - Datos históricos de partidas
- `enhanced_system/` - Modelo entrenado y datos de win rates

## Instalación

1. Asegúrate de tener Python 3.7+ instalado
2. Instala las dependencias:

```bash
pip install pandas numpy scikit-learn joblib
```

## Uso

### Ejecutar la aplicación

```bash
python lol_predictor_tkinter.py
```

### Hacer una predicción

1. Selecciona 5 campeones para el **Equipo Azul** (uno por rol: Top, Jungle, Mid, ADC, Support)
   - Se mostrarán automáticamente el **Win Rate (WR)** y el **Pick Rate** de cada campeón
2. Selecciona 5 campeones para el **Equipo Rojo**
3. **Restricciones**:
   - ✅ Debes seleccionar un campeón para cada rol
   - ❌ No puedes repetir el mismo campeón en diferentes roles
   - ❌ No puedes usar el mismo campeón en ambos equipos
4. Haz clic en **"🎯 Predecir Resultado"**
5. Verás:
   - Equipo predicho como ganador
   - Probabilidad de victoria para el equipo azul
   - Nivel de confianza de la predicción

### Entrenar un nuevo modelo

Si tienes nuevos datos, puedes entrenar un modelo actualizado:

```bash
python train_winrate_model.py
```

Este script:
- Calcula win rates por campeón y rol
- Entrena un modelo RandomForest
- Guarda el modelo en `enhanced_system/`

## Cómo funcionan las predicciones

El modelo analiza:

1. **Win rates históricos** - El porcentaje de victorias de cada campeón en su rol específico
   - Se muestra en tiempo real al seleccionar: `WR: 55.2%`
2. **Pick rate / Popularidad** - Qué tan frecuentemente es seleccionado cada campeón
   - Se muestra en tiempo real: `Pick: 12.5%`
3. **Composición del equipo** - Suma y promedio de win rates del equipo
4. **Diferencia de poder** - Compara la ventaja de una composición sobre la otra
5. **Validaciones**:
   - Asegura que no haya campos vacíos
   - Previene campeones duplicados
   - Verifica composiciones válidas

### Importancia de features

Las características más importantes para la predicción son:
- ADC (Bot lane) - 13% de importancia
- Mid lane - 10% de importancia
- Top lane - 9% de importancia
- Jungle - 8% de importancia

## Precisión del modelo

- **Precisión en entrenamiento**: 59.1%
- **Mejora sobre adivinar al azar**: +9.1%
- **Datos de entrenamiento**: 97,883 partidas históricas

## Notas importantes

- El modelo se basa en datos históricos y puede no reflejar el meta actual
- Las predicciones son estadísticas, no garantías
- Los campeones desconocidos (no en la base de datos) se tratan con un win rate del 50%

## Estructura del proyecto

```
lol_analysis-main/
├── lol_predictor_tkinter.py      # Aplicación principal
├── train_winrate_model.py         # Entrenamiento del modelo
├── archive/                       # Datos históricos
│   ├── ChampionTbl.csv           # Base de datos de campeones
│   ├── MatchTbl.csv              # Información de partidas
│   ├── MatchStatsTbl.csv         # Estadísticas por jugador
│   └── TeamMatchTbl.csv          # Composiciones de equipos
└── enhanced_system/              # Modelo entrenado
    ├── enhanced_model.pkl        # Modelo RandomForest
    ├── scaler.pkl                # Escalador de features
    ├── winrate_data.pkl          # Win rates por campeón/rol
    └── feature_names.pkl         # Nombres de features
```

## Desarrollo

El proyecto usa:
- **Pandas** - Procesamiento de datos
- **Scikit-learn** - Machine Learning
- **Tkinter** - Interfaz gráfica
- **Joblib** - Serialización de modelos

## Licencia

Este proyecto es de código abierto para propósitos educativos.

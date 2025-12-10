import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')
 
class WinRateModelTrainer:
    def __init__(self, data_dir="archive"):
        self.data_dir = Path(data_dir)
        self.match_stats = None
        self.team_matches = None
        self.champion_names = None
        self.winrate_data = {}
        self.model = None
        self.scaler = None
   
    def load_data(self):
        print("Cargando datos...")
        self.champion_names = pd.read_csv(self.data_dir / "ChampionTbl.csv")
        self.match_stats = pd.read_csv(self.data_dir / "MatchStatsTbl.csv")
        self.team_matches = pd.read_csv(self.data_dir / "TeamMatchTbl.csv")
        self.match_info = pd.read_csv(self.data_dir / "MatchTbl.csv")
       
        print("Datos cargados:")
        print(f"   - {len(self.match_stats)} registros en MatchStats")
        print(f"   - {len(self.team_matches)} registros en TeamMatch")
        print(f"   - {len(self.champion_names)} campeones")
   
    def calculate_winrates(self):
        print("Calculando win rates por campeón y rol...")
       
        champ_id_to_name = dict(zip(
            self.champion_names['ChampionId'],
            self.champion_names['ChampionName']
        ))
       
        valid_stats = self.match_stats[
            (self.match_stats['Lane'].notna()) &
            (self.match_stats['Lane'] != 'NONE')
        ].copy()
       
        valid_stats['ChampionName'] = valid_stats['EnemyChampionFk'].map(champ_id_to_name)
       
        champion_role_stats = valid_stats.groupby(['ChampionName', 'Lane']).agg({
            'Win': ['sum', 'count', 'mean']
        }).reset_index()
       
        champion_role_stats.columns = ['Champion', 'Role', 'Wins', 'Total', 'WinRate']
        champion_role_stats = champion_role_stats[champion_role_stats['Total'] >= 10]
       
        self.winrate_data = {}
        for _, row in champion_role_stats.iterrows():
            champ = row['Champion']
            role = row['Role']
            winrate = row['WinRate']
            if champ not in self.winrate_data:
                self.winrate_data[champ] = {}
            self.winrate_data[champ][role] = winrate
       
        print(f"Win rates calculados para {len(self.winrate_data)} campeones")
       
        print("Top 10 campeones por win rate general:")
        avg_winrates = {}
        for champ, roles in self.winrate_data.items():
            avg_winrates[champ] = np.mean(list(roles.values()))
       
        for champ, wr in sorted(avg_winrates.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {champ}: {wr:.1%}")
       
        return self.winrate_data
   
    def create_training_data(self):
        print("Creando datos de entrenamiento...")
       
        role_mapping = {
            0: 'TOP',
            1: 'JUNGLE',
            2: 'MIDDLE',
            3: 'BOTTOM',
            4: 'SUPPORT'
        }
       
        champ_id_to_name = dict(zip(
            self.champion_names['ChampionId'],
            self.champion_names['ChampionName']
        ))
       
        features_list = []
        labels = []
       
        for _, row in self.team_matches.iterrows():
            blue_champs = [
                champ_id_to_name.get(int(row['B1Champ']), 'Unknown'),
                champ_id_to_name.get(int(row['B2Champ']), 'Unknown'),
                champ_id_to_name.get(int(row['B3Champ']), 'Unknown'),
                champ_id_to_name.get(int(row['B4Champ']), 'Unknown'),
                champ_id_to_name.get(int(row['B5Champ']), 'Unknown'),
            ]
           
            red_champs = [
                champ_id_to_name.get(int(row['R1Champ']), 'Unknown'),
                champ_id_to_name.get(int(row['R2Champ']), 'Unknown'),
                champ_id_to_name.get(int(row['R3Champ']), 'Unknown'),
                champ_id_to_name.get(int(row['R4Champ']), 'Unknown'),
                champ_id_to_name.get(int(row['R5Champ']), 'Unknown'),
            ]
           
            features = self.create_match_features(blue_champs, red_champs, role_mapping)
           
            if features is not None:
                features_list.append(features)
                labels.append(int(row['BlueWin']))
       
        if features_list:
            X = np.array(features_list)
            y = np.array(labels)
            print(f"Datos de entrenamiento creados: {len(X)} muestras")
            print(f"   - Victorias Azul: {sum(y)} ({100*sum(y)/len(y):.1f}%)")
            return X, y
        else:
            print("No se pudieron crear datos de entrenamiento")
            return None, None
   
    def create_match_features(self, blue_champs, red_champs, role_mapping):
        features = []
       
        blue_winrates = []
        red_winrates = []
       
        for i, role_name in enumerate(['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'SUPPORT']):
            blue_champ = blue_champs[i]
            if blue_champ in self.winrate_data and role_name in self.winrate_data[blue_champ]:
                blue_wr = self.winrate_data[blue_champ][role_name]
            else:
                blue_wr = 0.5
            blue_winrates.append(blue_wr)
           
            red_champ = red_champs[i]
            if red_champ in self.winrate_data and role_name in self.winrate_data[red_champ]:
                red_wr = self.winrate_data[red_champ][role_name]
            else:
                red_wr = 0.5
            red_winrates.append(red_wr)
       
        features.extend(blue_winrates)
        features.extend(red_winrates)
        features.append(sum(blue_winrates))
        features.append(sum(red_winrates))
        features.append(np.mean(blue_winrates))
        features.append(np.mean(red_winrates))
        features.append(sum(blue_winrates) - sum(red_winrates))
       
        return features
   
    def train_model(self, X, y):
        print("Entrenando modelo...")
       
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
       
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.model.fit(X_scaled, y)
       
        train_score = self.model.score(X_scaled, y)
        print(f"Modelo entrenado")
        print(f"   - Precisión en entrenamiento: {train_score:.1%}")
       
        feature_names = [
            'Blue_TOP', 'Blue_JGL', 'Blue_MID', 'Blue_ADC', 'Blue_SUP',
            'Red_TOP', 'Red_JGL', 'Red_MID', 'Red_ADC', 'Red_SUP',
            'Blue_Sum', 'Red_Sum', 'Blue_Mean', 'Red_Mean', 'Diff'
        ]
       
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
       
        print("Importancia de features:")
        for _, row in importance_df.head(8).iterrows():
            print(f"   {row['Feature']}: {row['Importance']:.3f}")
       
        return self.model
   
    def save_model(self, output_dir="enhanced_system"):
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
       
        print(f"Guardando modelo en {output_dir}...")
       
        joblib.dump(self.model, output_path / "enhanced_model.pkl")
        joblib.dump(self.scaler, output_path / "scaler.pkl")
        joblib.dump(self.winrate_data, output_path / "winrate_data.pkl")
       
        feature_names = [
            'Blue_TOP', 'Blue_JGL', 'Blue_MID', 'Blue_ADC', 'Blue_SUP',
            'Red_TOP', 'Red_JGL', 'Red_MID', 'Red_ADC', 'Red_SUP',
            'Blue_Sum', 'Red_Sum', 'Blue_Mean', 'Red_Mean', 'Diff'
        ]
        joblib.dump(feature_names, output_path / "feature_names.pkl")
       
        print("Modelo guardado exitosamente")
 
def main():
    trainer = WinRateModelTrainer()
    trainer.load_data()
    trainer.calculate_winrates()
    X, y = trainer.create_training_data()
   
    if X is not None:
        trainer.train_model(X, y)
        trainer.save_model()
        print("Modelo entrenado y guardado exitosamente")
    else:
        print("Error al crear datos de entrenamiento")
 
if __name__ == "__main__":
    main()
 
 

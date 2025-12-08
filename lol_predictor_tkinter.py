# lol_predictor_tkinter.py
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import joblib
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class LoLPredictorApp:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.champ_stats = {}
        self.champ_id_to_name = {}
        self.champ_name_to_id = {}
        self.winrate_data = {}
        self.pick_stats = {}  # Estadísticas de popularidad
        self._load_system()
    
    def _load_system(self):
        """Cargar el sistema entrenado"""
        system_path = Path("enhanced_system")
        
        # Intentar cargar sistema mejorado primero
        model_file = system_path / "enhanced_model.pkl"
        
        if not model_file.exists():
            # Intentar con sistema básico
            system_path = Path("complete_system")
            model_file = system_path / "lol_model.pkl"
        
        if model_file.exists():
            try:
                self.model = joblib.load(model_file)
                print("✅ Modelo cargado exitosamente")
            except:
                print("❌ Error cargando el modelo")
                return
        
        # Cargar scaler
        scaler_file = system_path / "scaler.pkl"
        if scaler_file.exists():
            try:
                self.scaler = joblib.load(scaler_file)
            except:
                print("⚠️ No se pudo cargar el scaler")
        
        # Cargar nombres de características
        features_file = system_path / "feature_names.pkl"
        if features_file.exists():
            try:
                with open(features_file, 'rb') as f:
                    self.feature_names = pickle.load(f)
            except:
                print("⚠️ No se pudieron cargar los nombres de características")
        
        # Cargar datos de campeones
        self._load_champion_data()
    
    def _load_champion_data(self):
        """Cargar datos de campeones"""
        try:
            # Cargar desde archivo CSV
            script_dir = Path(__file__).parent
            champ_file = script_dir / "archive" / "ChampionTbl.csv"
            if champ_file.exists():
                champs_df = pd.read_csv(champ_file)
                self.champ_id_to_name = dict(zip(champs_df['ChampionId'], champs_df['ChampionName']))
                self.champ_name_to_id = {v: k for k, v in self.champ_id_to_name.items()}
                print(f"✅ {len(self.champ_id_to_name)} campeones cargados")
            else:
                # Crear datos de ejemplo si no existe el archivo
                print("⚠️ No se encontró archivo de campeones, usando datos de ejemplo")
                self._create_sample_champs()
        except Exception as e:
            print(f"❌ Error cargando datos de campeones: {e}")
            self._create_sample_champs()
        
        # Cargar win rates si existen
        self._load_winrate_data()
    
    def _calculate_pick_stats(self):
        """Calcular estadísticas de popularidad (pick rate) por campeón"""
        try:
            script_dir = Path(__file__).parent
            
            # Cargar datos de MatchStats para contar picks
            match_stats_file = script_dir / "archive" / "MatchStatsTbl.csv"
            if match_stats_file.exists():
                match_stats = pd.read_csv(match_stats_file)
                
                # Crear diccionario de campeón ID a nombre
                champ_id_to_name = dict(zip(
                    self.champ_id_to_name.values(),
                    self.champ_id_to_name.keys()
                ))
                
                # Invertir para obtener nombre a ID
                champ_name_to_id = {v: k for k, v in self.champ_id_to_name.items()}
                
                # Contar picks por campeón
                champion_picks = match_stats['EnemyChampionFk'].value_counts()
                total_picks = len(match_stats)
                
                # Calcular pick rate y win rate por campeón
                for champ_name, champ_id in champ_name_to_id.items():
                    picks = champion_picks.get(champ_id, 0)
                    pick_rate = picks / total_picks if total_picks > 0 else 0
                    
                    self.pick_stats[champ_name] = {
                        'picks': picks,
                        'pick_rate': pick_rate,
                        'popularity': min(100, int(pick_rate * 1000))  # Normalizar a escala 0-100
                    }
                
                print(f"✅ Estadísticas de popularidad calculadas para {len(self.pick_stats)} campeones")
            else:
                print("⚠️ No se encontró archivo de MatchStats para estadísticas de popularidad")
        except Exception as e:
            print(f"⚠️ Error calculando estadísticas de popularidad: {e}")
    
    def _load_winrate_data(self):
        """Cargar datos de win rates por campeón y rol"""
        try:
            script_dir = Path(__file__).parent
            winrate_file = script_dir / "enhanced_system" / "winrate_data.pkl"
            if winrate_file.exists():
                self.winrate_data = joblib.load(winrate_file)
                print(f"✅ Win rates cargados para {len(self.winrate_data)} campeones")
            else:
                print("⚠️ No se encontró archivo de win rates")
                self.winrate_data = {}
        except Exception as e:
            print(f"❌ Error cargando win rates: {e}")
            self.winrate_data = {}
        
        # Calcular estadísticas de popularidad
        self._calculate_pick_stats()
    
    def _create_sample_champs(self):
        """Crear datos de campeones de ejemplo"""
        sample_champs = {
            1: "Annie", 2: "Olaf", 3: "Galio", 4: "TwistedFate", 5: "XinZhao",
            6: "Urgot", 7: "Leblanc", 8: "Vladimir", 9: "Fiddlesticks", 10: "Kayle",
            11: "MasterYi", 12: "Alistar", 13: "Ryze", 14: "Sion", 15: "Sivir",
            16: "Soraka", 17: "Teemo", 18: "Tristana", 19: "Warwick", 20: "Nunu",
            21: "MissFortune", 22: "Ashe", 23: "Tryndamere", 24: "Jax", 25: "Morgana",
            26: "Zilean", 27: "Singed", 28: "Evelynn", 29: "Twitch", 30: "Karthus",
            31: "Chogath", 32: "Amumu", 33: "Rammus", 34: "Anivia", 35: "Shaco",
            36: "DrMundo", 37: "Sona", 38: "Kassadin", 39: "Irelia", 40: "Janna"
        }
        self.champ_id_to_name = sample_champs
        self.champ_name_to_id = {v: k for k, v in sample_champs.items()}
    
    def predict_match(self, blue_champs, red_champs, region="EUW"):
        """Predecir resultado de una partida"""
        if self.model is None:
            return self._simple_prediction(blue_champs, red_champs)
        
        # Crear características basadas en win rates
        features = self._create_features(blue_champs, red_champs, region)
        
        if features is None:
            return self._simple_prediction(blue_champs, red_champs)
        
        # Escalar si hay scaler
        if self.scaler is not None:
            try:
                features_scaled = self.scaler.transform([features])
            except:
                # Si hay error en escalado, usar sin escalar
                features_scaled = [features]
        else:
            features_scaled = [features]
        
        try:
            # Predecir
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0][1]
            
            return {
                'winner': 'Azul' if prediction == 1 else 'Rojo',
                'probability_blue': float(probability),
                'confidence': self._get_confidence(probability)
            }
        except Exception as e:
            print(f"Error en predicción: {e}")
            # Si falla, usar lógica simple basada en win rates
            return self._simple_prediction(blue_champs, red_champs)
    
    def _create_features(self, blue_champs, red_champs, region):
        """Crear características para predicción basadas en win rates"""
        # Mapeo de roles por posición
        role_mapping = {
            0: 'TOP',
            1: 'JUNGLE',
            2: 'MIDDLE',
            3: 'BOTTOM',  # ADC
            4: 'SUPPORT'
        }
        
        features = []
        
        # Obtener win rates para cada equipo por rol
        blue_winrates = []
        red_winrates = []
        
        for i, role_name in enumerate(role_mapping.values()):
            # Win rate azul en este rol
            blue_champ = blue_champs[i]
            if blue_champ in self.winrate_data and role_name in self.winrate_data[blue_champ]:
                blue_wr = self.winrate_data[blue_champ][role_name]
            else:
                blue_wr = 0.5  # Default 50% si no hay datos
            blue_winrates.append(blue_wr)
            
            # Win rate rojo en este rol
            red_champ = red_champs[i]
            if red_champ in self.winrate_data and role_name in self.winrate_data[red_champ]:
                red_wr = self.winrate_data[red_champ][role_name]
            else:
                red_wr = 0.5
            red_winrates.append(red_wr)
        
        # Features: win rates de cada rol + suma/media + diferencia
        features.extend(blue_winrates)  # 5 features
        features.extend(red_winrates)   # 5 features
        features.append(sum(blue_winrates))  # Suma azul
        features.append(sum(red_winrates))   # Suma rojo
        features.append(np.mean(blue_winrates))  # Media azul
        features.append(np.mean(red_winrates))   # Media rojo
        features.append(sum(blue_winrates) - sum(red_winrates))  # Diferencia
        
        return features
    
    def _simple_prediction(self, blue_champs, red_champs):
        """Predicción simple basada en win rates"""
        # Mapeo de roles
        role_mapping = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'SUPPORT']
        
        blue_total_wr = 0
        red_total_wr = 0
        valid_champs = 0
        
        for i, role_name in enumerate(role_mapping):
            blue_champ = blue_champs[i]
            red_champ = red_champs[i]
            
            # Obtener win rates
            blue_wr = 0.5
            red_wr = 0.5
            
            if blue_champ in self.winrate_data and role_name in self.winrate_data[blue_champ]:
                blue_wr = self.winrate_data[blue_champ][role_name]
            
            if red_champ in self.winrate_data and role_name in self.winrate_data[red_champ]:
                red_wr = self.winrate_data[red_champ][role_name]
            
            blue_total_wr += blue_wr
            red_total_wr += red_wr
            valid_champs += 1
        
        # Calcular probabilidad basada en win rates
        if valid_champs > 0:
            blue_avg_wr = blue_total_wr / valid_champs
            red_avg_wr = red_total_wr / valid_champs
            
            # Normalizar a probabilidad
            total = blue_avg_wr + red_avg_wr
            if total > 0:
                prob = blue_avg_wr / total
            else:
                prob = 0.5
        else:
            prob = 0.5
        
        return {
            'winner': 'Azul' if prob > 0.5 else 'Rojo',
            'probability_blue': prob,
            'confidence': 'Alta' if abs(prob - 0.5) > 0.15 else 'Baja'
        }
    
    def _get_confidence(self, probability):
        """Obtener nivel de confianza"""
        diff = abs(probability - 0.5)
        if diff > 0.3:
            return "Alta"
        elif diff > 0.15:
            return "Media"
        else:
            return "Baja"


class LoLPredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎮 League of Legends Match Predictor")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1a1a1a")
        
        # Cargar predictor
        self.predictor = LoLPredictorApp()
        self.champ_names = sorted(self.predictor.champ_id_to_name.values())
        
        self.blue_champs = {}
        self.red_champs = {}
        
        self.setup_ui()
    
    def setup_ui(self):
        """Configurar la interfaz"""
        # Título
        title_frame = tk.Frame(self.root, bg="#1a1a1a")
        title_frame.pack(fill=tk.X, padx=10, pady=10)
        
        title = tk.Label(title_frame, text="🎮 League of Legends Match Predictor",
                        font=("Arial", 16, "bold"), bg="#1a1a1a", fg="#00ff00")
        title.pack()
        
        subtitle = tk.Label(title_frame, text="Predice qué equipo ganará basado en la composición de campeones",
                           font=("Arial", 10), bg="#1a1a1a", fg="#cccccc")
        subtitle.pack()
        
        # Barra de herramientas
        toolbar = tk.Frame(self.root, bg="#2a2a2a")
        toolbar.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(toolbar, text="📁 Cargar Campeones desde CSV", command=self.load_champions_csv,
                 bg="#4a4a4a", fg="#00ff00", padx=10, pady=5).pack(side=tk.LEFT, padx=5)
        
        tk.Label(toolbar, text=f"Campeones cargados: {len(self.champ_names)}",
                bg="#2a2a2a", fg="#00ff00").pack(side=tk.LEFT, padx=20)
        
        # Frame principal
        main_frame = tk.Frame(self.root, bg="#1a1a1a")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Columna Azul
        blue_frame = tk.LabelFrame(main_frame, text="🔵 Equipo Azul", font=("Arial", 12, "bold"),
                                   bg="#1a1a1a", fg="#00aaff", padx=10, pady=10)
        blue_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        roles = ["Top", "Jungle", "Mid", "ADC", "Support"]
        for role in roles:
            self.create_selector(blue_frame, role, "blue")
        
        # Columna Roja
        red_frame = tk.LabelFrame(main_frame, text="🔴 Equipo Rojo", font=("Arial", 12, "bold"),
                                  bg="#1a1a1a", fg="#ff4444", padx=10, pady=10)
        red_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        for role in roles:
            self.create_selector(red_frame, role, "red")
        
        # Frame de predicción y resultados
        bottom_frame = tk.Frame(self.root, bg="#1a1a1a")
        bottom_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Botón predecir
        predict_btn = tk.Button(bottom_frame, text="🎯 Predecir Resultado", command=self.predict,
                               bg="#00aa00", fg="#000000", font=("Arial", 12, "bold"),
                               padx=20, pady=10)
        predict_btn.pack(pady=10)
        
        # Frame de resultados
        self.result_frame = tk.Frame(bottom_frame, bg="#2a2a2a", relief=tk.SUNKEN, bd=1)
        self.result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.result_text = tk.Text(self.result_frame, bg="#1a1a1a", fg="#00ff00",
                                   font=("Courier", 10), wrap=tk.WORD, state=tk.DISABLED)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def create_selector(self, parent, role, team):
        """Crear un selector de campeón con estadísticas"""
        frame = tk.Frame(parent, bg="#1a1a1a")
        frame.pack(fill=tk.X, pady=8)
        
        # Label del rol
        label = tk.Label(frame, text=f"{role}:", bg="#1a1a1a", fg="#cccccc", width=10, font=("Arial", 10, "bold"))
        label.pack(side=tk.LEFT, padx=5)
        
        # Variable para el combobox
        var = tk.StringVar(value="")
        
        # Combobox
        combo = ttk.Combobox(frame, textvariable=var, values=[""] + self.champ_names,
                            state="readonly", width=25)
        combo.pack(side=tk.LEFT, padx=5)
        
        # Frame para estadísticas
        stats_frame = tk.Frame(frame, bg="#1a1a1a")
        stats_frame.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        # Label para win rate
        wr_label = tk.Label(stats_frame, text="WR: -", bg="#1a1a1a", fg="#ffaa00", font=("Arial", 9))
        wr_label.pack(side=tk.LEFT, padx=5)
        
        # Label para popularidad
        pop_label = tk.Label(stats_frame, text="Pick: -", bg="#1a1a1a", fg="#00ccff", font=("Arial", 9))
        pop_label.pack(side=tk.LEFT, padx=5)
        
        # Función para actualizar estadísticas cuando se selecciona
        def on_select(event=None):
            selected = var.get()
            
            # Validar sin repetidos
            if selected and team == "blue":
                other_champs = [v.get() for v in self.red_champs.values() if v.get()]
            elif selected and team == "red":
                other_champs = [v.get() for v in self.blue_champs.values() if v.get()]
            else:
                other_champs = []
            
            # Si el campeón está repetido, mostrar error
            if selected in other_champs:
                messagebox.showerror("Error", f"❌ {selected} ya está seleccionado en el otro equipo")
                var.set("")
                wr_label.config(text="WR: -")
                pop_label.config(text="Pick: -")
                return
            
            # Obtener estadísticas del campeón
            if selected:
                # Obtener WR promedio
                if selected in self.predictor.winrate_data:
                    winrates = self.predictor.winrate_data[selected]
                    avg_wr = np.mean(list(winrates.values())) * 100
                    wr_text = f"WR: {avg_wr:.1f}%"
                else:
                    wr_text = "WR: N/A"
                
                # Obtener popularidad
                if selected in self.predictor.pick_stats:
                    pick_rate = self.predictor.pick_stats[selected]['pick_rate'] * 100
                    pop_text = f"Pick: {pick_rate:.1f}%"
                else:
                    pop_text = "Pick: N/A"
                
                wr_label.config(text=wr_text)
                pop_label.config(text=pop_text)
            else:
                wr_label.config(text="WR: -")
                pop_label.config(text="Pick: -")
        
        combo.bind("<<ComboboxSelected>>", on_select)
        
        if team == "blue":
            self.blue_champs[role] = var
        else:
            self.red_champs[role] = var
    
    def load_champions_csv(self):
        """Cargar campeones desde CSV"""
        file_path = filedialog.askopenfilename(
            title="Selecciona archivo CSV de campeones",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                champs_df = pd.read_csv(file_path)
                if 'ChampionId' in champs_df.columns and 'ChampionName' in champs_df.columns:
                    champ_dict = dict(zip(champs_df['ChampionId'], champs_df['ChampionName']))
                    self.predictor.champ_id_to_name = champ_dict
                    self.predictor.champ_name_to_id = {v: k for k, v in champ_dict.items()}
                    self.champ_names = sorted(champ_dict.values())
                    
                    # Actualizar todos los combos
                    for role_var in list(self.blue_champs.values()) + list(self.red_champs.values()):
                        # Obtener el combo y actualizar su lista
                        pass
                    
                    # Recrear la interfaz
                    self.root.destroy()
                    root = tk.Tk()
                    app = LoLPredictorGUI(root)
                    root.mainloop()
                    
                    messagebox.showinfo("Éxito", f"✅ {len(champ_dict)} campeones cargados exitosamente")
                else:
                    messagebox.showerror("Error", "❌ El archivo debe tener columnas 'ChampionId' y 'ChampionName'")
            except Exception as e:
                messagebox.showerror("Error", f"❌ Error cargando el archivo: {e}")
    
    def predict(self):
        """Hacer predicción"""
        # Obtener campeones seleccionados
        blue_selected = [self.blue_champs[role].get() for role in self.blue_champs.keys()]
        red_selected = [self.red_champs[role].get() for role in self.red_champs.keys()]
        
        # Validar que no haya campos vacíos
        if any(not c for c in blue_selected):
            messagebox.showerror("Error", "❌ Por favor selecciona un campeón para CADA rol del equipo azul")
            return
        if any(not c for c in red_selected):
            messagebox.showerror("Error", "❌ Por favor selecciona un campeón para CADA rol del equipo rojo")
            return
        
        # Validar que no haya repetidos dentro del mismo equipo
        if len(set(blue_selected)) < len(blue_selected):
            messagebox.showerror("Error", "❌ No puedes repetir campeones en el equipo azul")
            return
        if len(set(red_selected)) < len(red_selected):
            messagebox.showerror("Error", "❌ No puedes repetir campeones en el equipo rojo")
            return
        
        # Validar que no haya campeones iguales entre equipos
        blue_set = set(blue_selected)
        red_set = set(red_selected)
        if blue_set & red_set:
            messagebox.showerror("Error", "❌ No puedes usar el mismo campeón en ambos equipos")
            return
        
        # Predicción
        result = self.predictor.predict_match(blue_selected, red_selected)
        
        if result:
            # Mostrar resultado
            output = f"""
╔══════════════════════════════════════════════════════════════╗
║                    RESULTADO DE LA PREDICCIÓN                ║
╚══════════════════════════════════════════════════════════════╝

🏆 EQUIPO GANADOR PREDICHO: {result['winner'].upper()}

📊 PROBABILIDADES:
  • Equipo Azul:  {result['probability_blue']:.1%}
  • Equipo Rojo:  {1 - result['probability_blue']:.1%}

🎯 CONFIANZA: {result['confidence']}

═══════════════════════════════════════════════════════════════

📋 COMPOSICIÓN AZUL:
"""
            for role, champ in zip(self.blue_champs.keys(), blue_selected):
                output += f"  • {role}: {champ}\n"
            
            output += f"""
📋 COMPOSICIÓN ROJA:
"""
            for role, champ in zip(self.red_champs.keys(), red_selected):
                output += f"  • {role}: {champ}\n"
            
            output += "\n═══════════════════════════════════════════════════════════════\n"
            
            # Mostrar en el text widget
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert("1.0", output)
            self.result_text.config(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    app = LoLPredictorGUI(root)
    root.mainloop()

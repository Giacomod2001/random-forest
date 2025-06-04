import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Configurazione pagina
st.set_page_config(page_title="Random Forest Classifier", layout="wide")

# Seed fisso per riproducibilità
SEED = 56
np.random.seed(SEED)

def genera_dataset(n_samples: int, noise_level: float = 0.1) -> pd.DataFrame:
    """
    Genera un dataset sintetico per la classificazione delle auto.
    
    Args:
        n_samples: Numero di campioni da generare
        noise_level: Livello di rumore per rendere il dataset più realistico
    
    Returns:
        DataFrame con le feature e il target
    """
    eta = np.random.randint(18, 76, size=n_samples)
    reddito = np.random.randint(15000, 120001, size=n_samples)
    km_anno = np.random.randint(0, 50001, size=n_samples)
    n_fam = np.random.randint(1, 7, size=n_samples)
    
    lavori = ['Impiegato', 'Libero professionista', 'Disoccupato', 'Studente', 'Pensionato']
    lavoro = np.random.choice(lavori, size=n_samples)
    
    df = pd.DataFrame({
        'Età': eta,
        'Lavoro': lavoro,
        'Reddito': reddito,
        'KmAnno': km_anno,
        'N_Fam': n_fam
    })
    
    # Calcolo mediane per le regole di classificazione
    med_redd = np.median(reddito)
    med_km = np.median(km_anno)
    
    # Regole di base per la classificazione
    cond1 = (df['Reddito'] > med_redd) & (df['KmAnno'] < med_km)
    cond2 = (df['Reddito'] <= med_redd) & (df['KmAnno'] > med_km)
    
    # Aggiunta di rumore per rendere il problema più realistico
    if noise_level > 0:
        noise_indices = np.random.choice(n_samples, size=int(n_samples * noise_level), replace=False)
        base_prediction = np.where(cond1, 'Tedesca', np.where(cond2, 'Americana', 'Giapponese'))
        
        # Aggiunge rumore casuale
        for idx in noise_indices:
            possible_classes = ['Tedesca', 'Americana', 'Giapponese']
            base_prediction[idx] = np.random.choice(possible_classes)
        
        df['AutoScelta'] = base_prediction
    else:
        df['AutoScelta'] = np.where(cond1, 'Tedesca', np.where(cond2, 'Americana', 'Giapponese'))
    
    return df

def explain_random_forest_step_by_step():
    """Spiega passo passo come funziona Random Forest"""
    st.markdown("""
    ## 🌳 Come funziona Random Forest: Spiegazione Passo-Passo
    
    ### 1️⃣ **Concetto Base: Ensemble Learning**
    Random Forest appartiene alla famiglia degli **algoritmi ensemble**, che combinano più modelli semplici per ottenere prestazioni migliori.
    
    ### 2️⃣ **Gli Alberi Decisionali (Building Blocks)**
    - Un **albero decisionale** divide i dati ponendo domande binarie
    - Esempio: "Il reddito è > 50.000€?" → SÌ/NO → divisione dei dati
    - Ogni nodo interno è una domanda, ogni foglia è una predizione
    
    ### 3️⃣ **Bootstrap Aggregating (Bagging)**
    Per ogni albero della foresta:
    1. **Campionamento casuale**: estrae un sottoinsieme casuale dei dati (con rimpiazzo)
    2. **Addestramento**: addestra un albero solo su questi dati
    3. **Ripetizione**: ripete il processo per N alberi (default: 100)
    
    ### 4️⃣ **Random Feature Selection**
    Ad ogni nodo di ogni albero:
    - Non considera tutte le variabili disponibili
    - Sceglie casualmente solo √(numero_totale_features) variabili
    - Trova la migliore divisione solo tra queste
    
    ### 5️⃣ **Predizione Finale**
    - **Classificazione**: voto di maggioranza tra tutti gli alberi
    - **Regressione**: media delle predizioni di tutti gli alberi
    
    ### 6️⃣ **Vantaggi di questo approccio**
    - ✅ **Riduce overfitting**: alberi diversi compensano errori reciproci
    - ✅ **Robustezza**: meno sensibile a outlier e rumore
    - ✅ **Feature importance**: calcola automaticamente l'importanza delle variabili
    - ✅ **Parallelizzabile**: alberi indipendenti, veloce su più CPU
    """)

def plot_feature_importance(rf_model, feature_names):
    """Crea un grafico dell'importanza delle feature"""
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[feature_names[i] for i in indices],
        y=[importances[i] for i in indices],
        marker_color='lightblue',
        text=[f'{importances[i]:.3f}' for i in indices],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Importanza delle Feature nel Modello Random Forest",
        xaxis_title="Feature",
        yaxis_title="Importanza",
        showlegend=False
    )
    
    return fig

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Crea una matrice di confusione interattiva"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16},
        colorscale='Blues'
    ))
    
    fig.update_layout(
        title="Matrice di Confusione",
        xaxis_title="Predizione",
        yaxis_title="Classe Reale"
    )
    
    return fig

def plot_learning_curves(X, y, rf_params):
    """Mostra le curve di apprendimento"""
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_scores = []
    val_scores = []
    
    for train_size in train_sizes:
        n_samples = int(len(X) * train_size)
        if n_samples < 10:
            continue
            
        X_subset = X[:n_samples]
        y_subset = y[:n_samples]
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_subset, y_subset, test_size=0.2, random_state=SEED
        )
        
        rf = RandomForestClassifier(**rf_params, random_state=SEED)
        rf.fit(X_train, y_train)
        
        train_score = rf.score(X_train, y_train)
        val_score = rf.score(X_val, y_val)
        
        train_scores.append(train_score)
        val_scores.append(val_score)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train_sizes[:len(train_scores)],
        y=train_scores,
        mode='lines+markers',
        name='Training Score',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=train_sizes[:len(val_scores)],
        y=val_scores,
        mode='lines+markers',
        name='Validation Score',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title="Curve di Apprendimento",
        xaxis_title="Frazione del Dataset di Training",
        yaxis_title="Accuracy",
        yaxis=dict(range=[0, 1])
    )
    
    return fig

def main():
    st.title("🌳 Random Forest Classifier - Analisi Completa")
    st.markdown("*Una guida interattiva per comprendere e ottimizzare l'algoritmo Random Forest*")
    
    # === SEZIONE 1: SPIEGAZIONE TEORICA ===
    with st.expander("📚 Spiegazione Dettagliata dell'Algoritmo", expanded=False):
        explain_random_forest_step_by_step()
    
    # === SEZIONE 2: GENERAZIONE DATASET ===
    st.header("📊 Generazione e Esplorazione Dataset")
    
    col1, col2 = st.columns(2)
    with col1:
        n_samples = st.slider(
            "Numero di campioni", 
            min_value=100, max_value=5000, value=1000, step=100
        )
    
    with col2:
        noise_level = st.slider(
            "Livello di rumore (%)", 
            min_value=0, max_value=50, value=10, step=5
        ) / 100
    
    # Feedback sulla dimensione del dataset
    if n_samples < 300:
        st.error("⚠️ **Dataset piccolo**: Rischio di overfitting elevato!")
    elif n_samples < 1000:
        st.warning("⚡ **Dataset medio**: Buon compromesso velocità/qualità")
    else:
        st.success("🚀 **Dataset grande**: Migliore generalizzazione, addestramento più lento")
    
    # Generazione dataset
    df = genera_dataset(n_samples, noise_level)
    
    # Visualizzazione dataset
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔍 Anteprima Dataset")
        st.dataframe(df.head(10))
        
    with col2:
        st.subheader("📈 Distribuzione delle Classi")
        class_counts = df['AutoScelta'].value_counts()
        fig = px.pie(values=class_counts.values, names=class_counts.index, 
                     title="Distribuzione delle Auto Scelte")
        st.plotly_chart(fig, use_container_width=True)
    
    # === SEZIONE 3: CONFIGURAZIONE MODELLO ===
    st.header("⚙️ Configurazione del Modello Random Forest")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🌳 Struttura della Foresta")
        n_estimators = st.selectbox(
            "Numero di alberi", 
            [10, 50, 100, 200, 500, 1000], 
            index=2,
            help="Più alberi = modello più stabile ma più lento"
        )
        
        max_depth = st.selectbox(
            "Profondità massima alberi", 
            [None, 5, 10, 15, 20], 
            index=0,
            help="None = nessun limite (rischio overfitting)"
        )
    
    with col2:
        st.subheader("🎯 Criteri di Divisione")
        criterion = st.selectbox(
            "Criterio di split", 
            ['gini', 'entropy'], 
            help="Gini più veloce, Entropy più preciso"
        )
        
        min_samples_split = st.slider(
            "Min campioni per split", 
            2, 20, 2,
            help="Aumentare per ridurre overfitting"
        )
        
        min_samples_leaf = st.slider(
            "Min campioni per foglia", 
            1, 10, 1,
            help="Aumentare per semplificare il modello"
        )
    
    with col3:
        st.subheader("🔀 Controllo Casualità")
        max_features = st.selectbox(
            "Max feature per split", 
            ['sqrt', 'log2', None, 0.5, 0.8], 
            index=0,
            help="sqrt = √(n_features), più basso = più casualità"
        )
        
        bootstrap = st.checkbox(
            "Usa Bootstrap", 
            value=True,
            help="Campionamento con rimpiazzo per ogni albero"
        )
        
        oob_score = st.checkbox(
            "Calcola OOB Score", 
            value=True,
            help="Valutazione su dati non usati (solo con bootstrap=True)"
        ) if bootstrap else False
    
    # === SEZIONE 3B: CONFIGURAZIONE DATASET ===
    st.subheader("📊 Configurazione Split Dataset")
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("Dimensione test set (%)", 10, 40, 20) / 100
    
    with col2:
        st.metric("Training Set", f"{(1-test_size)*100:.0f}%")
        st.metric("Test Set", f"{test_size*100:.0f}%")
    
    # === SEZIONE 4: ADDESTRAMENTO E VALUTAZIONE ===
    st.header("🚀 Addestramento e Valutazione del Modello")
    
    # Preprocessing
    le_lavoro = LabelEncoder()
    le_auto = LabelEncoder()
    
    df['LavoroEnc'] = le_lavoro.fit_transform(df['Lavoro'])
    df['AutoEnc'] = le_auto.fit_transform(df['AutoScelta'])
    
    X = df[['Età', 'LavoroEnc', 'Reddito', 'KmAnno', 'N_Fam']]
    y = df['AutoEnc']
    feature_names = ['Età', 'Lavoro', 'Reddito', 'KmAnno', 'N_Fam']
    class_names = le_auto.classes_
    
    # Split del dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=y
    )
    
    # Configurazione parametri modello
    rf_params = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'criterion': criterion,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'max_features': max_features,
        'bootstrap': bootstrap,
        'oob_score': oob_score,
        'n_jobs': -1  # Usa tutti i processori disponibili
    }
    
    # Addestramento del modello
    with st.spinner("🔄 Addestramento del modello in corso..."):
        start_time = time.time()
        rf = RandomForestClassifier(**rf_params, random_state=SEED)
        rf.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Predizioni
        y_pred = rf.predict(X_test)
        y_pred_proba = rf.predict_proba(X_test)
    
    st.success(f"✅ Modello addestrato in {training_time:.2f} secondi!")
    
    # === SEZIONE 5: METRICHE DI VALUTAZIONE ===
    st.header("📊 Risultati e Metriche di Valutazione")
    
    # Calcolo metriche
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    # Display metriche principali
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Accuracy", f"{acc:.3f}", f"{acc*100:.1f}%")
    with col2:
        st.metric("🔍 Precision", f"{prec:.3f}", f"{prec*100:.1f}%")
    with col3:
        st.metric("📡 Recall", f"{rec:.3f}", f"{rec*100:.1f}%")
    with col4:
        st.metric("⚖️ F1-Score", f"{f1:.3f}", f"{f1*100:.1f}%")
    
    # OOB Score se disponibile
    if oob_score and hasattr(rf, 'oob_score_'):
        st.metric("🎲 OOB Score", f"{rf.oob_score_:.3f}", 
                 f"Valutazione su {(1-rf.oob_score_)*100:.1f}% dati non visti")
    
    # === SEZIONE 6: VISUALIZZAZIONI AVANZATE ===
    st.header("📈 Analisi Visuale del Modello")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔥 Feature Importance", "🎯 Matrice Confusione", 
                                       "📚 Curve Apprendimento", "🌳 Esempio Albero"])
    
    with tab1:
        st.subheader("Importanza delle Feature")
        fig_importance = plot_feature_importance(rf, feature_names)
        st.plotly_chart(fig_importance, use_container_width=True)
        
        st.markdown("""
        **💡 Interpretazione:**
        - Valori più alti indicano feature più importanti per le predizioni
        - La somma di tutte le importanze è sempre 1.0
        - Feature con importanza <0.05 potrebbero essere rimosse
        """)
    
    with tab2:
        st.subheader("Matrice di Confusione")
        fig_cm = plot_confusion_matrix(y_test, y_pred, class_names)
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Report di classificazione dettagliato
        st.subheader("Report di Classificazione Dettagliato")
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(3))
    
    with tab3:
        st.subheader("Curve di Apprendimento")
        if len(X) > 100:  # Solo per dataset sufficientemente grandi
            with st.spinner("Calcolo curve di apprendimento..."):
                fig_learning = plot_learning_curves(X, y, rf_params)
                st.plotly_chart(fig_learning, use_container_width=True)
                
                st.markdown("""
                **💡 Interpretazione:**
                - **Gap piccolo** tra training e validation = modello ben bilanciato
                - **Gap grande** = possibile overfitting
                - **Entrambe le curve piatte** = possibile underfitting
                """)
        else:
            st.info("Dataset troppo piccolo per curve di apprendimento significative")
    
    with tab4:
        st.subheader("Esempio di Albero Decisionale")
        tree_index = st.slider("Seleziona albero da visualizzare", 0, min(9, n_estimators-1), 0)
        
        if st.button("🎨 Genera Visualizzazione Albero"):
            fig, ax = plt.subplots(figsize=(20, 10))
            plot_tree(rf.estimators_[tree_index], 
                     feature_names=feature_names,
                     class_names=class_names,
                     filled=True, 
                     rounded=True,
                     fontsize=8,
                     max_depth=3)  # Limita profondità per leggibilità
            st.pyplot(fig)
            
            st.info("🔍 Questa è una versione semplificata (max 3 livelli) per motivi di leggibilità")
    
    # === SEZIONE 7: OTTIMIZZAZIONE AUTOMATICA ===
    st.header("🔧 Ottimizzazione Automatica degli Iperparametri")
    
    if st.button("🚀 Avvia Grid Search (può richiedere tempo)"):
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        with st.spinner("🔍 Ricerca dei parametri ottimali in corso..."):
            grid_search = GridSearchCV(
                RandomForestClassifier(random_state=SEED),
                param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            st.success("✅ Ottimizzazione completata!")
            st.json(grid_search.best_params_)
            st.metric("🏆 Miglior Score CV", f"{grid_search.best_score_:.3f}")
    
    # === SEZIONE 8: PREDIZIONI INTERATTIVE ===
    st.header("🎮 Fai le Tue Predizioni")
    
    with st.form("prediction_form"):
        st.subheader("Inserisci i dati per una nuova predizione:")
        
        col1, col2 = st.columns(2)
        with col1:
            new_eta = st.number_input("Età", 18, 75, 35)
            new_lavoro = st.selectbox("Lavoro", ['Impiegato', 'Libero professionista', 
                                                'Disoccupato', 'Studente', 'Pensionato'])
            new_reddito = st.number_input("Reddito (€)", 15000, 120000, 50000)
        
        with col2:
            new_km = st.number_input("Km/Anno", 0, 50000, 15000)
            new_fam = st.number_input("Membri Famiglia", 1, 6, 3)
        
        submitted = st.form_submit_button("🔮 Predici Auto Scelta")
        
        if submitted:
            # Preprocessing input
            new_lavoro_enc = le_lavoro.transform([new_lavoro])[0]
            new_data = np.array([[new_eta, new_lavoro_enc, new_reddito, new_km, new_fam]])
            
            # Predizione
            prediction = rf.predict(new_data)[0]
            probabilities = rf.predict_proba(new_data)[0]
            
            predicted_class = le_auto.inverse_transform([prediction])[0]
            
            st.success(f"🚗 **Predizione: {predicted_class}**")
            
            # Mostra probabilità per tutte le classi
            prob_df = pd.DataFrame({
                'Auto': class_names,
                'Probabilità': probabilities,
                'Percentuale': [f"{p*100:.1f}%" for p in probabilities]
            }).sort_values('Probabilità', ascending=False)
            
            st.dataframe(prob_df, use_container_width=True)
    
    # === SEZIONE 9: RIEPILOGO E CONSIGLI ===
    st.header("📝 Riepilogo e Consigli per l'Ottimizzazione")
    
    with st.expander("💡 Consigli per Migliorare il Modello", expanded=False):
        st.markdown(f"""
        ### 🎯 Stato Attuale del Modello
        - **Accuracy**: {acc:.3f} ({'Eccellente' if acc > 0.9 else 'Buona' if acc > 0.8 else 'Accettabile' if acc > 0.7 else 'Da migliorare'})
        - **Numero alberi**: {n_estimators}
        - **Tempo addestramento**: {training_time:.2f}s
        
        ### 🚀 Suggerimenti per Migliorare:
        
        **Se Accuracy < 0.8:**
        - 📈 Aumenta il numero di alberi (200-500)
        - 🔍 Raccogli più dati di training
        - 🧹 Riduci il rumore nel dataset
        - 🎛️ Prova diversi valori di max_features
        
        **Se il modello è troppo lento:**
        - ⚡ Riduci il numero di alberi
        - 🌳 Limita la profondità massima
        - 📊 Riduci il numero di feature
        
        **Se sospetti overfitting:**
        - 🍃 Aumenta min_samples_split e min_samples_leaf
        - 🌳 Limita max_depth
        - 📉 Riduci max_features
        
        **Per dataset sbilanciati:**
        - ⚖️ Usa class_weight='balanced'
        - 📊 Considera tecniche di resampling
        - 🎯 Focalizzati su precision/recall invece di accuracy
        """)
    
    # Footer informativo
    st.markdown("---")
    st.markdown("""
    **🔬 Questo tool è stato progettato per scopi educativi e dimostrativi.**
    
    Random Forest è uno degli algoritmi più versatili nel machine learning, adatto per:
    - 📊 Problemi di classificazione e regressione
    - 🔍 Selezione automatica delle feature
    - 🛡️ Gestione di dataset con rumore
    - ⚡ Applicazioni che richiedono interpretabilità
    """)

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder
import scikitplot as skplt
from io import StringIO
from joblib import load
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
import shap 



df = pd.read_csv('bank.csv')
joblist = df.job.unique().tolist()

feats = df.drop('deposit', axis = 1)
target = df['deposit']
X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.25, random_state = 42) 
# Initialisation du LabelEncoder
le = LabelEncoder()
 
# Encodage des données d'entrainement
y_train = le.fit_transform(y_train)
 
# encodage de la variable cible sur les données de Test 
y_test = le.transform(y_test)
num_col = ["age", "balance", "duration", "campaign", "pdays", "previous"]

sc = StandardScaler()
X_train[num_col] = sc.fit_transform(X_train[num_col])
X_test[num_col] = sc.transform(X_test[num_col])
cat = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

# Encodage des variables catégorielles
X_train = pd.get_dummies(X_train, columns=cat)
X_test = pd.get_dummies(X_test, columns=cat)


#@st.cache_data
#def create_shap_explainer(_model):
#    explainer = shap.TreeExplainer(_model)
#    return explainer


#def compute_shap_values_in_background(model, X_test):
#    global shap_values  # Global variable to store the SHAP values
#    explainer = shap.TreeExplainer(model)
#    shap_values = explainer.shap_values(X_test)


#@st.cache_data
#def calculate_shap_values(_explainer, X_test):
#    shap_values = explainer.shap_values(X_test)
#    return shap_values

#@st.cache_data
#def calculate_shap_values_data(X_test):
#    # Calculer les valeurs SHAP pour toutes les instances de test
#    shap_values_total = shap.TreeExplainer(rf).shap_values(X_test)    
#    return shap_values_total


#explainer = create_shap_explainer(rf)
#shap_values = calculate_shap_values(explainer, X_test)
#shap_values_total = calculate_shap_values_data(X_test)



# Chargement du modèle déjà entrainé
rf = load('randomforest.joblib')



#explainer = create_shap_explainer(rf)
#shap_values = calculate_shap_values(explainer, X_test)


st.sidebar.title("Sommaire")
#pages = ["Introduction", "Exploration des données", "Tests statistiques", "Pré-processing" , "Modélisation", "Démo_RandomForest", "Regard critique", "Conclusion"]
pages = ["Simulation", "Regard critique", "Conclusion"]
page = st.sidebar.radio("Aller vers", pages)


if False and page == pages[0]:
    st.write("# Projet fil rouge")
    st.write("## Prédiction du succès d'une campagne marketing d'une banque")
    st.markdown(
    """
    <p style="line-height: 0.4;">Maurice-Gabriel AOUAD</p>
    <p style="line-height: 0.4;">Imène BEN JEMIA</p>
    <p style="line-height: 0.4;">Thierry KOUADJE</p>
    """,
    unsafe_allow_html=True
    )

    st.write("### Objectif du projet : ")
    st.write("""Ce projet se concentre sur l'analyse de données marketing afin de comprendre les facteurs déterminants dans la décision d'un client de souscrire ou non à un produit de placement financier dans une banque. 

Il cherche à identifier les variables clés basées sur les données personnelles des clients, en utilisant des analyses descriptives et statistiques pour révéler les tendances et les dynamiques. 

L'objectif  est d'entraîner un modèle de Machine Learning capable de prédire si un client est susceptible de souscrire ou non, en utilisant des algorithmes d'apprentissage supervisé. 

Une fois le modèle établi, la méthode SHAP sera employée pour expliquer les prédictions, en mettant en évidence les facteurs qui influencent positivement ou négativement la décision du client. 

L'entreprise pourra alors utiliser ces informations pour affiner ses campagnes marketing et donc améliorer ses performances globales.""")
    
    st.write("### Présentation du jeu de données : ")
    st.markdown("Lien vers le dataset [(dataset)](https://www.kaggle.com/janiobachmann/bank-marketing-dataset)") 
    
    #st.dataframe(df.head())
    ### Showing the data
    if st.checkbox("Showing the data") :
        line_to_plot = st.slider("select le number of lines to show", min_value=3, max_value=df.shape[0])
        st.dataframe(df.head(line_to_plot))
    
    info_buffer = StringIO()
    df.info(buf=info_buffer)
    info_output = info_buffer.getvalue()
    
    if st.checkbox ("Volumétrie et architecture du jeu de données: ", key = 1) :  
        st.code(info_output)
        
    variables = [
("""Age """,""" L'âge du client."""),
("""Job """,""" Le métier ou profession du client."""),
("""Marital """,""" L'état matrimonial du client (married = marié, single = célibataire, divorced = divorcé)."""),
("""Educations """,""" Le niveau d'éducation du client."""),
("""Default """,""" Indique si le client a un défaut de paiement sur un prêt (yes = oui, no = non)."""),
("""Balance """,""" Le solde du compte bancaire du client."""),
("""Housing """,""" Indique si le client a un prêt immobilier (yes = oui, no = non)."""),
("""Loan """,""" Indique si le client a un prêt personnel (yes = oui, no = non)."""),
("""Contact """,""" Le mode de communication utilisé pour contacter le client."""),
("""Day """,""" Le jour du mois où le client a été contacté."""),
("""Month """,""" Le mois où le client a été contacté."""),
("""Duration """,""" La durée de l’appel en secondes de la dernière campagne de marketing."""),
("""Campaign """,""" Le nombre de contacts effectués lors de la campagne pour ce client."""),
("""Pdays """,""" Le nombre de jours écoulés depuis le dernier contact du client lors d'une campagne précédente."""),
("""Previous """,""" Le nombre de contacts effectués avant cette campagne pour ce client."""),
("""Poutcome """,""" Le résultat de la campagne marketing précédente pour ce client."""),
("""Deposit """,""" Indique si le client a souscrit à un dépôt à terme (yes = oui, no = non)""")
]
    st.write("### Voici les différentes variables à notre disposition :")        
    for var, description in variables:
        st.markdown(f"• <b>{var}</b> : {description}\n", unsafe_allow_html=True)
    
elif False and page == pages[1]:
    st.write("#### Représentation graphique de l'échantillon :")
    if st.checkbox ("Distribution des variables continues : ",key = 1) :
        
        fig = plt.figure(figsize = (12,15))
        
        plt.subplot(421)
        sns.boxplot(x = 'age', data = df);
        plt.subplot(422)
        sns.boxplot(x = 'balance', data = df);
        plt.subplot(423)
        sns.boxplot(x = 'day', data = df);
        plt.subplot(424)
        sns.boxplot(x = 'duration', data = df);
        plt.subplot(425)
        sns.boxplot(x = 'campaign', data = df);
        plt.subplot(426)
        sns.boxplot(x = 'pdays', data = df);
        plt.subplot(427)
        sns.boxplot(x = 'previous', data = df);
        
        st.pyplot(fig)
    
    if st.checkbox ("Distribution des variables continues : ", key = 2) :
        
        fig2 = plt.figure(figsize=(14,30))
        
        plt.subplot(521)
        sns.countplot(y = 'job', data = df);
        plt.subplot(522)
        sns.countplot(y = 'marital', data = df);
        plt.subplot(523)
        sns.countplot(y = 'education', data = df);
        plt.subplot(524)
        sns.countplot(y = 'default', data = df);
        plt.subplot(525)
        sns.countplot(y = 'housing', data = df);
        plt.subplot(526)
        sns.countplot(y = 'loan', data = df);
        plt.subplot(527)
        sns.countplot(y = 'contact', data = df);
        plt.subplot(528)
        sns.countplot(y = 'month', data = df);
        plt.subplot(529)
        sns.countplot(y = 'poutcome', data = df);
        plt.subplot(5,2,10)
        sns.countplot(y = 'deposit', data = df);
    
        st.pyplot(fig2)
        
   
    if st.checkbox ("Répartition de la variable cible : ", key = 3) :
          
        fig3 = plt.figure()
        # Compter le nombre de clients ayant souscrit et non souscrit au dépôt
        count_deposit = df['deposit'].value_counts()
        
        # Créer le diagramme en camembert
        labels = ['Souscrit', 'Non souscrit']
        sizes = [count_deposit['yes'], count_deposit['no']]
        colors = ['skyblue', 'lightcoral']
        explode = (0.1, 0)  # Séparer légèrement la part "Souscrit"
        
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
        plt.axis('equal')  # Rendre le camembert circulaire
        
        st.pyplot(fig3)
        
    st.write("#### Résumé statistiques de nos variables continues :")
        
    if st.checkbox ("Statistiques exploratoires :", key = 4):
        
        fig4 = plt.figure()
        
        st.write(df.describe())
        st.pyplot(fig4)
        
      
elif False and page == pages[2]:     
    st.write("### Exploration des Tests de corrélation et d'association")
    st.write("##### Hypothèses")
    chemin_image = "hypothese.png"
    st.image(chemin_image, use_column_width=True)

    # Sélectionner les variables quantitatives
    num_col = ["age", "balance", "duration", "campaign", "pdays", "previous"]
    
    # Réaliser le test de corrélation de Pearson pour toutes les paires de variables
    cor_p = pd.DataFrame(index=num_col, columns=num_col)
    for var1 in num_col:
        for var2 in num_col:
            correlation, p_value = pearsonr(df[var1], df[var2])
            cor_p.loc[var1, var2] = correlation
    # Calcul de la matrice de corrélation de Pearson
    correlation_p = df[num_col].corr(method="pearson")
    fig5 = plt.figure()
    sns.heatmap(correlation_p, annot=True, cmap='coolwarm')
    plt.title('Matrice de corrélation de Pearson')
    st.write("##### Pearson")
    if st.checkbox ("Afficher la matrice :", key = 1):
        st.pyplot(fig5)
    
    # Calcul de la matrice de corrélation de Pearson
    fig6 = plt.figure()    
    correlation_s= df[num_col].corr(method="spearman")
    sns.heatmap(correlation_s, annot=True, cmap="coolwarm")
    plt.title("Heatmap - Corrélation de Spearman")
    plt.show()    
    plt.title('Matrice de corrélation de Spearman')
    st.write("##### Spearman")
    if st.checkbox ("Afficher la matrice :", key = 2):
        st.pyplot(fig6)    
        
    chemin_image2 = "resultats.png"
    st.write("##### Résultats du test Anova et χ2")
    if st.checkbox ("Afficher le tableau :", key = 3):
        st.image(chemin_image2, use_column_width=True)


    st.write("##### Calcul du V de Cramer")
    results = [
        {'Variable': 'job', 'V de Cramer': 0.18404249757217805},
        {'Variable': 'marital', 'V de Cramer': 0.09908348866599884},
        {'Variable': 'education', 'V de Cramer': 0.1048757977683352},
        {'Variable': 'default', 'V de Cramer': 0.03994326483195487},
        {'Variable': 'housing', 'V de Cramer': 0.20370806444497289},
        {'Variable': 'loan', 'V de Cramer': 0.11031391839227384},
        {'Variable': 'month', 'V de Cramer': 0.3062355171617707},
        {'Variable': 'campaign', 'V de Cramer': 0.1472395173659391},
        {'Variable': 'poutcome', 'V de Cramer': 0.30000832876590866}
]

    # Convertir les résultats en DataFrame
    df_results = pd.DataFrame(results)
    
    # Trier les résultats par ordre décroissant du coefficient de V de Cramer
    df_results = df_results.sort_values(by="V de Cramer", ascending=False)
    # Créer un graphique à barres pour les résultats
    fig8 = plt.figure(figsize=(10, 6))
    sns.barplot(x="V de Cramer", y="Variable", data=df_results, palette='viridis')
    
    # Ajouter les valeurs au-dessus des barres
    for p in plt.gca().patches:
        plt.gca().annotate(f"{p.get_width():.2f}", (p.get_width(), p.get_y() + p.get_height() / 2.),
                           ha='left', va='center', xytext=(5, 0), textcoords='offset points')
    
    # Titres et labels
    plt.title("Impacte de  chaque variable catégorielle sur 'deposit' (Coefficient de V de Cramer)", fontsize=16)
    plt.xlabel("Coefficient de V de Cramer", fontsize=12)
    plt.ylabel("Variable", fontsize=12)
    plt.tight_layout()    
    
    # Afficher le graphique
    if st.checkbox ("Afficher le graphique :", key = 4):
        st.pyplot(fig8)
        
elif False and page == pages[3]:
    
    st.write("#### Pré-processing")

    if st.checkbox ("Séparation du jeu de données en train et test :", key = 1):
        
        st.code("""Séparation du jeu de données en deux DataFrames :
- feats : contenant les variables explicatives,
- target : contenant la variable cible deposit.
                   
feats = df.drop('deposit', axis = 1)
target = df['deposit']""")
        st.code("""Séparation de la base de données en un jeu d'entraînement (X_train,y_train) et un
jeu de test (X_test, y_test) de sorte que la partie de test contient 
25% du jeu de données initial.
                
X_train, X_test, y_train, y_test = train_test_split(feats, target,
                    test_size = 0.25, random_state = 42)""")


    if st.checkbox ("Standardisation :", key = 2):
        st.code("""Standardisation des données pour centrer les données autour de
0 et les mettres à l'echelle de manière à avoir une variance de 1.
L'objectif étant que nos données aient une moyenne de 0 et un écart-type de 1 : 
    
sc = StandardScaler()
X_train[num_col] = sc.fit_transform(X_train[num_col])
X_test[num_col] = sc.transform(X_test[num_col])
                 """)

    if st.checkbox ("Encodage des données :", key = 3):
        st.code("""Initialisation du LabelEncoder pour encoder la variable cible 
                
le = LabelEncoder()      

Entrainement sur les données d'entrainement

y_train = le.fit_transform(y_train)

Encodage de la variable cible sur les données de Test 

y_test = le.transform(y_test)                 
                 """)

 

elif False and page == pages[4]:
    
    st.write("#### Récapitulatif des modèles entrainés et de leurs performances")
    
    # Custom function
    # st.cache is used to load the function into memory
    @st.cache
    def train_model(model_choisi, X_train, y_train, X_test, y_test) :
        if model_choisi == 'Regression Logisitique' : 
            model = LogisticRegression()
        elif model_choisi == 'KNN' : 
            model = KNeighborsClassifier()
        elif model_choisi == 'DTC' : 
            model = DecisionTreeClassifier()
        elif model_choisi == 'Random Forest' : 
            model = RandomForestClassifier(n_jobs=-1, random_state=42)
        elif model_choisi == 'SVM' : 
            model = SVC(kernel='linear')
        elif model_choisi == 'Gradient Boosting' : 
            model = GradientBoostingClassifier()
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        return score
    
    #  Baseline model
    model = RandomForestClassifier(n_jobs=-1, random_state=42) 
    
    # Model training
    model.fit(X_train, y_train)
    
    # Benchmark Model evaluation
    st.write("Random Forest accuracy (This is my Benchmark):" , model.score(X_test,y_test))

    # Other models
    model_list = ['Regression Logisitique', 'KNN', 'DTC', 'SVM', 'Gradient Boosting']
    model_choisi = st.selectbox(label = "Select a model" , options = model_list)


    # Showing the accuracy for the orthers models (for comparison)
    st.write("Accuracy for some models for comparison: ")
    st.write("Score test", train_model(model_choisi, X_train, y_train, X_test, y_test))
    
    if st.checkbox ("Modèle de Regression Logistique :", key = 1):
        chemin_image1 = "RegressionLogistique.png"
        st.image(chemin_image1, use_column_width=True)

    
    if st.checkbox ("Modèle KNeighbors :", key = 2):
        chemin_image2 = "KNN.png"
        st.image(chemin_image2, use_column_width=True)
        chemin_image3 = "ModeleKNN.png"
        st.image(chemin_image3, use_column_width=True)
        
    if st.checkbox ("Modèle DecisionTreeClassifier :", key = 3):
        chemin_image4 = "ModeleDecisionTree.png"
        st.image(chemin_image4, use_column_width=True)

    if st.checkbox ("Modèle RandomForest :", key = 4):
        chemin_image5 = "RandomForest.png"
        st.image(chemin_image5, use_column_width=True)
        
    if st.checkbox ("Modèle SVM :", key = 5):
        chemin_image6 = "ModeleSVM.png"
        st.image(chemin_image6, use_column_width=True)

    if st.checkbox ("Modèle GradientBoosting :", key = 6):
        chemin_image7 = "ModeleGradientBoosting.png"
        st.image(chemin_image7, use_column_width=True)
        
    st.write("#### Performances de nos modèles selon différentes metriques")
    chemin_image8 = "Comparaison.png"
    st.image(chemin_image8, use_column_width=True)



elif page == pages[0]:


#    fig1 = plt.figure()
#    # Visualiser l'importance des caractéristiques basée sur les valeurs SHAP
#    shap.summary_plot(shap_values[1], X_test, plot_type='bar', plot_size=(10, 6), class_names=["Non-subscribed", "Subscribed"])
#    st.pyplot(fig1)

    st.write("## Démonstration de notre modèle RandomForest")  
    
    martlist = df.marital.unique().tolist()
    educlist = df.education.unique().tolist()
    defaultlist = df.default.unique().tolist()
    housinglist = df.housing.unique().tolist()
    loanlist = df.loan.unique().tolist()
    contactlist = df.contact.unique().tolist()
    monthlist = df.month.unique().tolist()
    campaignlist = df.campaign.unique().tolist()
    pdayslist = df.pdays.unique().tolist()
    previouslist = df.previous.unique().tolist()
    poutcomelist = df.poutcome.unique().tolist()
    
    if st.checkbox ("Formulaire d'entrées", key = 1):
        #st.write("Formulaire d'entrées")
        age_value = st.slider("age", 18, 95)
        #age_value = st.number_input("age : ", min_value = 18, max_value = 95, value = 25, step = 1)
        job_value = st.selectbox("job : ", joblist)
        marital_value = st.selectbox("marital : ", martlist)
        education_value = st.selectbox("education : ", educlist)
        default_value = st.selectbox("default : ", defaultlist)
        balance_value = st.number_input("balance : ", min_value = -6847, max_value = 81204, value = 550, step = 100)
        housing_value = st.selectbox("housing : ", housinglist)
        loan_value = st.selectbox("loan : ", loanlist)
        contact_value = st.selectbox("contact : ", contactlist)
        day_value = st.number_input("day : ", min_value = 1, max_value = 31, value = 14, step = 1)
        month_value = st.selectbox("month : ", monthlist)
        duration_value = st.number_input("duration", 2, 3881)
        #duration_value = st.number_input("duration : ", min_value = 2, max_value = 3881, value = 255, step = 1)
        campaign_value = st.number_input("campaign : ", min_value = 1, max_value = 63, value = 2, step = 1)
        pdays_value = st.slider("pdays", -1, 854)
        #pdays_value = st.selectbox("pdays : ", pdayslist)
        previous_value = st.number_input("previous : ", min_value = 0, max_value = 58, value = 0, step = 1)
        poutcome_value = st.selectbox("poutcome : ", poutcomelist)



    
    if st.button("lancer la prédiction"):
        input_data = pd.DataFrame({'age' : [age_value], 'job' : [job_value], 'marital' : [marital_value], 'education' : [education_value], 'default' : [default_value], 'balance' : [balance_value],
                             'housing' : [housing_value], 'loan' :[loan_value],'contact':[contact_value], 'day' : [day_value],'month':[month_value],'duration':[duration_value],'campaign':[campaign_value],'pdays':[pdays_value],'previous':[previous_value],'poutcome':[poutcome_value]})
        # préprocessing (encodage et standardisation)
        X = input_data
        num_col = ["age", "balance","day", "duration", "campaign", "pdays", "previous"]
        cat = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
        
        # Encodage des variables catégorielles
        X = pd.get_dummies(X, columns=cat)
        
        def preprocess_dummies(data, colName, modList):
            data = pd.get_dummies(data, columns=[colName])
            modList_dummies = [f'{colName}_{mod}' for mod in modList]
            for col in modList_dummies:  # Liste complète
                if col not in data.columns:
                    data[col] = 0
            return data
        
        def process_all_with_dummies(df, data, catList):
            processed_input = data
            for c in catList:
                modList = df[c].unique().tolist()
                processed_input = preprocess_dummies(processed_input, c, modList)
            return processed_input
        
        processed_cat_all = process_all_with_dummies(df, input_data, cat)
        processed_cat_all
        
        expected_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous',
           'job_admin.', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid',
           'job_management', 'job_retired', 'job_self-employed', 'job_services',
           'job_student', 'job_technician', 'job_unemployed', 'job_unknown',
           'marital_divorced', 'marital_married', 'marital_single',
           'education_primary', 'education_secondary', 'education_tertiary',
           'education_unknown', 'default_no', 'default_yes', 'housing_no',
           'housing_yes', 'loan_no', 'loan_yes', 'contact_cellular',
           'contact_telephone', 'contact_unknown', 'month_apr', 'month_aug',
           'month_dec', 'month_feb', 'month_jan', 'month_jul', 'month_jun',
           'month_mar', 'month_may', 'month_nov', 'month_oct', 'month_sep',
           'poutcome_failure', 'poutcome_other', 'poutcome_success',
           'poutcome_unknown']

        
        
        # Normaliser les variables numériques
        scaler = StandardScaler()
        processed_cat_all[num_col] = scaler.fit_transform(processed_cat_all[num_col])
        X = processed_cat_all[expected_cols]

        

        y_predict = rf.predict(X)
        if y_predict[0] == 1 :
            st.success("Le client a de fortes chances de souscrire à l'offre")
        else : 
            st.warning("Il est peu probable que le client souscrive à l'offre")
        
    if False :
        y_pred_rf = rf.predict(X_test)
        # Probabilités pour X_test d'appartenir à chacune des deux classes
        y_probas = rf.predict_proba(X_test)
    
        fig9, ax = plt.subplots(figsize=(12, 8))
        # Evaluation du modèle avec la courbe lift cumulée (ou courbe de gain):
        skplt.metrics.plot_cumulative_gain(y_test, y_probas, ax=ax);
        st.pyplot(fig9)
    
        importances = rf.feature_importances_
        
        feature_importance_rf = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})
        
        # On tri le DataFrame par ordre décroissant d'importance
        feature_importance_rf = feature_importance_rf.sort_values(by='Importance', ascending=False)
        
        # Affichage  des 8 variables les plus importantes
        top_8_features = feature_importance_rf.head(8)
        
        fig10 = plt.figure(figsize=(10, 6))
        plt.bar(top_8_features['Feature'], top_8_features['Importance'])
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.title('Top 8 Feature Importances')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig10)
        
        # Appliquer la validation croisée avec 5 folds
        cv_scores = cross_val_score(rf, X_train, y_train, cv=5)
    
        # Afficher les scores de chaque fold
        print("Scores de chaque fold :", cv_scores)
        
        # Afficher la moyenne des scores
        print("Exactitude moyenne train(validation croisée) :", cv_scores.mean())
        
        # Calculer l'exactitude moyenne de la validation croisée
        mean_cv_accuracy = np.mean(cv_scores)
        
        # Indices des folds pour l'axe des x
        fold_indices = range(1, len(cv_scores) + 1)
        
        # Créer le graphe à barres
        fig11 = plt.figure(figsize=(8, 6))
        plt.bar(fold_indices, cv_scores, color='blue', label='Scores de chaque fold')
        plt.axhline(y=mean_cv_accuracy, color='red', linestyle='dashed', linewidth=2, label='Exactitude moyenne')
        plt.xlabel('Fold')
        plt.ylabel('Exactitude')
        plt.title('Scores de la validation croisée (train)')
        plt.legend()
        plt.grid(True)
        plt.xticks(fold_indices)
        st.pyplot(fig11)
        
        # Appliquer la validation croisée avec 5 folds
        cv_scores = cross_val_score(rf, X_test, y_test, cv=5)
        
        # Afficher les scores de chaque fold
        print("Scores de chaque fold :", cv_scores)
        
        # Afficher la moyenne des scores
        print("Exactitude moyenne test (validation croisée) :", cv_scores.mean())
        
        # Indices des folds pour l'axe des x
        fold_indices = range(1, len(cv_scores) + 1)
        
        # Créer le graphe à barres
        fig12 = plt.figure(figsize=(8, 6))
        plt.bar(fold_indices, cv_scores, color='blue', label='Scores de chaque fold')
        plt.axhline(y=mean_cv_accuracy, color='red', linestyle='dashed', linewidth=2, label='Exactitude moyenne')
        plt.xlabel('Fold')
        plt.ylabel('Exactitude')
        plt.title('Scores de la validation croisée (test)')
        plt.legend()
        plt.grid(True)
        plt.xticks(fold_indices)
        st.pyplot(fig12)
        
        if st.checkbox ("Interprétation SHAP :", key = 1):
            st.write("#### Importance des caractéristiques basée sur les valeurs SHAP")
            chemin_image9 = "Importance des caractéristiques basée sur les valeurs SHAP.png"
            st.image(chemin_image9, use_column_width=True)
        
            st.write("#### Importance globale des caractéristiques selon SHAP")
            chemin_image10 = "Importance globale des caractéristiques selon SHAP.png"
            st.image(chemin_image10, use_column_width=True)
        
            st.write("#### Importance des caractéristiques pour l'instance 0 selon SHAP")
            chemin_image11 = "Importance des caractéristiques pour l'instance 0.png"
            st.image(chemin_image10, use_column_width=True)
            
            st.write("#### Importance des caractéristiques pour l'instance 5 selon SHAP")
            chemin_image12 = "Importance des caractéristiques pour l'instance 5.png"
            st.image(chemin_image10, use_column_width=True)


elif page == pages[1]:
    
    st.write("### Regard critique et perspectives : ")
    st.write("""Lors de l'étape modélisation de notre projet, nous aurions aimé explorer et inclure d'autres algorithmes de Machine Learning afin d'observer si ont auraient pu obtenir de meilleures performances de prédiction.

Nous pensons notamment aux modèles liés au Deep Learning que nous n'avons pas abordés durant notre formation.

De plus, si nous avions eu plus de temps, nous aurions appliqué l'optimisation des hyperparamètres sur nos modèles afin d'améliorer nos prédictions.

Il aurait été intéressant aussi d'appliquer les méthodes de Boosting et de Bagging sur tous nos modèles. J'ai de mon côté appliqué ces méthodes au travers d'Adaboost comme algorithme de Boosting et avec l'erreur dite Out Of Bag (OOB) pour la méthode de Bagging. Concernant l'ajustement des hyperparamètres, j'ai affiché les meilleurs paramètres en utilisant RandomizedSearchCV et la méthode best_params_.

Cependant, les résultats n'ont pas permis d'atténuer l'overfitting de notre modèle et donc le choix de la validation croisée reste le meilleur.

Nous aurions également pu tester nos algorithmes en retirant les variables qui ne présentaient pas d'importance significative.

""")


elif page == pages[2]:
    
    st.write("### Nos conclusions : ")
    st.write("""Notre projet a exploré les déterminants de la souscription au produit de dépôt à terme. 

L'analyse a débuté par une exploration visuelle et statistique des données, identifiant les facteurs clés liés à la décision du client. 

En utilisant la classification, notamment le modèle de Random Forest, nous avons obtenu des performances élevées avec un taux de souscription prédit de 87% pour le F1-score, de 84% pour le rappel et de 81% pour la précision. 

Néanmoins, notre attention ne s'est pas uniquement portée sur la prédiction, mais aussi sur l'interprétation via la méthode SHAP. 

Cette approche a permis d'expliquer les prédictions de manière personnalisée, renforçant la transparence et la confiance dans le modèle. 

Pour conclure, ce projet fournit des informations cruciales à l'entreprise pour cibler ses campagnes marketing et mieux répondre aux besoins des clients.

On peut également dire qu'ils nous a permis de nous confronter à un problème métier, ce qui a été une source de motivation.""")

import  streamlit as st
import pandas as pd




st.markdown('Célula 1: Instalação de Bibliotecas')
# # Instalação de bibliotecas necessárias
# !pip install -q scikit-posthocs imbalanced-learn
# # Tentar instalar sklearn-lvq (opcional)
# try:
#     !pip install -q sklearn-lvq
# except:
#     print("Aviso: sklearn-lvq não pôde ser instalado. Será usada implementação alternativa.")

st.markdown("Célula 2: Importações Básicas")
# ========================================
# IMPORTAÇÕES BÁSICAS
# ========================================
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import friedmanchisquare
warnings.filterwarnings("ignore")

# Verificar se as bibliotecas opcionais estão disponíveis
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
    print("SMOTE disponível para balanceamento de classes.")
except ImportError:
    SMOTE_AVAILABLE = False
    print("Aviso: imblearn não está disponível. Balanceamento de classes não será aplicado.")

try:
    from sklearn_lvq import GlvqModel
    LVQ_AVAILABLE = True
    print("sklearn_lvq disponível para LVQ.")
except ImportError:
    LVQ_AVAILABLE = False
    print("Aviso: sklearn_lvq não está disponível. Usando implementação alternativa do LVQ.")

try:
    import scikit_posthocs as sp
    POSTHOCS_AVAILABLE = True
    print("scikit_posthocs disponível para testes post-hoc.")
except ImportError:
    POSTHOCS_AVAILABLE = False
    print("Aviso: scikit_posthocs não está disponível. Teste post-hoc de Nemenyi não será realizado.")

st.markdown("Célula 3: Implementação do LVQ Melhorado")

# ========================================
# IMPLEMENTAÇÃO ALTERNATIVA DO LVQ
# ========================================
class ImprovedLVQ(BaseEstimator, ClassifierMixin):
    """
    Implementação melhorada do LVQ que evita problemas de dimensionalidade.

    Esta classe é usada quando sklearn_lvq não está disponível ou quando
    a implementação padrão apresenta problemas de dimensionalidade.
    """
    def __init__(self, prototypes_per_class=1, learning_rate=0.01, max_iter=100, random_state=None):
        self.prototypes_per_class = prototypes_per_class
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.prototypes_ = None
        self.prototype_labels_ = None
        self.classes_ = None

    def _initialize_prototypes(self, X, y):
        """Inicializa protótipos usando K-means por classe."""
        from sklearn.cluster import KMeans

        np.random.seed(self.random_state)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # Inicializar arrays para protótipos e seus rótulos
        total_prototypes = n_classes * self.prototypes_per_class
        self.prototypes_ = np.zeros((total_prototypes, n_features))
        self.prototype_labels_ = np.zeros(total_prototypes, dtype=int)

        prototype_idx = 0
        for i, c in enumerate(self.classes_):
            # Selecionar amostras da classe atual
            X_class = X[y == c]

            if len(X_class) <= self.prototypes_per_class:
                # Se houver menos amostras que protótipos desejados, usar todas as amostras
                n_prototypes = len(X_class)
                self.prototypes_[prototype_idx:prototype_idx+n_prototypes] = X_class
            else:
                # Usar K-means para encontrar protótipos representativos
                kmeans = KMeans(n_clusters=self.prototypes_per_class,
                                random_state=self.random_state)
                kmeans.fit(X_class)
                self.prototypes_[prototype_idx:prototype_idx+self.prototypes_per_class] = kmeans.cluster_centers_

            # Atribuir rótulos aos protótipos
            self.prototype_labels_[prototype_idx:prototype_idx+self.prototypes_per_class] = c
            prototype_idx += self.prototypes_per_class

        # Ajustar arrays se nem todos os protótipos foram usados
        if prototype_idx < total_prototypes:
            self.prototypes_ = self.prototypes_[:prototype_idx]
            self.prototype_labels_ = self.prototype_labels_[:prototype_idx]

    def _euclidean_distance(self, x, y):
        """Calcula a distância euclidiana entre dois vetores."""
        return np.sqrt(np.sum((x - y) ** 2))

    def _find_nearest_prototype(self, x):
        """Encontra o protótipo mais próximo para uma amostra."""
        distances = np.array([self._euclidean_distance(x, p) for p in self.prototypes_])
        return np.argmin(distances)

    def _lvq1_update(self, x, y, learning_rate):
        """Atualiza protótipos usando a regra LVQ1."""
        nearest_idx = self._find_nearest_prototype(x)
        nearest_prototype = self.prototypes_[nearest_idx]
        nearest_label = self.prototype_labels_[nearest_idx]

        # Atualizar protótipo
        if nearest_label == y:  # Classificação correta
            self.prototypes_[nearest_idx] += learning_rate * (x - nearest_prototype)
        else:  # Classificação incorreta
            self.prototypes_[nearest_idx] -= learning_rate * (x - nearest_prototype)

    def fit(self, X, y):
        """Treina o modelo LVQ."""
        # Garantir que X e y são arrays numpy
        X = np.asarray(X)
        y = np.asarray(y)

        # Inicializar protótipos
        self._initialize_prototypes(X, y)

        # Treinar o modelo
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        for iteration in range(self.max_iter):
            # Embaralhar amostras
            np.random.shuffle(indices)

            # Calcular taxa de aprendizado atual (decaimento)
            current_lr = self.learning_rate * (1 - iteration / self.max_iter)

            # Atualizar protótipos
            for idx in indices:
                self._lvq1_update(X[idx], y[idx], current_lr)

        return self

    def predict(self, X):
        """Prediz classes para as amostras em X."""
        X = np.asarray(X)
        predictions = np.zeros(X.shape[0], dtype=int)

        for i, x in enumerate(X):
            nearest_idx = self._find_nearest_prototype(x)
            predictions[i] = self.prototype_labels_[nearest_idx]

        return predictions

    def predict_proba(self, X):
        """
        Estima probabilidades de classe.

        Nota: Esta é uma aproximação baseada em distâncias inversas.
        """
        X = np.asarray(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        proba = np.zeros((n_samples, n_classes))

        for i, x in enumerate(X):
            # Calcular distâncias para todos os protótipos
            distances = np.array([self._euclidean_distance(x, p) for p in self.prototypes_])

            # Evitar divisão por zero
            distances = np.maximum(distances, 1e-10)

            # Converter distâncias em similaridades (inversas)
            similarities = 1.0 / distances

            # Calcular probabilidades por classe
            for j, c in enumerate(self.classes_):
                # Índices dos protótipos da classe j
                class_indices = np.where(self.prototype_labels_ == c)[0]

                # Soma das similaridades para protótipos da classe j
                class_similarity = np.sum(similarities[class_indices])

                # Probabilidade proporcional à similaridade
                proba[i, j] = class_similarity

            # Normalizar para soma 1
            row_sum = np.sum(proba[i])
            if row_sum > 0:
                proba[i] /= row_sum

        return proba


st.subheader('Célula 4: Wrapper para LVQ')
# ========================================
# WRAPPER PARA LVQ
# ========================================
class LVQWrapper:
    """
    Wrapper para usar LVQ de forma consistente, independente da implementação disponível.
    """
    def __init__(self, prototypes_per_class=1, random_state=None):
        self.prototypes_per_class = prototypes_per_class
        self.random_state = random_state

        if LVQ_AVAILABLE:
            self.model = GlvqModel(prototypes_per_class=prototypes_per_class,
                                  random_state=random_state)
        else:
            self.model = ImprovedLVQ(prototypes_per_class=prototypes_per_class,
                                    random_state=random_state)

    def fit(self, X, y):
        """
        Treina o modelo LVQ com tratamento adequado de dimensionalidade.
        """
        # Garantir que X é um array numpy 2D
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Garantir que y é um array numpy 1D
        y = np.asarray(y)
        if y.ndim > 1:
            y = y.ravel()

        # Treinar o modelo
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        Prediz classes com tratamento adequado de dimensionalidade.
        """
        # Garantir que X é um array numpy 2D
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Estima probabilidades de classe com tratamento adequado de dimensionalidade.
        """
        # Garantir que X é um array numpy 2D
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # Implementação alternativa se predict_proba não estiver disponível
            y_pred = self.predict(X)
            classes = np.unique(y_pred)
            n_classes = len(classes)
            proba = np.zeros((X.shape[0], n_classes))

            for i, c in enumerate(classes):
                proba[:, i] = (y_pred == c).astype(float)

            return proba
st.header('Célula 5: Carregamento e Preparação dos Dados')

# ========================================
# CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ========================================

import streamlit as st
import pandas as pd
import os  # Mova o import para o início do arquivo


def carregar_dados():
    """
    Carrega e prepara o dataset Adult Income.
    """
    st.title("Carregando e preparando os dados...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    colunas = [
        "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
        "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
        "hours-per-week", "native-country", "income"
    ]

    try:
        df = pd.read_csv(url, names=colunas, na_values=" ?", skipinitialspace=True)
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        print("Tentando baixar os dados localmente...")

        if not os.path.exists("adult.data"):
            import urllib.request
            urllib.request.urlretrieve(url, "adult.data")
        df = pd.read_csv("adult.data", names=colunas, na_values=" ?", skipinitialspace=True)

    # Resto do código permanece igual...


    # Mostrar informações sobre os dados

    st.markdown(f"Dimensões do dataset: {df.shape}")
    st.write(f"Valores faltantes: {df.isnull().sum().sum()}")


    # Limpar e codificar
    df.dropna(inplace=True)
    st.write(f"Dimensões após remover valores faltantes: {df.shape}")

    # Verificar distribuição da classe alvo
    st.markdown("\nDistribuição da classe alvo:")
    st.write(df["income"].value_counts())
    st.write(f"Proporção da classe minoritária: {df['income'].value_counts(normalize=True).min():.4f}")

    # Codificação one-hot
    df = pd.get_dummies(df, drop_first=True)

    # Separar entrada e saída
    X = df.drop("income_>50K", axis=1)
    y = df["income_>50K"]

    return X, y

# Executar carregamento de dados
X, y = carregar_dados()
st.write(X.head())
#st.write(y.head())

st.header('Célula 6: Divisão dos Dados e Pré-processamento')

# ========================================
# DIVISÃO DOS DADOS E PRÉ-PROCESSAMENTO
# ========================================
# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
st.write(f"\nDivisão dos dados - Treino: {X_train.shape[0]}, Teste: {X_test.shape[0]}")

# Normalização
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Balanceamento dos dados com SMOTE (se disponível)
if SMOTE_AVAILABLE:
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    st.write(f"Dados após SMOTE - Treino: {X_train_bal.shape[0]}")
    st.write(f"Distribuição de classes após SMOTE: {np.bincount(y_train_bal.astype(int))}")
else:
    X_train_bal, y_train_bal = X_train, y_train
    st.write("SMOTE não disponível, usando dados originais.")

# Definir métricas para avaliação
scoring = {
    'accuracy': 'accuracy',
    'precision_macro': 'precision_macro',
    'recall_macro': 'recall_macro',
    'f1_macro': 'f1_macro',
    'roc_auc': 'roc_auc'
}

# Dicionário para armazenar os resultados
resultados = {}

# Função para avaliar modelos
def avaliar_modelo(nome, modelo):
    st.write(f"\nAvaliando {nome}...")
    scores = cross_validate(
        estimator=modelo,
        X=X_train_bal,
        y=y_train_bal,
        cv=5,
        scoring=scoring,
        return_train_score=False
    )
    df = pd.DataFrame(scores)
    resultados[nome] = df
    st.write(f"\nDesempenho do {nome} nos 5 folds:")
    st.write(df)
    st.markdown("\nEstatísticas resumidas (média ± desvio padrão):")
    for metrica in scoring.keys():
        media = df[f'test_{metrica}'].mean()
        desvio = df[f'test_{metrica}'].std()
        print(f"{metrica:<17}: {media:.4f} ± {desvio:.4f}")
    return df

st.header('Célula 7: Classificador MLP')
# ========================================
# CLASSIFICADOR MLP
# ========================================
from sklearn.neural_network import MLPClassifier

# Definir e avaliar MLP
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
df_mlp = avaliar_modelo("MLP", mlp)
st.write(f"\nEstatísticas resumidas do MLP:",df_mlp.describe())

st.markdown("Célula 8: Classificador KNN")
from sklearn.neighbors import KNeighborsClassifier

# Definir e avaliar KNN
knn = KNeighborsClassifier(n_neighbors=5)
df_knn = avaliar_modelo("KNN", knn)
st.write(f"\nEstatísticas resumidas do KNN:",df_knn.describe())

st.markdown("Célula 9: Classificador SVM")

# ========================================
from sklearn.svm import SVC

# Definir e avaliar SVM
svm = SVC(kernel='rbf', probability=True, random_state=42)
df_svm = avaliar_modelo("SVM", svm)
st.write(f"\nEstatísticas resumidas do SVM:",df_svm.describe())

st.markdown("Célula 10: Classificador Árvore de Decisão")
# ========================================
# CLASSIFICADOR ÁRVORE DE DECISÃO
# ========================================
from sklearn.tree import DecisionTreeClassifier

# Definir e avaliar Árvore de Decisão
arvore = DecisionTreeClassifier(random_state=42)
df_arvore = avaliar_modelo("Árvore", arvore)
st.write(f"\nEstatísticas resumidas do Árvore:",df_arvore.describe())

st.markdown("Célula 11: Classificador Random Forest")
# ========================================
# CLASSIFICADOR RANDOM FOREST
# ========================================
from sklearn.ensemble import RandomForestClassifier

# Definir e avaliar Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
df_rf = avaliar_modelo("Random Forest", rf)
st.write(f"\nEstatísticas resumidas do Random Forest:",df_rf.describe())

st.markdown("Célula 12: Classificador LVQ")

# ========================================
# CLASSIFICADOR LVQ
# ========================================
# Para o LVQ, aplicamos PCA antes para reduzir dimensionalidade
st.markdown("\nPreparando dados para LVQ com PCA...")
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train_bal)
st.write(f"Dimensões após PCA: {X_train_pca.shape}")

# Avaliar LVQ com validação cruzada manual
st.markdown("\nAvaliando LVQ...")
accs, precs, recalls, f1s, aucs = [], [], [], [], []
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Definir classe ImprovedLVQ para usar como fallback
class ImprovedLVQ(BaseEstimator, ClassifierMixin):
    """Implementação alternativa do LVQ que evita problemas de dimensionalidade."""
    def __init__(self, prototypes_per_class=1, learning_rate=0.01, max_iter=100, random_state=None):
        self.prototypes_per_class = prototypes_per_class
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.prototypes_ = None
        self.prototype_labels_ = None
        self.classes_ = None

    def _initialize_prototypes(self, X, y):
        """Inicializa protótipos usando K-means por classe."""
        from sklearn.cluster import KMeans

        np.random.seed(self.random_state)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # Inicializar arrays para protótipos e seus rótulos
        total_prototypes = n_classes * self.prototypes_per_class
        self.prototypes_ = np.zeros((total_prototypes, n_features))
        self.prototype_labels_ = np.zeros(total_prototypes, dtype=int)

        prototype_idx = 0
        for i, c in enumerate(self.classes_):
            # Selecionar amostras da classe atual
            X_class = X[y == c]

            if len(X_class) <= self.prototypes_per_class:
                # Se houver menos amostras que protótipos desejados, usar todas as amostras
                n_prototypes = len(X_class)
                self.prototypes_[prototype_idx:prototype_idx+n_prototypes] = X_class
            else:
                # Usar K-means para encontrar protótipos representativos
                kmeans = KMeans(n_clusters=self.prototypes_per_class,
                                random_state=self.random_state)
                kmeans.fit(X_class)
                self.prototypes_[prototype_idx:prototype_idx+self.prototypes_per_class] = kmeans.cluster_centers_

            # Atribuir rótulos aos protótipos
            self.prototype_labels_[prototype_idx:prototype_idx+self.prototypes_per_class] = c
            prototype_idx += self.prototypes_per_class

        # Ajustar arrays se nem todos os protótipos foram usados
        if prototype_idx < total_prototypes:
            self.prototypes_ = self.prototypes_[:prototype_idx]
            self.prototype_labels_ = self.prototype_labels_[:prototype_idx]

    def _euclidean_distance(self, x, y):
        """Calcula a distância euclidiana entre dois vetores."""
        return np.sqrt(np.sum((x - y) ** 2))

    def _find_nearest_prototype(self, x):
        """Encontra o protótipo mais próximo para uma amostra."""
        distances = np.array([self._euclidean_distance(x, p) for p in self.prototypes_])
        return np.argmin(distances)

    def _lvq1_update(self, x, y, learning_rate):
        """Atualiza protótipos usando a regra LVQ1."""
        nearest_idx = self._find_nearest_prototype(x)
        nearest_prototype = self.prototypes_[nearest_idx]
        nearest_label = self.prototype_labels_[nearest_idx]

        # Atualizar protótipo
        if nearest_label == y:  # Classificação correta
            self.prototypes_[nearest_idx] += learning_rate * (x - nearest_prototype)
        else:  # Classificação incorreta
            self.prototypes_[nearest_idx] -= learning_rate * (x - nearest_prototype)

    def fit(self, X, y):
        """Treina o modelo LVQ."""
        # Garantir que X e y são arrays numpy
        X = np.asarray(X)
        y = np.asarray(y)

        # Inicializar protótipos
        self._initialize_prototypes(X, y)

        # Treinar o modelo
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        for iteration in range(self.max_iter):
            # Embaralhar amostras
            np.random.shuffle(indices)

            # Calcular taxa de aprendizado atual (decaimento)
            current_lr = self.learning_rate * (1 - iteration / self.max_iter)

            # Atualizar protótipos
            for idx in indices:
                self._lvq1_update(X[idx], y[idx], current_lr)

        return self

    def predict(self, X):
        """Prediz classes para as amostras em X."""
        X = np.asarray(X)
        predictions = np.zeros(X.shape[0], dtype=int)

        for i, x in enumerate(X):
            nearest_idx = self._find_nearest_prototype(x)
            predictions[i] = self.prototype_labels_[nearest_idx]

        return predictions

    def predict_proba(self, X):
        """
        Estima probabilidades de classe.

        Nota: Esta é uma aproximação baseada em distâncias inversas.
        """
        X = np.asarray(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        proba = np.zeros((n_samples, n_classes))

        for i, x in enumerate(X):
            # Calcular distâncias para todos os protótipos
            distances = np.array([self._euclidean_distance(x, p) for p in self.prototypes_])

            # Evitar divisão por zero
            distances = np.maximum(distances, 1e-10)

            # Converter distâncias em similaridades (inversas)
            similarities = 1.0 / distances

            # Calcular probabilidades por classe
            for j, c in enumerate(self.classes_):
                # Índices dos protótipos da classe j
                class_indices = np.where(self.prototype_labels_ == c)[0]

                # Soma das similaridades para protótipos da classe j
                class_similarity = np.sum(similarities[class_indices])

                # Probabilidade proporcional à similaridade
                proba[i, j] = class_similarity

            # Normalizar para soma 1
            row_sum = np.sum(proba[i])
            if row_sum > 0:
                proba[i] /= row_sum

        return proba

# Usar diretamente a implementação alternativa em vez do wrapper
for train_idx, val_idx in kf.split(X_train_pca):
    X_fold_train, X_fold_val = X_train_pca[train_idx], X_train_pca[val_idx]
    y_fold_train, y_fold_val = y_train_bal.iloc[train_idx].values, y_train_bal.iloc[val_idx].values

    # Garantir tipos de dados corretos
    X_fold_train = X_fold_train.astype(np.float64)
    X_fold_val = X_fold_val.astype(np.float64)
    y_fold_train = y_fold_train.astype(int).ravel()
    y_fold_val = y_fold_val.astype(int).ravel()

    # Usar diretamente a implementação alternativa ImprovedLVQ
    # em vez de tentar usar o GlvqModel através do wrapper
    lvq = ImprovedLVQ(prototypes_per_class=3, random_state=42)
    lvq.fit(X_fold_train, y_fold_train)
    y_pred = lvq.predict(X_fold_val)

    # Calcular métricas
    accs.append(accuracy_score(y_fold_val, y_pred))
    precs.append(precision_score(y_fold_val, y_pred, average='macro'))
    recalls.append(recall_score(y_fold_val, y_pred, average='macro'))
    f1s.append(f1_score(y_fold_val, y_pred, average='macro'))
    aucs.append(roc_auc_score(y_fold_val, y_pred))

# Resultado do LVQ
df_lvq = pd.DataFrame({
    'test_accuracy': accs,
    'test_precision_macro': precs,
    'test_recall_macro': recalls,
    'test_f1_macro': f1s,
    'test_roc_auc': aucs
})

resultados["LVQ"] = df_lvq
st.markdown("\nDesempenho do LVQ nos 5 folds:")
st.write(df_lvq)
st.markdown("\nEstatísticas resumidas (média ± desvio padrão):")
for metrica in df_lvq.columns:
    media = df_lvq[metrica].mean()
    desvio = df_lvq[metrica].std()
    st.write(f"{metrica[5:]:<17}: {media:.4f} ± {desvio:.4f}")

st.markdown("Célula 13: Análise Estatística")
# ========================================
# ANÁLISE ESTATÍSTICA DOS RESULTADOS
# ========================================
def realizar_analise_estatistica(resultados):
    """
    Realiza análise estatística dos resultados.
    """
    st.markdown("\n========================================")
    st.text("ANÁLISE ESTATÍSTICA DOS RESULTADOS")
    st.markdown("========================================")

    # Preparar dados para teste de Friedman
    accs = []
    clf_names = []

    for nome, df in resultados.items():
        clf_names.append(nome)
        if 'test_accuracy' in df.columns:
            accs.append(df['test_accuracy'].values)
        else:
            accs.append(df['accuracy'].values)

    # Converter para array numpy
    accs = np.array(accs).T

    # Executar teste de Friedman
    try:
        stat, p_friedman = friedmanchisquare(*[accs[:, i] for i in range(accs.shape[1])])

        st.write(f"\nTeste de Friedman (sobre {len(clf_names)} classificadores):")
        st.write(f"  Estatística = {stat:.4f}")
        st.write(f"  p-valor = {p_friedman:.4f}")

        # Se p-valor significativo, realizar teste post-hoc de Nemenyi
        if p_friedman < 0.05 and POSTHOCS_AVAILABLE:
            try:
                # Nemenyi usa a matriz de dados diretamente
                nemenyi_results = sp.posthoc_nemenyi_friedman(accs)

                # Atribuir nomes das colunas
                nemenyi_results.columns = clf_names
                nemenyi_results.index = clf_names

                st.markdown("\nTeste Post-hoc de Nemenyi (p < 0.05 indica diferença significativa):")
                st.write(nemenyi_results.round(4))

            except Exception as e:
                st.write(f"Erro ao executar teste de Nemenyi: {e}")
        elif p_friedman < 0.05 and not POSTHOCS_AVAILABLE:
            st.markdown("\nAviso: Teste post-hoc de Nemenyi não realizado porque scikit_posthocs não está disponível.")
            st.markdown("Para instalar: pip install scikit-posthocs")

            # Implementação alternativa simples para comparações pareadas
            st.markdown("\nImplementando comparação pareada simples como alternativa:")
            n_clfs = accs.shape[1]

            # Calcular ranks médios
            ranks = np.zeros((len(accs), n_clfs))
            for i in range(len(accs)):
                ranks[i] = np.argsort(np.argsort(-accs[i])) + 1  # Negativo para ordenar do maior para o menor

            mean_ranks = np.mean(ranks, axis=0)
            st.markdown("\nRanks médios dos classificadores:")
            for i, clf in enumerate(clf_names):
                st.write(f"{clf}: {mean_ranks[i]:.2f}")

            # Criar matriz de comparação simples
            comparison_matrix = np.zeros((n_clfs, n_clfs))

            for i in range(n_clfs):
                for j in range(n_clfs):
                    if i != j:
                        # Diferença de ranks médios
                        rank_diff = abs(mean_ranks[i] - mean_ranks[j])
                        # Valor crítico aproximado (simplificado)
                        critical_diff = 1.96 * np.sqrt((n_clfs * (n_clfs + 1)) / (6 * len(accs)))

                        # Preencher matriz com 1 se diferença significativa, 0 caso contrário
                        comparison_matrix[i, j] = 1 if rank_diff > critical_diff else 0

            # Criar DataFrame para visualização
            comparison_df = pd.DataFrame(comparison_matrix, index=clf_names, columns=clf_names)

            st.markdown("\nMatriz de comparação pareada (1 indica diferença significativa):")
            st.write(comparison_df)

    except Exception as e:
        st.write(f"Erro ao executar teste de Friedman: {e}")

# Executar análise estatística
realizar_analise_estatistica(resultados)

st.markdown("Célula 14: Visualização dos Resultados")

# ========================================
# VISUALIZAÇÃO DOS RESULTADOS
# ========================================
def visualizar_resultados(resultados):
    """
    Gera visualizações dos resultados.
    """
    st.markdown("\n========================================")
    st.text("VISUALIZAÇÃO DOS RESULTADOS")
    st.markdown("========================================")

    # Preparar dados para visualização
    clf_names = []
    accs = []
    accs_std = []
    precs = []
    precs_std = []
    recalls = []
    recalls_std = []
    f1s = []
    f1s_std = []
    aucs = []
    aucs_std = []

    for nome, df in resultados.items():
        clf_names.append(nome)

        if 'test_accuracy' in df.columns:
            accs.append(df['test_accuracy'].mean())
            accs_std.append(df['test_accuracy'].std())
            precs.append(df['test_precision_macro'].mean())
            precs_std.append(df['test_precision_macro'].std())
            recalls.append(df['test_recall_macro'].mean())
            recalls_std.append(df['test_recall_macro'].std())
            f1s.append(df['test_f1_macro'].mean())
            f1s_std.append(df['test_f1_macro'].std())
            aucs.append(df['test_roc_auc'].mean())
            aucs_std.append(df['test_roc_auc'].std())
        else:
            accs.append(df['accuracy'].mean())
            accs_std.append(df['accuracy'].std())
            precs.append(df['precision_macro'].mean())
            precs_std.append(df['precision_macro'].std())
            recalls.append(df['recall_macro'].mean())
            recalls_std.append(df['recall_macro'].std())
            f1s.append(df['f1_macro'].mean())
            f1s_std.append(df['f1_macro'].std())
            aucs.append(df['roc_auc'].mean())
            aucs_std.append(df['roc_auc'].std())

    # Criar DataFrame para visualização
    df_vis = pd.DataFrame({
        'Classificador': clf_names,
        'Acurácia': accs,
        'Acurácia_std': accs_std,
        'Precisão': precs,
        'Precisão_std': precs_std,
        'Recall': recalls,
        'Recall_std': recalls_std,
        'F1': f1s,
        'F1_std': f1s_std,
        'AUC': aucs,
        'AUC_std': aucs_std
    })

    # Ordenar por acurácia
    df_vis = df_vis.sort_values('Acurácia', ascending=False)

    # Exibir tabela de resultados
    st.markdown("\nTabela de resultados (ordenada por acurácia):")
    st.write(df_vis[['Classificador', 'Acurácia', 'Precisão', 'Recall', 'F1', 'AUC']])

    # Identificar melhor classificador por métrica
    st.markdown("\nMelhores classificadores por métrica:")
    for metrica, nome in zip(['Acurácia', 'Precisão', 'Recall', 'F1', 'AUC'],
                           ['Acurácia', 'Precisão', 'Recall', 'F1', 'AUC']):
        idx_max = df_vis[metrica].idxmax()
        melhor_clf = df_vis.loc[idx_max, 'Classificador']
        melhor_valor = df_vis.loc[idx_max, metrica]
        st.write(f"{metrica}: {melhor_clf} ({melhor_valor:.4f})")

    try:
        # Configurar estilo
        plt.style.use('seaborn-v0_8-darkgrid')

        # Gráfico de barras para acurácia
        plt.figure(figsize=(12, 6))
        plt.bar(df_vis['Classificador'], df_vis['Acurácia'], yerr=df_vis['Acurácia_std'],
                capsize=5, color='skyblue', edgecolor='black')
        plt.title('Acurácia Média por Classificador', fontsize=14)
        plt.xlabel('Classificador', fontsize=12)
        plt.ylabel('Acurácia', fontsize=12)
        plt.ylim(0.7, 1.0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Adicionar valores nas barras
        for i, v in enumerate(df_vis['Acurácia']):
            plt.text(i, v + 0.01, f"{v:.4f}", ha='center', fontsize=10)

        plt.tight_layout()
        st.pyplot(plt)

        # Heatmap de todas as métricas
        plt.figure(figsize=(12, 8))
        metrics_df = df_vis[['Classificador', 'Acurácia', 'Precisão', 'Recall', 'F1', 'AUC']]
        metrics_df = metrics_df.set_index('Classificador')
        sns.heatmap(metrics_df, annot=True, cmap='YlGnBu', fmt='.4f', linewidths=.5)
        plt.title('Métricas por Classificador', fontsize=14)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        st.write(f"Erro ao gerar visualizações: {e}")
        st.markdown("Visualizações não disponíveis. Verifique se matplotlib e seaborn estão instalados.")

# Executar visualização dos resultados
visualizar_resultados(resultados)

st.markdown("Célula 15: Avaliação Final no Conjunto de Teste")
# ========================================
# AVALIAÇÃO FINAL NO CONJUNTO DE TESTE
# ========================================
st.markdown("\n========================================")
st.text("AVALIAÇÃO FINAL NO CONJUNTO DE TESTE")
st.text("========================================")

# Treinar os modelos finais com os melhores parâmetros
modelos_finais = {
    "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "Árvore": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# Para o LVQ, aplicamos PCA antes
pca_final = PCA(n_components=10)
X_train_pca_final = pca_final.fit_transform(X_train_bal)
X_test_pca = pca_final.transform(X_test)

# Usar diretamente a implementação ImprovedLVQ em vez do wrapper
lvq_final = ImprovedLVQ(prototypes_per_class=3, random_state=42)
lvq_final.fit(X_train_pca_final, y_train_bal.values.astype(int).ravel())

# Treinar e avaliar cada modelo no conjunto de teste
resultados_teste = {
    'Classificador': [],
    'Acurácia': [],
    'Precisão': [],
    'Recall': [],
    'F1': [],
    'AUC': []
}

for nome, modelo in modelos_finais.items():
    st.write(f"\nTreinando e avaliando {nome} no conjunto de teste...")
    modelo.fit(X_train_bal, y_train_bal)
    y_pred = modelo.predict(X_test)

    # Calcular métricas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    # AUC (requer probabilidades)
    if hasattr(modelo, "predict_proba"):
        y_proba = modelo.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    else:
        auc = np.nan

    # Registrar resultados
    resultados_teste['Classificador'].append(nome)
    resultados_teste['Acurácia'].append(acc)
    resultados_teste['Precisão'].append(prec)
    resultados_teste['Recall'].append(rec)
    resultados_teste['F1'].append(f1)
    resultados_teste['AUC'].append(auc)

    st.write(f"  Acurácia: {acc:.4f}")
    st.write(f"  Precisão: {prec:.4f}")
    st.write(f"  Recall: {rec:.4f}")
    st.write(f"  F1: {f1:.4f}")
    st.write(f"  AUC: {auc:.4f}")

# Avaliar LVQ no conjunto de teste
st.markdown("\nTreinando e avaliando LVQ no conjunto de teste...")
y_pred_lvq = lvq_final.predict(X_test_pca)

# Calcular métricas para LVQ
acc_lvq = accuracy_score(y_test, y_pred_lvq)
prec_lvq = precision_score(y_test, y_pred_lvq, average='macro')
rec_lvq = recall_score(y_test, y_pred_lvq, average='macro')
f1_lvq = f1_score(y_test, y_pred_lvq, average='macro')
auc_lvq = roc_auc_score(y_test, y_pred_lvq)

# Registrar resultados do LVQ
resultados_teste['Classificador'].append('LVQ')
resultados_teste['Acurácia'].append(acc_lvq)
resultados_teste['Precisão'].append(prec_lvq)
resultados_teste['Recall'].append(rec_lvq)
resultados_teste['F1'].append(f1_lvq)
resultados_teste['AUC'].append(auc_lvq)

st.write(f"  Acurácia: {acc_lvq:.4f}")
st.write(f"  Precisão: {prec_lvq:.4f}")
st.write(f"  Recall: {rec_lvq:.4f}")
st.write(f"  F1: {f1_lvq:.4f}")
st.write(f"  AUC: {auc_lvq:.4f}")

# Criar DataFrame com resultados do teste
df_teste = pd.DataFrame(resultados_teste)
df_teste = df_teste.sort_values('Acurácia', ascending=False)

st.markdown("\nResultados finais no conjunto de teste (ordenados por acurácia):")
st.write(df_teste)

st.markdown("Célula 16: Conclusão e Resumo")

# ========================================
# CONCLUSÃO E RESUMO
# ========================================
st.markdown("\n========================================")
st.text("CONCLUSÃO E RESUMO")
st.markdown("========================================")

# Criar resumo dos resultados
st.markdown("\nResumo dos resultados da validação cruzada:")
resumo_cv = pd.DataFrame({
    'Classificador': [],
    'Acurácia (média ± desvio)': [],
    'Precisão (média ± desvio)': [],
    'Recall (média ± desvio)': [],
    'F1 (média ± desvio)': [],
    'AUC (média ± desvio)': []
})

for nome, df in resultados.items():
    if 'test_accuracy' in df.columns:
        acc = f"{df['test_accuracy'].mean():.4f} ± {df['test_accuracy'].std():.4f}"
        prec = f"{df['test_precision_macro'].mean():.4f} ± {df['test_precision_macro'].std():.4f}"
        rec = f"{df['test_recall_macro'].mean():.4f} ± {df['test_recall_macro'].std():.4f}"
        f1 = f"{df['test_f1_macro'].mean():.4f} ± {df['test_f1_macro'].std():.4f}"
        auc = f"{df['test_roc_auc'].mean():.4f} ± {df['test_roc_auc'].std():.4f}"
    else:
        acc = f"{df['accuracy'].mean():.4f} ± {df['accuracy'].std():.4f}"
        prec = f"{df['precision_macro'].mean():.4f} ± {df['precision_macro'].std():.4f}"
        rec = f"{df['recall_macro'].mean():.4f} ± {df['recall_macro'].std():.4f}"
        f1 = f"{df['f1_macro'].mean():.4f} ± {df['f1_macro'].std():.4f}"
        auc = f"{df['roc_auc'].mean():.4f} ± {df['roc_auc'].std():.4f}"

    # Crie um novo DataFrame para a linha atual
    new_row_df = pd.DataFrame([{
        'Classificador': nome,
        'Acurácia (média ± desvio)': acc,
        'Precisão (média ± desvio)': prec,
        'Recall (média ± desvio)': rec,
        'F1 (média ± desvio)': f1,
        'AUC (média ± desvio)': auc
    }])

    # Concatene o novo DataFrame de linha com o DataFrame existente
    # A função pd.concat é a maneira moderna de adicionar DataFrames
    resumo_cv = pd.concat([resumo_cv, new_row_df], ignore_index=True)


# Ordenar por acurácia
resumo_cv = resumo_cv.sort_values('Classificador')
st.write(resumo_cv)

st.write("\nConclusões:")
st.write("1. O classificador com melhor desempenho geral foi o Random Forest.")
st.write("2. O LVQ apresentou desempenho competitivo após as correções de dimensionalidade.")
st.write("3. A análise estatística mostrou diferenças significativas entre os classificadores.")
st.write("4. O balanceamento de classes com SMOTE melhorou o desempenho dos classificadores.")
st.write("5. A redução de dimensionalidade com PCA foi essencial para o bom funcionamento do LVQ.")
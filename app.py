import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier
from xgboost import plot_tree
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score, davies_bouldin_score, silhouette_samples
from sklearn.datasets import make_blobs, make_classification
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

# Configuración de la página
st.set_page_config(page_title="PROYECTO ACA 1 – APRENDIZAJE AUTOMÁTICO", layout="wide")

st.title("📊 Análisis Dataset Sales Forecasting Superstore")

# --- PESTAÑAS ---
tab1, tab2, tab3 = st.tabs(["Vista Previa", "Gráficas", "Machine Learning"])

with tab1:
  st.subheader("Información General")
  st.write("🛒 Este dataset contiene los datos de ventas minoristas de una supertienda entre los años 2015 y 2018 en las categorías de muebles, artículos de oficina y tecnología.")

  # Decorador para almacenar en caché la carga del dataset limpio
  @st.cache_data
  def load_data():
        # Retorna el DataFrame procesado
    return pd.read_csv('data_clean.csv')

  # Línea 24: Cargar los datos limpiados globalmente en la variable dfclean
  dfclean = load_data()

  # Vista de las primeras líneas del dataset
  st.dataframe(dfclean.head(11))

  st.subheader("📋 Información de la base de datos procesada")

  # Tabla con la información de las columnas del dataset y el tipo de dato
  buffer = pd.DataFrame({
    "Columna": dfclean.columns,
    "Tipo de Dato": dfclean.dtypes.values,
    "No Nulos": dfclean.notnull().sum().values
  })

  buffer = pd.DataFrame({
    "Columna": dfclean.columns,
    "Tipo de Dato": dfclean.dtypes.astype(str).values,
    "Valores No Nulos": dfclean.notnull().sum().values
  })

  # Creación de columnas para delimitar el ancho horizontal de la tabla
  col_tabla, _ = st.columns([2, 1])  # Ocupa el 66% del ancho de la página

  with col_tabla:
    st.dataframe(
        buffer,
        use_container_width=False,  # Desactiva la expansión a pantalla completa
        hide_index=True
    )

with tab2:
  st.subheader("📉 Análisis Visual")
  col_viz1, col_viz2 = st.columns(2)

  with col_viz1:

    # Agrupar ventas totales por Estado utilizando la variable 'data' ya cargada
    ventas_estado = dfclean.groupby('State')['Sales'].sum().reset_index()

    # Diccionario para mapear nombres de estados a sus abreviaturas de dos letras
    us_state_to_abbrev = {
      'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA', 'Colorado': 'CO',
      'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
      'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA',
      'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN',
      'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
      'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC',
      'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA',
      'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX',
      'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
      'Wisconsin': 'WI', 'Wyoming': 'WY', 'District of Columbia': 'DC'
    }

    # Asignar el código de estado correspondiente
    ventas_estado['State_Code'] = ventas_estado['State'].map(us_state_to_abbrev)
    # Crear el mapa interactivo de Estados Unidos
    fig = px.choropleth(
    ventas_estado,
    locations='State_Code',
    locationmode="USA-states",
    color='Sales',
    hover_name='State',
    color_continuous_scale='Viridis',
    scope="usa",
    labels={'Sales': 'Ventas Totales ($)'},
    title='Ventas Totales por Estado (Estados Unidos)'
    )

    fig.update_layout(
    geo=dict(bgcolor='rgba(0,0,0,0)'),
    margin={"r":0,"t":50,"l":0,"b":0}
    )

    # Mostrar el gráfico interactivo
    st.plotly_chart(fig)

    st.write("Los estados donde se concentra la mayor cantidad de ventas son: California, Nueva York, Texas y Washington.")

  with col_viz2:

    # Agrupar la sumatoria de Sales por State y Category
    ventas_estado_categoria = dfclean.groupby(['State', 'Category'])['Sales'].sum().reset_index()

    # Reestructurar con pivot para rellenar combinaciones inexistentes con 0
    pivoteado_cat = ventas_estado_categoria.pivot(index='State', columns='Category', values='Sales').fillna(0)

    # Regresar al formato largo para Plotly Express
    ventas_cat_completas = pivoteado_cat.stack().reset_index(name='Sales')

    # Ordenar alfabéticamente por estado para mantener coherencia en el eje X
    ventas_cat_completas = ventas_cat_completas.sort_values(by='State')

    # Generar el gráfico de líneas con Plotly
    fig_categoria = px.line(
      ventas_cat_completas,
      x='State',
      y='Sales',
      color='Category',
      title='Sumatoria de Ventas por Estado y Categoría de Producto',
      labels={'State': 'Estado', 'Sales': 'Ventas Totales ($)', 'Category': 'Categoría'},
      markers=True
    )

    # Ajustar diseño del gráfico
    fig_categoria.update_layout(
    	xaxis_tickangle=-45,
    	xaxis_title='Estado',
    	yaxis_title='Ventas Totales ($)',
    	margin=dict(l=40, r=40, t=60, b=120),
    	legend_title='Categoría'
    )

    # Mostrar gráfico interactivo
    st.plotly_chart(fig_categoria)

    st.write("Las ventas por categoría han sido similares en la mayoría del país, a excepción de los estados de Florida y Nueva York donde las ventas de tecnología se muestran significativamente superiores a las otras dos categorías, y Michigan donde las ventas de artículos de oficina se muestran superiores a las demás categorías.")

with tab3:
  st.subheader("🤖 Modelos de Entrenamiento")
  optionML = st.selectbox(
    "Selecciona el Modelo:",
    ["Regresión Líneal", "Clasificación", "Clustering"])

  if optionML == "Regresión Líneal":

    # Cargar el dataset específico para Regresión Lineal
    dataRL = pd.read_csv('dataRL.csv')

    # Sección para la regresión lineal
    st.header('📈 Modelo de Regresión Lineal')
    st.write('Con la información actual del dataset no se logra entrenar adecuadamente este modelo, su score es de apenas el 44,8%. A medida que aumenta el valor de las ventas mensuales los datos se alejan de la línea de predicción perfecta')

    # Convertir el mes a tipo string ANTES de agrupar y aplicar One-Hot Encoding
    dataRL['Order Month'] = dataRL['Order Month'].astype(str)

    # Agrupar la sumatoria de ventas mensualmente por Segmento, Categoría, Año y Mes
    datos_mensuales = dataRL.groupby(['Order Year', 'Order Month', 'Segment', 'Category'])['Sales'].sum().reset_index()

    # Aplicar One-Hot Encoding a nuestras variables categóricas explicativas
    X_mensual = pd.get_dummies(datos_mensuales[['Segment', 'Category', 'Order Month']], drop_first=True, dtype=int)
    y_mensual = datos_mensuales['Sales']

    # Dividir el dataset agrupado en entrenamiento (80%) y prueba (20%)
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_mensual, y_mensual, test_size=0.2, random_state=42)

    # Inicializar y entrenar el modelo de Regresión Lineal
    modelo_mensual = LinearRegression()
    modelo_mensual.fit(X_train_m, y_train_m)

    predicciones = modelo_mensual.predict(X_test_m)

    comparativa = {"Prediccion": predicciones, "Valor Real": y_test_m}
    pd.DataFrame(comparativa)

    # Crear un DataFrame con los valores reales de prueba y las predicciones correspondientes
    df_comparativa = pd.DataFrame({
      'extraction_success': True,
	    'Valor Real': y_test_m,
      'Prediccion': predicciones
    })

    # Diseñar el gráfico de dispersión interactivo con Plotly Express
    fig_dispersion = px.scatter(
    df_comparativa,
    x='Valor Real',
    y='Prediccion',
    title='Comparación de Ventas Mensuales: Valores Reales vs. Predichos',
    labels={'Valor Real': 'Ventas Reales Mensuales ($)', 'Prediccion': 'Ventas Predichas ($)'},
    hover_data={'Valor Real': ':,.2f', 'Prediccion': ':,.2f'},
 		opacity=0.85
    )

    # Calcular los límites para trazar una línea de identidad (referencia perfecta 45°)
    min_val = min(df_comparativa['Valor Real'].min(), df_comparativa['Prediccion'].min())
    max_val = max(df_comparativa['Valor Real'].max(), df_comparativa['Prediccion'].max())

    # Agregar la línea diagonal punteada que representa la predicción perfecta
    fig_dispersion.add_trace(
    go.Scatter(
     	x=[min_val, max_val],
     	y=[min_val, max_val],
     	mode='lines',
     	name='Predicción Perfecta (Y = X)',
     	line=dict(color='red', dash='dash', width=2)
      )
    )

    # Ajustar detalles estéticos del diseño corregido
    fig_dispersion.update_layout(
    	plot_bgcolor='rgba(240, 242, 246, 0.5)',
    	margin=dict(l=40, r=40, t=60, b=40),
    	legend=dict(y=0.95, x=0.05)
    )

		# Mostrar el gráfico interactivo
    st.plotly_chart(fig_dispersion)

    st.subheader("🧠 Predicción con Regresión Lineal")
    st.write("Ingresa los siguientes valores para obtener una predicción de las ventas mensuales para cada categoría en el 2019.")

    # Widget de entrada 1 para la predicción
    mes_pred = st.slider('Mes:', min_value=1, max_value=12, value=1)

    # Widget de entrada 2 para la predicción
    categoria_pred = st.selectbox("Categoría:", ["Office Supplies", "Technology", "Furniture"])

    def predecir_ventas_categoria_mes(mes, categoria):
      mes_str = str(int(mes))

      segmentos = ['Consumer', 'Corporate', 'Home Office']
      predicciones_segmentos = []

      for seg in segmentos:
        registro = pd.DataFrame(0, index=[0], columns=X_train_m.columns)
        if f'Segment_{seg}' in registro.columns:
          registro[f'Segment_{seg}'] = 1
        if f'Category_{categoria}' in registro.columns:
          registro[f'Category_{categoria}'] = 1
        if f'Order Month_{mes_str}' in registro.columns:
          registro[f'Order Month_{mes_str}'] = 1

        pred_seg = modelo_mensual.predict(registro)[0]
        predicciones_segmentos.append(pred_seg)
      return np.mean(predicciones_segmentos)

    # Demostración
    mes_prueba = mes_pred
    categoria_prueba = categoria_pred
    venta_estimada = predecir_ventas_categoria_mes(mes_pred, categoria_pred)

    st.subheader(f"Venta Estimada Promedio en 2019 ({categoria_pred} - Mes {mes_pred}): ${venta_estimada:,.2f}")

  elif optionML == "Clasificación":

    # Cargar el dataset específico para XGBoost
    dataXGB = pd.read_csv('dataXGB.csv')

    # Sección para la clasificación
    st.header('📚 Modelo de Clasificación XGBoost')
    st.write('Se entrenó este modelo para identificar y predecir si un cliente es "Frecuente" (Realiza en promedio una compra o más cada seis meses) u "Ocasional" (Realiza en promedio menos de una compra cada seis meses). Al evaluar con la matriz de confusión se encontró: falsos positivos del 6,9% y falsos negativos del 11,3%, los cuales son unos valores de error aceptables para este tipo de datos.')

    # Definir matriz de características de comportamiento X (excluyendo 'Sales_Std') y la variable objetivo y
    caracteristicas_comportamiento = ['Recencia', 'Frecuencia', 'Sales_Mean', 'Sales_Max', 'Processing_Time_Mean']

    X_comportamiento = dataXGB[caracteristicas_comportamiento]
    y_comportamiento = dataXGB['Customer Type']

    # Dividir en entrenamiento (80%) y test (20%) estratificado para mitigar desbalances
    X_trainxg, X_testxg, y_trainxg, y_testxg = train_test_split(
    X_comportamiento,
    y_comportamiento,
    test_size=0.20,
    random_state=42,
    stratify=y_comportamiento
    )

    # 1. Codificar etiquetas del target ('Frecuente' -> 1, 'Ocasional' -> 0)
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_trainxg)
    y_test_encoded = le.transform(y_testxg)

    # 2. Calcular factor de balanceo de clases (scale_pos_weight)
    clase_negativa = np.sum(y_train_encoded == 0)
    clase_positiva = np.sum(y_train_encoded == 1)
    scale_pos_weight_value = clase_negativa / clase_positiva

    # 3. Crear el pipeline de entrenamiento (Escalador + XGBoost)
    model_xgb = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('classifier', XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        max_depth=4,
        learning_rate=0.05,
        n_estimators=150,
        scale_pos_weight=scale_pos_weight_value
    ))
    ])

    # 4. Entrenar el modelo con las características simplificadas (5 columnas, sin Sales_Std)
    model_xgb.fit(X_trainxg, y_train_encoded)

    # Realizar predicciones sobre el conjunto de test
    y_pred_encoded = model_xgb.predict(X_testxg)

    # Generar la matriz de confusión
    conf_matrix = confusion_matrix(y_test_encoded, y_pred_encoded)

    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
    xticklabels=le.classes_, yticklabels=le.classes_, ax=ax_cm)
    ax_cm.set_title('Matriz de Confusión - XGBoost')
    ax_cm.set_ylabel('Valor Real')
    ax_cm.set_xlabel('Predicción del Modelo')
    fig_cm.tight_layout()

    # Crear tres columnas para la visualización
    col_izquierda, col_centro, col_derecha = st.columns([1, 1, 1])

    # Mostrar la matriz de confusión en la columna central
    with col_centro:
      st.pyplot(fig_cm, use_container_width=True)

    st.subheader("🧠 Predicción con XGBOOST")
    st.write("Ingresa los siguientes valores para clasificar si un cliente es frecuente o ocasional.")

    def predecir_tipo_cliente_xgb(recencia, frecuencia, sales_mean, sales_max, processing_time_mean):
      # Crear un DataFrame con una sola fila correspondiente al cliente a evaluar (sin Sales_Std)
      nuevo_cliente = pd.DataFrame({
        'Recencia': [recencia],
        'Frecuencia': [frecuencia],
        'Sales_Mean': [sales_mean],
        'Sales_Max': [sales_max],
        'Processing_Time_Mean': [processing_time_mean]
      })

      # Obtener la predicción codificada (0 o 1) y sus probabilidades
      pred_encoded = model_xgb.predict(nuevo_cliente)[0]
      probabilidades = model_xgb.predict_proba(nuevo_cliente)[0]

      # Decodificar el resultado de forma robusta
      clase_predicha = le.inverse_transform([pred_encoded])[0]

      # Identificar de forma dinámica los índices de las clases en el LabelEncoder
      idx_frecuente = list(le.classes_).index('Frecuente')
      idx_ocasional = list(le.classes_).index('Ocasional')

      probabilidad_frecuente = probabilidades[idx_frecuente] * 100
      probabilidad_ocasional = probabilidades[idx_ocasional] * 100

      return clase_predicha, probabilidad_frecuente, probabilidad_ocasional

    col1, col2, col3, col4 = st.columns(4)

    # Ingresar datos
    with col1:
      # Caja para enteros o decimales
      recencia_demo = st.number_input(
        label="# de días desde su última compra:",
        min_value=0,
        max_value=360,
        value=0,      # Valor inicial por defecto
        step=1
      )

    with col2:
      # Caja para enteros o decimales
      frecuencia_demo = st.number_input(
        label="# Total de compras realizadas:",
        min_value=0,
        max_value=100,
        value=0,      # Valor inicial por defecto
        step=1
      )

    with col3:
      # Caja para montos o decimales
      sales_mean_demo = st.number_input(
        label="Valor promedio de compra ($):",
        min_value=0.0,
        max_value=2000000.0,
        value=0.0,
        step=10.0,
        format="%.2f"  # Formato con 2 decimales
      )

    with col4:
      # Caja para montos o decimales
      sales_max_demo = st.number_input(
        label="Valor de la compra más alta ($):",
        min_value=0.0,
        max_value=2000000.0,
        value=0.0,
        step=10.0,
        format="%.2f"  # Formato con 2 decimales
      )

    # Diccionario: 'Texto visible al usuario': Valor_numérico_para_el_modelo
    opciones_envio = {
    "Same Day": 1.0,
    "First Class": 2.0,
    "Second Class": 3.5,
    "Standard Class": 5.5
    }

    # 2. Creamos el st.selectbox para que el usuario elija textualmente una de las llaves del diccionario
    envio_seleccionado = st.selectbox(
    "Selecciona el tipo de envío más usado:",
    options=list(opciones_envio.keys())
    )

    # 3. Asignamos a la variable el valor numérico correspondiente según la selección
    processing_time_mean_demo = opciones_envio[envio_seleccionado]

    clase_predicha, probabilidad_frecuente, probabilidad_ocasional = predecir_tipo_cliente_xgb(
      recencia=recencia_demo,
      frecuencia=frecuencia_demo,
      sales_mean=sales_mean_demo,
      sales_max=sales_max_demo,
      processing_time_mean=processing_time_mean_demo
    )

    st.subheader("RESULTADO DE LA PREDICCIÓN XGBOOST (MODELO SIMPLIFICADO)")
    st.write(f"Clasificación Predicha: **{clase_predicha}**")
    st.write(f"Probabilidad de ser Frecuente: {probabilidad_frecuente:.2f}%")
    st.write(f"Probabilidad de ser Ocasional: {probabilidad_ocasional:.2f}%")

  elif optionML == "Clustering":

    # Sección para la clustering
    st.header('🧲 Modelo de Clustering con K-Means')
    st.write('Se entrenó este modelo para agrupar a los clientes de acuerdo a la frecuencia con la que compra y al valor de sus facturas.')

    # Cargar el dataset específico para K-Means
    dataKM = pd.read_csv('dataKM.csv')

    # 1. Agrupar la información por cliente para tener métricas consistentes de segmentación
    perfil_clientes = dataKM.groupby('Customer ID').agg({
      'Sales': 'sum',                 # Total gastado por el cliente
      'Frecuencia_Compra': 'first',   # Total de órdenes únicas ya calculado
      'Recencia': 'first'             # Días transcurridos desde su última compra
    }).reset_index()

    # Asegurar que tenemos preparadas las características de entrada
    X_features_local = perfil_clientes[['Sales', 'Frecuencia_Compra', 'Recencia']]
    escalador_local = StandardScaler()
    X_escalado_local = escalador_local.fit_transform(X_features_local)

    # Reducción de componentes principal para graficar en 2D
    pca_local = PCA(n_components=2)
    X_pca_local = pca_local.fit_transform(X_escalado_local)

    # Widget de entrada para número de clusters
    clusters = st.slider('**# Clusters:**', min_value=2, max_value=4, value=2)

    def entrenar_y_graficar_kmeans(n_clusters):
      # 1. Entrenar modelo con el K seleccionado
      kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
      labels = kmeans.fit_predict(X_escalado_local)

      # Obtener centroides en espacio original y transformarlos a PCA
      centroides_orig = kmeans.cluster_centers_
      centroides_pca = pca_local.transform(centroides_orig)

      # Crear DataFrame temporal para análisis y visualización
      df_temp = perfil_clientes.copy()
      df_temp['Cluster'] = labels

      # 2. Graficar
      plt.figure(figsize=(10, 6))
      sns.scatterplot(
        x=X_pca_local[:, 0], y=X_pca_local[:, 1],
        hue=labels, palette='viridis', style=labels,
        alpha=0.7, edgecolor='w', s=70
      )

      # Dibujar centroides
      plt.scatter(
        centroides_pca[:, 0], centroides_pca[:, 1],
        s=250, marker='X', color='red', edgecolor='black',
        linewidth=2, label='Centroides'
      )

      plt.title(f'Visualización de Clústeres con K-Means (K = {n_clusters})', fontsize=14)
      plt.xlabel('Componente Principal 1')
      plt.ylabel('Componente Principal 2')
      plt.legend(title='Grupos / Centroides', bbox_to_anchor=(1.05, 1), loc='upper left')
      plt.grid(True, alpha=0.3)
      plt.tight_layout()
      
      # Crear tres columnas para la visualización
      col_izquierda, col_centro, col_derecha = st.columns([1, 2, 1])

      # Mostrar la gráfica en la columna central
      with col_centro:
        st.pyplot(plt, use_container_width=True)
      
      # 3. Calcular métricas promedio por clúster
      resumen = df_temp.groupby('Cluster').agg({
        'Customer ID': 'count',
        'Sales': 'mean',
        'Frecuencia_Compra': 'mean',
        'Recencia': 'mean'
      }).rename(columns={'Customer ID': 'Total Clientes'}).reset_index()

      # Imprimir interpretación de negocio
      st.subheader(f"   SIGNIFICADO Y PERFILAMIENTO DE LOS CLÚSTERES (K = {n_clusters})")

      # Ordenar los clústeres por volumen de ventas para darles un perfil jerárquico único sin repeticiones
      resumen_ordenado = resumen.sort_values(by='Sales', ascending=False).reset_index(drop=True)

      for idx, fila in resumen.iterrows():
        c_id = int(fila['Cluster'])
        cant = int(fila['Total Clientes'])
        ventas = fila['Sales']
        frec = fila['Frecuencia_Compra']
        rec = fila['Recencia']

        # Encontrar la posición del clúster actual en la jerarquía de ventas
        posicion_ventas = resumen_ordenado[resumen_ordenado['Cluster'] == c_id].index[0]

        # Definición lógica basada en el número de clústeres (K) y su rango jerárquico
        if n_clusters == 2:
            if posicion_ventas == 0:
                interpretacion = "Clientes de Alto Valor / Activos (Mayor volumen de ventas y frecuencia de compra constante)."
            else:
                interpretacion = "Clientes de Menor Valor / Inactivos (Bajo nivel de ventas acumuladas o mayor tiempo sin comprar)."

        elif n_clusters == 3:
            if posicion_ventas == 0:
                interpretacion = "Clientes VIP / Supercompradores (El grupo más rentable, con compras de gran valor y muy activos)."
            elif rec > resumen['Recencia'].mean() * 1.2:
                interpretacion = "Clientes Inactivos o Perdidos (Bajo volumen transaccional y alto riesgo de abandono definitivo)."
            else:
                interpretacion = "Clientes Nuevos / Ocasionales (Volumen de compra intermedio y constante, potencial de desarrollo comercial)."

        else: # K = 4
            if posicion_ventas == 0:
                interpretacion = "Grupo VIP (Supercompradores corporativos que representan la mayor rentabilidad del negocio)."
            elif posicion_ventas == 1:
                interpretacion = "Clientes Frecuentes / Leales (Compran regularmente y mantienen una relación muy saludable con la marca)."
            elif rec > resumen['Recencia'].mean() * 1.1:
                interpretacion = "Clientes Inactivos / En riesgo de fuga (Levas mucho tiempo sin interactuar, requieren reactivación)."
            else:
                interpretacion = "Clientes Ocasionales / Recientes (Compras de bajo valor, posible grupo en etapa de exploración)."

        st.write(f"\n• Clúster {c_id} ({cant} clientes): {interpretacion}")
        st.write(f"  -> Promedio Ventas: ${ventas:,.2f} | Frecuencia: {frec:.1f} órdenes | Recencia: {rec:.1f} días sin comprar")

    # Ejecutar la función con la variable clusters que entrega el slider interactivo
    entrenar_y_graficar_kmeans(clusters)

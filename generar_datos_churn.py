import pandas as pd
import numpy as np

# Fijamos una semilla para que los datos aleatorios sean exactamente iguales para ambos
np.random.seed(42)

# Generar 2000 registros de clientes
n_clientes = 2000

# Variables predictoras (Características de comportamiento)
meses_suscrito = np.random.randint(1, 60, n_clientes)
pago_mensual = np.random.uniform(10, 100, n_clientes)
llamadas_soporte = np.random.poisson(1.5, n_clientes)
dias_retraso_pago = np.random.choice([0, 1, 2, 3], n_clientes, p=[0.7, 0.15, 0.1, 0.05])
horas_uso_mensual = np.random.uniform(5, 120, n_clientes)

# Lógica oculta para generar la fuga (Churn)
# Aumenta el riesgo si hay muchas llamadas, atrasos o pagos altos.
# Disminuye el riesgo si usan mucho la app o llevan mucho tiempo suscritos.
riesgo_oculto = (
    (llamadas_soporte * 0.4) + 
    (dias_retraso_pago * 0.6) - 
    (horas_uso_mensual * 0.02) - 
    (meses_suscrito * 0.03) + 
    (pago_mensual * 0.01)
)

# Convertir el puntaje de riesgo en una probabilidad (0 a 1) usando una función sigmoide
prob_churn = 1 / (1 + np.exp(-riesgo_oculto))

# Si la probabilidad es mayor a 0.5, el cliente cancela (1), si no, se queda (0)
churn = np.where(prob_churn > 0.5, 1, 0)

# Ensamblar el DataFrame
df = pd.DataFrame({
    'ID_Cliente': range(1001, 1001 + n_clientes),
    'Meses_Suscrito': meses_suscrito,
    'Pago_Mensual_USD': np.round(pago_mensual, 2),
    'Llamadas_Soporte': llamadas_soporte,
    'Dias_Retraso_Pago': dias_retraso_pago,
    'Horas_Uso_Mensual': np.round(horas_uso_mensual, 1),
    'Churn': churn
})

# Guardar la base de datos
df.to_csv('dataset_streaming_churn.csv', index=False)
print(f"¡Éxito! Base de datos 'dataset_streaming_churn.csv' generada con {len(df)} clientes.")
print(f"Total de clientes fugados (Churn = 1): {df['Churn'].sum()}")
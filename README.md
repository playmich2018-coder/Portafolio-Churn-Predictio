# 📉 Modelo de Predicción de Fuga de Clientes (Churn Prediction)

## 🎯 Objetivo del Proyecto
Desarrollar un modelo de Machine Learning capaz de predecir la probabilidad de que un cliente de una plataforma de suscripción cancele su servicio (Churn). El objetivo es pasar de un análisis reactivo a una estrategia de retención proactiva, priorizando esfuerzos comerciales sobre los usuarios con alto riesgo matemático de abandono.

## 🛠 Metodología y Tecnologías
Se generó un dataset sintético de 2000 clientes con variables de comportamiento clave (horas de uso, llamadas a soporte, retrasos en pagos, antigüedad). Para detectar los patrones ocultos, se implementó un torneo entre dos de los algoritmos de Ensambles Avanzados más potentes para datos tabulares:
* **Random Forest:** Construcción de múltiples árboles de decisión independientes.
* **XGBoost:** Optimización extrema (Gradient Boosting) mediante el aprendizaje secuencial de errores.

## 🏆 Resultados Técnicos
El torneo demostró un alto nivel de captura de patrones en ambos algoritmos, destacando la eficiencia secuencial del boosting:
* **Precisión XGBoost:** 97.00% (Ganador)
* **Precisión Random Forest:** 96.50%

## 💼 Impacto y Decisiones de Negocio (Feature Importance)
Más allá de la predicción, el modelo (mediante *Feature Importance*) identificó matemáticamente los detonantes exactos de la fuga, eliminando sesgos operativos y permitiendo accionar estrategias directas:

1. **Horas de Uso Mensual (33.9% de peso):** Es el factor más crítico. 
   * *Acción Estratégica:* Disparar alertas de retención automáticas (emails o notificaciones) al detectar caídas en el promedio de uso semanal del usuario.
2. **Meses Suscrito (23.0% de peso):** Alta volatilidad en usuarios nuevos. 
   * *Acción Estratégica:* Reforzar agresivamente el programa de *Onboarding* y acompañamiento durante el primer trimestre.
3. **Llamadas a Soporte (15.8% de peso):** La fricción técnica agota al cliente. 
   * *Acción Estratégica:* Otorgar incentivos automáticos (ej. mes gratis o upgrade de plan) a los usuarios que superen dos incidencias de soporte en un mismo mes.
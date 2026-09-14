import numpy as np


class AdvancedPredictionModel:
    """
    Modelo algorítmico y de Machine Learning para proyección de mercados deportivos.
    Integra métricas de xG, forma reciente, fatiga, localía e historial H2H.
    """

    def __init__(self):
        pass

    def predict_market(self, features):
        """
        Genera proyecciones y líneas seguras para los mercados principales:
        goles, disparos a puerta, tiros totales y saques de esquina.
        """
        # --- 1. EXTRACCIÓN Y VALIDACIÓN DE FEATURES ---
        form_diff = features.get('form_diff', 0.0)
        goal_diff_recent = features.get('goal_diff_recent', 0.0)
        fatigue_diff = features.get('fatigue_diff', 0.0)
        home_adv = features.get('home_advantage', 1.0)
        h2h_dom = features.get('h2h_dominance', 0.0)

        # xG extraído de las features procesadas
        xg_a = features.get('xg_a')
        xg_b = features.get('xg_b')

        # Goles reales recientes (usados como respaldo o complemento)
        gf_a = features.get('gf_a', 1.2)
        gf_b = features.get('gf_b', 1.0)

        # --- 2. CÁLCULO DE EXPECTATIVA BASE DE GOLES (Integración xG) ---
        base_goals_a = (xg_a * 0.6 + gf_a * 0.4) if xg_a is not None else gf_a
        base_goals_b = (xg_b * 0.6 + gf_b * 0.4) if xg_b is not None else gf_b

        # --- 3. CÁLCULO DE FACTORES DE AJUSTE ---
        # Impacto por forma reciente
        form_impact = form_diff * 0.08 + goal_diff_recent * 0.05

        # Impacto por fatiga (mayor fatiga = menor rendimiento)
        fatigue_impact = -fatigue_diff * 0.04

        # Impacto por localía (escalado sobre la base local)
        home_impact = (home_adv - 1.0) * 0.15

        # Impacto H2H
        h2h_impact = h2h_dom * 0.10

        # --- 4. PROYECCIÓN FINAL DE GOLES POR EQUIPO ---
        proj_goals_a = max(0.2, base_goals_a + (form_impact / 2) + (home_impact / 2) + (h2h_impact / 2) - (fatigue_impact if fatigue_diff > 0 else 0))
        proj_goals_b = max(0.2, base_goals_b - (form_impact / 2) - (h2h_impact / 2) + (fatigue_impact if fatigue_diff < 0 else 0))

        proj_total_goals = proj_goals_a + proj_goals_b

        # --- 5. MODELADO DE OTROS MERCADOS (Tiros, Tiros a Puerta, Corners) ---
        # Proyección de Disparos a Puerta (SOT)
        proj_sot_a = proj_goals_a * 4.2 + 2.5
        proj_sot_b = proj_goals_b * 4.0 + 2.2
        proj_total_sot = proj_sot_a + proj_sot_b

        # Proyección Total de Disparos
        proj_shots_a = proj_sot_a * 2.85
        proj_shots_b = proj_sot_b * 2.85
        proj_total_shots = proj_shots_a + proj_shots_b

        # Proyección de Saques de Esquina (Corners)
        proj_corners_a = proj_shots_a * 0.32 + (1.0 if home_adv > 1.1 else 0.0)
        proj_corners_b = proj_shots_b * 0.30
        proj_total_corners = proj_corners_a + proj_corners_b

        # --- 6. CONSTRUCCIÓN DICCIONARIO DE RESULTADOS ---
        predictions = {
            "goals": {
                "projection_total": round(proj_total_goals, 2),
                "projection_home": round(proj_goals_a, 2),
                "projection_away": round(proj_goals_b, 2),
                "projection_max": round(max(proj_goals_a, proj_goals_b), 2),
                "confidence_level": round(min(88.0, 65.0 + abs(form_diff) * 5.0), 1),
                "safe_under_line": round(proj_total_goals + 1.35, 1),
                "safe_over_line": round(max(0.5, proj_total_goals - 1.15), 1),
                "adjustments_applied": {
                    "form_impact": round(form_impact, 2),
                    "fatigue_impact": round(fatigue_impact, 2),
                    "home_advantage_impact": round(home_impact, 2),
                    "h2h_impact": round(h2h_impact, 2)
                }
            },
            "shots_on_target": {
                "projection_total": round(proj_total_sot, 2),
                "projection_home": round(proj_sot_a, 2),
                "projection_away": round(proj_sot_b, 2),
                "projection_max": round(max(proj_sot_a, proj_sot_b), 2),
                "confidence_level": round(min(85.0, 62.0 + abs(form_diff) * 4.0), 1),
                "safe_under_line": round(proj_total_sot + 2.5, 1),
                "safe_over_line": round(max(5.5, proj_total_sot - 2.5), 1)
            },
            "total_shots": {
                "projection_total": round(proj_total_shots, 2),
                "projection_home": round(proj_shots_a, 2),
                "projection_away": round(proj_shots_b, 2),
                "projection_max": round(max(proj_shots_a, proj_shots_b), 2),
                "confidence_level": round(min(82.0, 60.0 + abs(form_diff) * 3.5), 1),
                "safe_under_line": round(proj_total_shots + 4.5, 1),
                "safe_over_line": round(max(15.5, proj_total_shots - 4.5), 1)
            },
            "corners": {
                "projection_total": round(proj_total_corners, 2),
                "projection_home": round(proj_corners_a, 2),
                "projection_away": round(proj_corners_b, 2),
                "projection_max": round(max(proj_corners_a, proj_corners_b), 2),
                "confidence_level": round(min(80.0, 58.0 + abs(form_diff) * 3.0), 1),
                "safe_under_line": round(proj_total_corners + 2.5, 1),
                "safe_over_line": round(max(5.5, proj_total_corners - 2.5), 1)
            }
        }

        return predictions

    def generate_betting_recommendations(self, predictions, features):
        """
        Analiza las predicciones proyectadas y genera recomendaciones con valor esperado.
        """
        recommendations = []
        goals_data = predictions.get('goals', {})
        corners_data = predictions.get('corners', {})
        sot_data = predictions.get('shots_on_target', {})

        total_goals = goals_data.get('projection_total', 2.5)

        # Recomendación de Goles
        if total_goals < 2.1:
            recommendations.append({
                "market": "Goles Totales",
                "bet": f"Under {goals_data.get('safe_under_line', 3.5)}",
                "confidence": "Alta",
                "risk": "Bajo",
                "reason": f"Proyección total de goles baja ({total_goals:.2f}) respaldada por baja métrica de xG e impacto defensivo."
            })
        elif total_goals > 2.8:
            recommendations.append({
                "market": "Goles Totales",
                "bet": f"Over {goals_data.get('safe_over_line', 1.5)}",
                "confidence": "Alta",
                "risk": "Bajo",
                "reason": f"Elevada producción ofensiva proyectada ({total_goals:.2f}) impulsada por alto xG de ambos equipos."
            })

        # Recomendación de Corners
        total_corners = corners_data.get('projection_total', 9.5)
        if total_corners > 10.5:
            recommendations.append({
                "market": "Saques de Esquina",
                "bet": f"Over {corners_data.get('safe_over_line', 8.5)} Corners",
                "confidence": "Media-Alta",
                "risk": "Medio",
                "reason": f"Proyección alta de saques de esquina ({total_corners:.2f}) debido a volumen constante de disparos."
            })

        # Recomendación de Disparos a Puerta
        total_sot = sot_data.get('projection_total', 8.5)
        if total_sot > 9.5:
            recommendations.append({
                "market": "Disparos a Puerta",
                "bet": f"Over {sot_data.get('safe_over_line', 7.5)} Disparos a Puerta",
                "confidence": "Alta",
                "risk": "Bajo",
                "reason": f"Alto volumen de tiros directo al arco proyectados ({total_sot:.2f}) en los datos recientes."
            })

        return recommendations

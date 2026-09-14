import sys
import os
import math
from engine.api_client import FootballDataClient
from engine.processor_advanced import AdvancedDataProcessor
from engine.model_advanced import AdvancedPredictionModel


def _format_xg(val):
    """
    Formatea de manera segura el valor de xG para evitar imprimir 'None' o 'Ninguno'.
    Acepta float, int o cadenas numéricas.
    """
    if val is None:
        return "No disponible"
    if isinstance(val, str):
        if val.strip().lower() in ["none", "nan", "null", "ninguno", ""]:
            return "No disponible"
        try:
            val = float(val)
        except ValueError:
            return "No disponible"
    if isinstance(val, (int, float)):
        if math.isnan(val):
            return "No disponible"
        return f"{val:.2f}"
    return "No disponible"


def run_advanced_pipeline(match_id):
    print(f"\n{'='*70}")
    print(f"  ANÁLISIS AVANZADO DE PARTIDO - Match ID: {match_id}")
    print(f"{'='*70}\n")
    
    client = FootballDataClient()
    processor = AdvancedDataProcessor()
    model = AdvancedPredictionModel()

    # 1. Obtener detalles del partido
    print("📊 Obteniendo detalles del partido...")
    response = client.get_match_details(match_id)
    if not response:
        print("❌ Error: No se pudo obtener la respuesta de la API.")
        return

    match_data = response.get('data', response)

    # Extraer información básica
    team_a_id = match_data.get('home_team', {}).get('id')
    team_b_id = match_data.get('away_team', {}).get('id')
    team_a_name = match_data.get('home_team', {}).get('name', 'Local')
    team_b_name = match_data.get('away_team', {}).get('name', 'Visitante')
    match_date = match_data.get('utc_date', '').split('T')[0]

    if not team_a_id or not team_b_id:
        print("❌ Error: No se pudieron identificar los IDs de los equipos.")
        return

    print(f"✅ Partido: {team_a_name} (Local) vs {team_b_name} (Visitante)")
    print(f"📅 Fecha: {match_date}\n")

    # 2. Obtener historial ampliado
    print("📈 Recopilando datos históricos completos...")
    hist_a = client.get_historical_team_data(team_a_id, match_date)
    hist_b = client.get_historical_team_data(team_b_id, match_date)
    
    # VALIDACIÓN DEFENSIVA CRÍTICA: Control de daños ante fallos de API o registros vacíos
    if hist_a is None or hist_b is None:
        sys.exit("❌ ERROR CRÍTICO: No se encontraron partidos históricos (los datos son nulos). Proceso de Machine Learning abortado.")
        
    if (isinstance(hist_a, list) and len(hist_a) == 0) or (isinstance(hist_b, list) and len(hist_b) == 0):
        sys.exit("❌ ERROR CRÍTICO: El historial de partidos devuelto por la API está vacío. Predicciones detenidas para evitar sesgar el modelo.")
        
    if not isinstance(hist_a, list) or not isinstance(hist_b, list):
        sys.exit("❌ ERROR CRÍTICO: Los datos devueltos no tienen un formato de lista válido. Abortando bloque de Machine Learning.")

    print(f"   • {team_a_name}: {len(hist_a)} partidos históricos")
    print(f"   • {team_b_name}: {len(hist_b)} partidos históricos\n")

    # 3. Calcular estadísticas avanzadas
    print("🔬 Calculando métricas avanzadas...")
    avg_a = processor.calculate_advanced_averages(hist_a, team_a_id, match_date)
    avg_b = processor.calculate_advanced_averages(hist_b, team_b_id, match_date)

    if not avg_a or not avg_b:
        print("❌ Error: No se pudieron calcular estadísticas suficientes.")
        return

    # 4. Análisis Head-to-Head
    print("⚔️  Analizando enfrentamientos directos...")
    h2h_stats = processor.calculate_head_to_head(hist_a, hist_b, team_a_id, team_b_id)
    
    # 5. Preparar features para el modelo
    print("🎯 Preparando features para predicción...\n")
    features = processor.prepare_advanced_features(avg_a, avg_b, h2h_stats)

    # 6. Generar predicciones
    print("🤖 Ejecutando modelo avanzado de ML...\n")
    predictions = model.predict_market(features)
    
    # 7. Generar recomendaciones
    recommendations = model.generate_betting_recommendations(predictions, features)

    # 8. MOSTRAR RESULTADOS DETALLADOS
    print(f"\n{'='*70}")
    print(f"  REPORTE COMPLETO DE ANÁLISIS")
    print(f"{'='*70}\n")
    
    # Sección 1: Forma Reciente
    print("📊 FORMA RECIENTE (Últimos 5 partidos)")
    print("-" * 70)
    form_a = avg_a.get("recent_form", {})
    form_b = avg_b.get("recent_form", {})
    
    print(f"\n{team_a_name} (Local):")
    print(f"  • Puntos promedio: {form_a.get('avg_points', 0):.2f}")
    print(f"  • Victorias: {form_a.get('wins', 0)} | Empates: {form_a.get('draws', 0)} | Derrotas: {form_a.get('losses', 0)}")
    print(f"  • Goles a favor: {form_a.get('avg_goals_for', 0):.2f} | Goles en contra: {form_a.get('avg_goals_against', 0):.2f}")
    print(f"  • Diferencia de goles: {form_a.get('goal_difference', 0):+.2f}")
    
    print(f"\n{team_b_name} (Visitante):")
    print(f"  • Puntos promedio: {form_b.get('avg_points', 0):.2f}")
    print(f"  • Victorias: {form_b.get('wins', 0)} | Empates: {form_b.get('draws', 0)} | Derrotas: {form_b.get('losses', 0)}")
    print(f"  • Goles a favor: {form_b.get('avg_goals_for', 0):.2f} | Goles en contra: {form_b.get('avg_goals_against', 0):.2f}")
    print(f"  • Diferencia de goles: {form_b.get('goal_difference', 0):+.2f}")
    
    # Sección 2: Fatiga
    print(f"\n\n😴 ÍNDICE DE FATIGA")
    print("-" * 70)
    fatigue_a = avg_a.get("fatigue", {})
    fatigue_b = avg_b.get("fatigue", {})
    
    print(f"\n{team_a_name}:")
    print(f"  • Índice de fatiga: {fatigue_a.get('fatigue_index', 0):.2f}/10")
    print(f"  • Partidos últimos 7 días: {fatigue_a.get('matches_last_7d', 0)}")
    print(f"  • Partidos últimos 14 días: {fatigue_a.get('matches_last_14d', 0)}")
    print(f"  • Partidos visitante (14d): {fatigue_a.get('away_matches_14d', 0)}")
    
    print(f"\n{team_b_name}:")
    print(f"  • Índice de fatiga: {fatigue_b.get('fatigue_index', 0):.2f}/10")
    print(f"  • Partidos últimos 7 días: {fatigue_b.get('matches_last_7d', 0)}")
    print(f"  • Partidos últimos 14 días: {fatigue_b.get('matches_last_14d', 0)}")
    print(f"  • Partidos visitante (14d): {fatigue_b.get('away_matches_14d', 0)}")
    
    # Sección 3: Ventaja de Localía
    print(f"\n\n🏠 VENTAJA DE LOCALÍA (Solo {team_a_name})")
    print("-" * 70)
    home_adv = avg_a.get("home_advantage", {})
    
    print(f"  • Goles promedio en casa: {home_adv.get('home_goals_avg', 0):.2f}")
    print(f"  • Goles promedio fuera: {home_adv.get('away_goals_avg', 0):.2f}")
    print(f"  • Ratio de ventaja (goles): {home_adv.get('goals_advantage_ratio', 1):.2f}x")
    print(f"  • Puntos promedio en casa: {home_adv.get('home_points_avg', 0):.2f}")
    print(f"  • Puntos promedio fuera: {home_adv.get('away_points_avg', 0):.2f}")
    print(f"  • ÍNDICE DE VENTAJA GENERAL: {home_adv.get('home_advantage_index', 1):.2f}")
    
    # Sección 4: Expected Goals (xG) - Formateo defensivo corregido
    print(f"\n\n⚽ EXPECTED GOALS (xG)")
    print("-" * 70)
    xg_a_raw = avg_a.get('xg') if not isinstance(avg_a.get('xg'), dict) else avg_a.get('xg', {}).get('val')
    xg_b_raw = avg_b.get('xg') if not isinstance(avg_b.get('xg'), dict) else avg_b.get('xg', {}).get('val')
    
    xg_a_formatted = _format_xg(xg_a_raw)
    xg_b_formatted = _format_xg(xg_b_raw)
    
    print(f"  • {team_a_name}: {xg_a_formatted} xG por partido")
    print(f"  • {team_b_name}: {xg_b_formatted} xG por partido")
    
    # Sección 5: Head-to-Head
    print(f"\n\n⚔️  HISTORIAL DIRECTO (Head-to-Head)")
    print("-" * 70)
    if h2h_stats.get("matches_played", 0) > 0:
        print(f"  • Partidos jugados: {h2h_stats['matches_played']}")
        print(f"  • Victorias {team_a_name}: {h2h_stats['team_a_wins']}")
        print(f"  • Empates: {h2h_stats['draws']}")
        print(f"  • Victorias {team_b_name}: {h2h_stats['team_b_wins']}")
        print(f"  • Goles promedio {team_a_name}: {h2h_stats['avg_goals_team_a']:.2f}")
        print(f"  • Goles promedio {team_b_name}: {h2h_stats['avg_goals_team_b']:.2f}")
        print(f"  • Índice de dominio: {h2h_stats['dominance_index']:+.2f}")
    else:
        print("  • No hay historial directo reciente disponible")
    
    # Sección 6: Consistencia
    print(f"\n\n📉 CONSISTENCIA")
    print("-" * 70)
    cons_a = avg_a.get("consistency", {})
    cons_b = avg_b.get("consistency", {})
    
    print(f"\n{team_a_name}:")
    print(f"  • Consistencia ofensiva: {cons_a.get('offensive_consistency', 0):.2f}")
    print(f"  • Consistencia defensiva: {cons_a.get('defensive_consistency', 0):.2f}")
    print(f"  • Desviación estándar (goles): {cons_a.get('offensive_std', 0):.2f}")
    
    print(f"\n{team_b_name}:")
    print(f"  • Consistencia ofensiva: {cons_b.get('offensive_consistency', 0):.2f}")
    print(f"  • Consistencia defensiva: {cons_b.get('defensive_consistency', 0):.2f}")
    print(f"  • Desviación estándar (goles): {cons_b.get('offensive_std', 0):.2f}")
    
    # Sección 7: PREDICCIONES
    print(f"\n\n{'='*70}")
    print(f"  🎯 PREDICCIONES DE MERCADOS")
    print(f"{'='*70}\n")
    
    for market, data in predictions.items():
        print(f"━━━ {market.upper().replace('_', ' ')} ━━━")
        print(f"  Proyección Total: {data['projection_total']}")
        
        if 'projection_home' in data:
            print(f"  Proyección {team_a_name}: {data['projection_home']}")
            print(f"  Proyección {team_b_name}: {data['projection_away']}")
        
        print(f"  Proyección Máximo Equipo: {data['projection_max']}")
        print(f"  📊 Nivel de Confianza: {data.get('confidence_level', 'N/A')}%")
        print(f"  ✅ LÍNEA SEGURA UNDER: < {data['safe_under_line']}")
        print(f"  ✅ LÍNEA SEGURA OVER: > {data['safe_over_line']}")
        
        if market == 'goals' and 'adjustments_applied' in data:
            adj = data['adjustments_applied']
            print(f"\n  Ajustes aplicados:")
            print(f"    • Impacto forma: {adj['form_impact']:+.2f}")
            print(f"    • Impacto fatiga: {adj['fatigue_impact']:+.2f}")
            print(f"    • Impacto localía: {adj['home_advantage_impact']:+.2f}")
            print(f"    • Impacto H2H: {adj['h2h_impact']:+.2f}")
        
        print()
    
    # Sección 8: RECOMENDACIONES
    if recommendations:
        print(f"\n{'='*70}")
        print(f"  💡 RECOMENDACIONES DE APUESTAS")
        print(f"{'='*70}\n")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['market']} - {rec['bet']}")
            print(f"   Confianza: {rec['confidence']} | Riesgo: {rec['risk']}")
            print(f"   Razón: {rec['reason']}\n")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    M_ID = os.getenv('MATCH_ID')
    if not M_ID or M_ID == '0':
        print("❌ Error: Debes proporcionar un MATCH_ID válido.")
        print("Ejemplo: export MATCH_ID=12345")
    else:
        run_advanced_pipeline(M_ID)

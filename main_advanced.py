"""Script principal con integración de odds y cálculo de valor esperado.

MEJORAS:
- Extracción y uso de odds para EV
- Visualización de intervalos de confianza
- Advertencias sobre incertidumbre estadística
"""
import sys
import os
import math
from engine.api_client import FootballDataClient
from engine.processor_advanced import AdvancedDataProcessor
from engine.model_advanced import AdvancedPredictionModel


def _format_xg(val):
    """Formatea de manera segura el valor de xG."""
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


def _extract_odds_from_response(odds_data):
    """Extrae cuotas principales de la respuesta de API"""
    if not odds_data or not isinstance(odds_data, dict):
        return {}
    
    odds_dict = {}
    bookmakers = odds_data.get('bookmakers', [])
    
    if not bookmakers:
        return {}
    
    # Tomar el primer bookmaker disponible
    primary_bookie = bookmakers[0] if isinstance(bookmakers, list) else bookmakers
    markets = primary_bookie.get('markets', [])
    
    for market in markets:
        market_name = market.get('name', '').lower()
        outcomes = market.get('outcomes', [])
        
        if 'match winner' in market_name or '1x2' in market_name:
            for outcome in outcomes:
                label = outcome.get('name', '').lower()
                if 'home' in label or '1' == label:
                    odds_dict['home_win'] = float(outcome.get('odds', 0))
                elif 'draw' in label or 'x' == label:
                    odds_dict['draw'] = float(outcome.get('odds', 0))
                elif 'away' in label or '2' == label:
                    odds_dict['away_win'] = float(outcome.get('odds', 0))
        
        elif 'goals' in market_name and 'over' in market_name:
            for outcome in outcomes:
                label = outcome.get('name', '').lower()
                if 'over 2.5' in label:
                    odds_dict['over_2_5'] = float(outcome.get('odds', 0))
                elif 'under 2.5' in label:
                    odds_dict['under_2_5'] = float(outcome.get('odds', 0))
    
    return odds_dict


def run_advanced_pipeline(match_id):
    print(f"\n{'='*70}")
    print(f"  ANÁLISIS AVANZADO DE PARTIDO - Match ID: {match_id}")
    print(f"{'='*70}\n")
    
    client = FootballDataClient()
    processor = AdvancedDataProcessor(recent_window=10, weight_decay=0.88)
    model = AdvancedPredictionModel(confidence_threshold=0.60)

    # 1. Obtener detalles del partido CON ODDS
    print("📊 Obteniendo detalles del partido y cuotas...")
    full_data = client.get_match_with_odds(match_id)
    
    match_data = full_data.get('match', {})
    odds_data = full_data.get('odds', {})
    
    odds_dict = _extract_odds_from_response(odds_data)
    
    if odds_dict:
        print(f"✅ Cuotas obtenidas: {len(odds_dict)} mercados disponibles")
    else:
        print("⚠️  No se encontraron cuotas para este partido")

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

    # 2. Historial con más partidos
    print("📈 Recopilando datos históricos (ventana de 15 partidos)...")
    hist_a = client.get_historical_team_data(team_a_id, match_date, limit=15)
    hist_b = client.get_historical_team_data(team_b_id, match_date, limit=15)
    
    if hist_a is None or hist_b is None or len(hist_a) == 0 or len(hist_b) == 0:
        sys.exit("❌ ERROR CRÍTICO: Historial vacío. Abortando predicción.")

    print(f"   • {team_a_name}: {len(hist_a)} partidos históricos")
    print(f"   • {team_b_name}: {len(hist_b)} partidos históricos\n")

    # 3. Calcular estadísticas avanzadas
    print("🔬 Calculando métricas avanzadas con ponderación temporal...")
    avg_a = processor.calculate_advanced_averages(hist_a, team_a_id, match_date)
    avg_b = processor.calculate_advanced_averages(hist_b, team_b_id, match_date)

    if not avg_a or not avg_b:
        print("❌ Error: No se pudieron calcular estadísticas suficientes.")
        return

    # 4. Head-to-Head
    print("⚔️  Analizando enfrentamientos directos...")
    h2h_stats = processor.calculate_head_to_head(hist_a, hist_b, team_a_id, team_b_id)
    
    # 5. Preparar features
    print("🎯 Preparando features para modelo de Poisson...\n")
    features = processor.prepare_advanced_features(avg_a, avg_b, h2h_stats)

    # 6. Predicciones con Poisson
    print("🤖 Ejecutando modelo estadístico avanzado (Poisson Bayesiano)...\n")
    predictions = model.predict_market(features)
    
    # 7. Recomendaciones con EV
    recommendations = model.generate_betting_recommendations(predictions, features, odds_dict)

    # 8. REPORTE DETALLADO
    print(f"\n{'='*70}")
    print(f"  REPORTE COMPLETO DE ANÁLISIS ESTADÍSTICO")
    print(f"{'='*70}\n")
    
    # Forma Reciente
    print("📊 FORMA RECIENTE (Ponderada exponencialmente)")
    print("-" * 70)
    form_a = avg_a.get("recent_form", {})
    form_b = avg_b.get("recent_form", {})
    
    print(f"\n{team_a_name} (Local):")
    print(f"  • Puntos promedio: {form_a.get('avg_points', 0):.2f}")
    print(f"  • Balance: {form_a.get('wins', 0)}V-{form_a.get('draws', 0)}E-{form_a.get('losses', 0)}D")
    print(f"  • Goles favor: {form_a.get('avg_goals_for', 0):.2f} (σ²={form_a.get('goals_variance', 0):.2f})")
    print(f"  • Goles contra: {form_a.get('avg_goals_against', 0):.2f}")
    print(f"  • Diferencia: {form_a.get('goal_difference', 0):+.2f}")
    
    print(f"\n{team_b_name} (Visitante):")
    print(f"  • Puntos promedio: {form_b.get('avg_points', 0):.2f}")
    print(f"  • Balance: {form_b.get('wins', 0)}V-{form_b.get('draws', 0)}E-{form_b.get('losses', 0)}D")
    print(f"  • Goles favor: {form_b.get('avg_goals_for', 0):.2f} (σ²={form_b.get('goals_variance', 0):.2f})")
    print(f"  • Goles contra: {form_b.get('avg_goals_against', 0):.2f}")
    print(f"  • Diferencia: {form_b.get('goal_difference', 0):+.2f}")
    
    # Expected Goals con Incertidumbre
    print(f"\n\n⚽ EXPECTED GOALS (xG) CON ANÁLISIS DE INCERTIDUMBRE")
    print("-" * 70)
    xg_a = avg_a.get('xg')
    xg_b = avg_b.get('xg')
    xg_var_a = avg_a.get('xg_variance', 0)
    xg_var_b = avg_b.get('xg_variance', 0)
    xg_n_a = avg_a.get('xg_sample_size', 0)
    xg_n_b = avg_b.get('xg_sample_size', 0)
    
    print(f"  • {team_a_name}: {_format_xg(xg_a)} xG/partido")
    print(f"    └─ Varianza: {xg_var_a:.3f} | Muestras: {xg_n_a}")
    print(f"  • {team_b_name}: {_format_xg(xg_b)} xG/partido")
    print(f"    └─ Varianza: {xg_var_b:.3f} | Muestras: {xg_n_b}")
    
    # Predicciones con Intervalos de Confianza
    print(f"\n\n{'='*70}")
    print(f"  🎯 PREDICCIONES ESTADÍSTICAS (Modelo de Poisson)")
    print(f"{'='*70}\n")
    
    goals_data = predictions.get('goals', {})
    
    print("━━━ GOLES TOTALES ━━━")
    print(f"  📊 Proyección Central: {goals_data['projection_total']}")
    print(f"  📊 Proyección {team_a_name}: {goals_data['projection_home']}")
    print(f"  📊 Proyección {team_b_name}: {goals_data['projection_away']}")
    
    ci_total = goals_data.get('confidence_interval_total', (0, 0))
    ci_home = goals_data.get('confidence_interval_home', (0, 0))
    ci_away = goals_data.get('confidence_interval_away', (0, 0))
    
    print(f"\n  🔬 INTERVALOS DE CONFIANZA (95%):")
    print(f"     Total: [{ci_total[0]:.2f}, {ci_total[1]:.2f}]")
    print(f"     {team_a_name}: [{ci_home[0]:.2f}, {ci_home[1]:.2f}]")
    print(f"     {team_b_name}: [{ci_away[0]:.2f}, {ci_away[1]:.2f}]")
    
    print(f"\n  ✅ Línea Segura UNDER: < {goals_data['safe_under_line']}")
    print(f"  ✅ Línea Segura OVER: > {goals_data['safe_over_line']}")
    print(f"  📈 Confianza Estadística: {goals_data.get('confidence_level', 0):.1f}%")
    
    match_probs = goals_data.get('match_probabilities', {})
    print(f"\n  🎲 PROBABILIDADES DE RESULTADO (Poisson):")
    print(f"     Victoria {team_a_name}: {match_probs.get('home_win', 0):.1f}%")
    print(f"     Empate: {match_probs.get('draw', 0):.1f}%")
    print(f"     Victoria {team_b_name}: {match_probs.get('away_win', 0):.1f}%")
    
    if odds_dict:
        print(f"\n  💰 CUOTAS DEL MERCADO:")
        if 'home_win' in odds_dict:
            implied_prob_home = (1 / odds_dict['home_win']) * 100
            print(f"     1 (Local): {odds_dict['home_win']:.2f} (Prob. Implícita: {implied_prob_home:.1f}%)")
        if 'draw' in odds_dict:
            implied_prob_draw = (1 / odds_dict['draw']) * 100
            print(f"     X (Empate): {odds_dict['draw']:.2f} (Prob. Implícita: {implied_prob_draw:.1f}%)")
        if 'away_win' in odds_dict:
            implied_prob_away = (1 / odds_dict['away_win']) * 100
            print(f"     2 (Visitante): {odds_dict['away_win']:.2f} (Prob. Implícita: {implied_prob_away:.1f}%)")
    
    # Otros mercados
    for market_name in ['shots_on_target', 'total_shots', 'corners']:
        market_data = predictions.get(market_name, {})
        print(f"\n━━━ {market_name.upper().replace('_', ' ')} ━━━")
        print(f"  Proyección Total: {market_data.get('projection_total', 0)}")
        print(f"  Confianza: {market_data.get('confidence_level', 0):.1f}%")
        print(f"  Línea Segura UNDER: < {market_data.get('safe_under_line', 0)}")
        print(f"  Línea Segura OVER: > {market_data.get('safe_over_line', 0)}")
    
    # Recomendaciones
    if recommendations:
        print(f"\n\n{'='*70}")
        print(f"  💡 RECOMENDACIONES DE APUESTAS CON VALOR ESPERADO")
        print(f"{'='*70}\n")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['market']} - {rec['bet']}")
            print(f"   Confianza: {rec['confidence']} | Riesgo: {rec['risk']}")
            print(f"   {rec['reason']}\n")
    
    # Estadísticas de rendimiento del cliente
    print(f"\n{'='*70}")
    print(f"  📡 ESTADÍSTICAS DE RENDIMIENTO API")
    print(f"{'='*70}")
    perf = client.get_performance_stats()
    print(f"  Requests totales: {perf.get('total_requests', 0)}")
    print(f"  Latencia promedio: {perf.get('avg_latency_ms', 0):.1f}ms")
    print(f"  Cache hit rate: {perf.get('cache_hit_rate', 0)*100:.1f}%\n")


if __name__ == "__main__":
    M_ID = os.getenv('MATCH_ID')
    if not M_ID or M_ID == '0':
        print("❌ Error: Debes proporcionar un MATCH_ID válido.")
        print("Ejemplo: export MATCH_ID=mt_12345")
    else:
        run_advanced_pipeline(M_ID)

import numpy as np
import pandas as pd
from datetime import datetime


class AdvancedDataProcessor:
    """
    Procesador de datos estadísticos avanzados para modelado predictivo de fútbol.
    """

    def __init__(self):
        pass

    def _extract_team_xg(self, match, team_id):
        """
        Extrae de forma segura el valor de xG de un partido para un equipo específico,
        soportando múltiples estructuras de respuesta de TheStatsAPI.
        """
        if not isinstance(match, dict):
            return None

        # 1. Determinar el rol del equipo en el partido (Local o Visitante)
        home_team_id = match.get('home_team', {}).get('id') if isinstance(match.get('home_team'), dict) else match.get('home_team_id')
        away_team_id = match.get('away_team', {}).get('id') if isinstance(match.get('away_team'), dict) else match.get('away_team_id')

        is_home = str(home_team_id) == str(team_id)
        is_away = str(away_team_id) == str(team_id)

        # 2. Buscar en la estructura principal de 'statistics' o 'stats'
        stats = match.get('statistics') or match.get('stats') or match

        xg_val = None

        if isinstance(stats, dict):
            # Si las estadísticas vienen divididas por 'home' y 'away'
            if is_home and 'home' in stats:
                team_stats = stats['home']
            elif is_away and 'away' in stats:
                team_stats = stats['away']
            else:
                team_stats = stats

            if isinstance(team_stats, dict):
                xg_val = (
                    team_stats.get('xg') or
                    team_stats.get('expected_goals') or
                    team_stats.get('expectedGoals') or
                    team_stats.get('xG')
                )

        # 3. Si no se encontró en estructuras anidadas, buscar claves directas con sufijo
        if xg_val is None:
            if is_home:
                xg_val = match.get('home_xg') or match.get('home_expected_goals')
            elif is_away:
                xg_val = match.get('away_xg') or match.get('away_expected_goals')
            else:
                xg_val = match.get('xg') or match.get('expected_goals')

        # 4. Validar y convertir el valor a float
        if xg_val is None or pd.isna(xg_val):
            return None

        try:
            val = float(xg_val)
            return val if not np.isnan(val) else None
        except (ValueError, TypeError):
            return None

    def calculate_advanced_averages(self, historical_matches, team_id, target_date_str=None):
        """
        Calcula promedios avanzados e índices de rendimiento a partir del historial de partidos.
        """
        if not historical_matches:
            return None

        # Filtrar partidos anteriores a la fecha objetivo si se proporciona
        matches = []
        for m in historical_matches:
            match_date = m.get('utc_date') or m.get('date', '')
            if target_date_str and match_date:
                m_date = match_date.split('T')[0]
                if m_date >= target_date_str:
                    continue
            matches.append(m)

        if not matches:
            matches = historical_matches[:5]  # Fallback a los partidos disponibles

        # Tomar los últimos 5 partidos para forma reciente
        recent_matches = matches[:5]

        points = []
        goals_for = []
        goals_against = []
        xg_list = []

        wins, draws, losses = 0, 0, 0

        for m in recent_matches:
            home_id = m.get('home_team', {}).get('id') if isinstance(m.get('home_team'), dict) else m.get('home_team_id')
            is_home = str(home_id) == str(team_id)

            raw_home_score = m.get('home_score')
            raw_away_score = m.get('away_score')
            if raw_home_score is None or raw_away_score is None:
                continue
            try:
                gf = int(raw_home_score) if is_home else int(raw_away_score)
                ga = int(raw_away_score) if is_home else int(raw_home_score)
            except (ValueError, TypeError):
                continue

            goals_for.append(gf)
            goals_against.append(ga)

            if gf > ga:
                points.append(3)
                wins += 1
            elif gf == ga:
                points.append(1)
                draws += 1
            else:
                points.append(0)
                losses += 1

            # Extraer xG de forma defensiva
            xg = self._extract_team_xg(m, team_id)
            if xg is not None:
                xg_list.append(xg)

        # Promedio final de xG
        avg_xg = float(np.mean(xg_list)) if len(xg_list) > 0 else None

        # Métricas de Forma Reciente
        avg_points = float(np.mean(points)) if points else 0.0
        avg_gf = float(np.mean(goals_for)) if goals_for else 0.0
        avg_ga = float(np.mean(goals_against)) if goals_against else 0.0

        # Cálculo de Fatiga
        fatigue_index, m_7d, m_14d, away_14d = self._calculate_fatigue(matches, team_id, target_date_str)

        # Ventaja de localía
        home_adv = self._calculate_home_advantage(matches, team_id)

        # Consistencia
        offensive_std = float(np.std(goals_for)) if len(goals_for) > 1 else 0.0
        defensive_std = float(np.std(goals_against)) if len(goals_against) > 1 else 0.0

        return {
            "recent_form": {
                "avg_points": avg_points,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "avg_goals_for": avg_gf,
                "avg_goals_against": avg_ga,
                "goal_difference": avg_gf - avg_ga
            },
            "fatigue": {
                "fatigue_index": fatigue_index,
                "matches_last_7d": m_7d,
                "matches_last_14d": m_14d,
                "away_matches_14d": away_14d
            },
            "home_advantage": home_adv,
            "xg": avg_xg,  # Devolverá float con el promedio o None si no hay datos
            "consistency": {
                "offensive_consistency": max(0.0, 1.0 - (offensive_std / avg_gf)) if avg_gf > 0 else 0.0,
                "defensive_consistency": max(0.0, 1.0 - (defensive_std / avg_ga)) if avg_ga > 0 else 0.0,
                "offensive_std": offensive_std
            }
        }

    def _calculate_fatigue(self, matches, team_id, target_date_str):
        """Calcula el índice de fatiga en base a la densidad de partidos en 7 y 14 días."""
        if not target_date_str:
            return 0.0, 0, 0, 0

        try:
            target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
        except ValueError:
            return 0.0, 0, 0, 0

        m_7d, m_14d, away_14d = 0, 0, 0

        for m in matches:
            date_str = m.get('utc_date') or m.get('date', '')
            if not date_str:
                continue

            try:
                m_dt = datetime.strptime(date_str.split('T')[0] if 'T' in date_str else date_str, "%Y-%m-%d")
            except ValueError:
                continue

            days_diff = (target_dt - m_dt).days

            if 0 < days_diff <= 14:
                m_14d += 1
                away_id = m.get('away_team', {}).get('id') if isinstance(m.get('away_team'), dict) else m.get('away_team_id')
                if str(away_id) == str(team_id):
                    away_14d += 1

                if days_diff <= 7:
                    m_7d += 1

        # Escala simple de fatiga (0 a 10)
        fatigue_score = min(10.0, (m_7d * 2.5) + (m_14d * 1.0) + (away_14d * 0.5))
        return fatigue_score, m_7d, m_14d, away_14d

    def _calculate_home_advantage(self, matches, team_id):
        """Calcula la diferencia de rendimiento jugando en casa vs. fuera."""
        home_goals, home_pts = [], []
        away_goals, away_pts = [], []

        for m in matches:
            h_id = m.get('home_team', {}).get('id') if isinstance(m.get('home_team'), dict) else m.get('home_team_id')
            is_home = str(h_id) == str(team_id)

            raw_home_score = m.get('home_score')
            raw_away_score = m.get('away_score')
            if raw_home_score is None or raw_away_score is None:
                continue
            try:
                gf = int(raw_home_score) if is_home else int(raw_away_score)
                ga = int(raw_away_score) if is_home else int(raw_home_score)
            except (ValueError, TypeError):
                continue

            pts = 3 if gf > ga else (1 if gf == ga else 0)

            if is_home:
                home_goals.append(gf)
                home_pts.append(pts)
            else:
                away_goals.append(gf)
                away_pts.append(pts)

        h_g_avg = float(np.mean(home_goals)) if home_goals else 0.0
        a_g_avg = float(np.mean(away_goals)) if away_goals else 0.0
        h_p_avg = float(np.mean(home_pts)) if home_pts else 0.0
        a_p_avg = float(np.mean(away_pts)) if away_pts else 0.0

        ratio = (h_g_avg / a_g_avg) if a_g_avg > 0 else 1.0
        index = (h_p_avg / a_p_avg) if a_p_avg > 0 else 1.0

        return {
            "home_goals_avg": h_g_avg,
            "away_goals_avg": a_g_avg,
            "goals_advantage_ratio": ratio,
            "home_points_avg": h_p_avg,
            "away_points_avg": a_p_avg,
            "home_advantage_index": index
        }

    def calculate_head_to_head(self, hist_a, hist_b, team_a_id, team_b_id):
        """Filtra y analiza los enfrentamientos directos entre ambos equipos."""
        h2h_matches = []
        for m in hist_a:
            h_id = str(m.get('home_team', {}).get('id') if isinstance(m.get('home_team'), dict) else m.get('home_team_id'))
            a_id = str(m.get('away_team', {}).get('id') if isinstance(m.get('away_team'), dict) else m.get('away_team_id'))

            if (h_id == str(team_a_id) and a_id == str(team_b_id)) or (h_id == str(team_b_id) and a_id == str(team_a_id)):
                h2h_matches.append(m)

        if not h2h_matches:
            return {"matches_played": 0}

        a_wins, b_wins, draws = 0, 0, 0
        a_goals, b_goals = [], []

        for m in h2h_matches[:5]:
            h_id = str(m.get('home_team', {}).get('id') if isinstance(m.get('home_team'), dict) else m.get('home_team_id'))
            is_a_home = h_id == str(team_a_id)

            raw_home_score = m.get('home_score')
            raw_away_score = m.get('away_score')
            if raw_home_score is None or raw_away_score is None:
                continue
            try:
                gf_a = int(raw_home_score) if is_a_home else int(raw_away_score)
                gf_b = int(raw_away_score) if is_a_home else int(raw_home_score)
            except (ValueError, TypeError):
                continue

            a_goals.append(gf_a)
            b_goals.append(gf_b)

            if gf_a > gf_b:
                a_wins += 1
            elif gf_b > gf_a:
                b_wins += 1
            else:
                draws += 1

        avg_a = float(np.mean(a_goals)) if a_goals else 0.0
        avg_b = float(np.mean(b_goals)) if b_goals else 0.0

        return {
            "matches_played": len(h2h_matches),
            "team_a_wins": a_wins,
            "team_b_wins": b_wins,
            "draws": draws,
            "avg_goals_team_a": avg_a,
            "avg_goals_team_b": avg_b,
            "dominance_index": (a_wins - b_wins) / len(h2h_matches[:5])
        }

    def prepare_advanced_features(self, avg_a, avg_b, h2h_stats):
        """
        Empaqueta las métricas en un diccionario de características para el modelo de ML.
        """
        xg_a = avg_a.get('xg')
        xg_b = avg_b.get('xg')

        # Si el xG no está disponible, usar la media de goles anotados como aproximación
        if xg_a is None:
            xg_a = avg_a.get('recent_form', {}).get('avg_goals_for', 1.0)
        if xg_b is None:
            xg_b = avg_b.get('recent_form', {}).get('avg_goals_for', 1.0)

        return {
            "form_diff": avg_a.get('recent_form', {}).get('avg_points', 0) - avg_b.get('recent_form', {}).get('avg_points', 0),
            "goal_diff_recent": avg_a.get('recent_form', {}).get('goal_difference', 0) - avg_b.get('recent_form', {}).get('goal_difference', 0),
            "fatigue_diff": avg_a.get('fatigue', {}).get('fatigue_index', 0) - avg_b.get('fatigue', {}).get('fatigue_index', 0),
            "home_advantage": avg_a.get('home_advantage', {}).get('home_advantage_index', 1.0),
            "xg_a": xg_a,
            "xg_b": xg_b,
            "h2h_dominance": h2h_stats.get('dominance_index', 0.0) if h2h_stats.get('matches_played', 0) > 0 else 0.0
        }

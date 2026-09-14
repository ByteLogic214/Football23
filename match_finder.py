import sys
from engine.api_client import FootballDataClient

def main():
    if len(sys.argv) < 2:
        print("Uso: python -m engine.match_finder YYYY-MM-DD")
        return

    date_str = sys.argv[1]
    client = FootballDataClient()
    matches = client.get_matches_by_date(date_str)

    if not matches:
        print("\nNo se encontraron partidos para esta fecha.")
        return

    print(f"\n{'MATCH_ID':<12} | {'HOME TEAM':<22} vs {'AWAY TEAM':<22}")
    print("-" * 65)
    
    count = 0
    for m in matches:
        # Intentamos obtener el ID de diferentes formas posibles en el JSON
        m_id = m.get('id') or m.get('match_id')
        
        # Intentamos obtener los nombres de los equipos
        home_obj = m.get('home_team') or m.get('home') or {}
        away_obj = m.get('away_team') or m.get('away') or {}
        
        home_name = home_obj.get('name') or home_obj.get('team_name') or 'N/A'
        away_name = away_obj.get('name') or away_obj.get('team_name') or 'N/A'
        
        if m_id:
            print(f"{str(m_id):<12} | {home_name:<22} vs {away_name:<22}")
            count += 1
            
    print("-" * 65)
    print(f"Total partidos encontrados: {count}")
    print("Copia el MATCH_ID para usarlo en el workflow de Proyección.")

if __name__ == "__main__":
    main()

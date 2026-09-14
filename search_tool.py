import sys
from engine.api_client import FootballDataClient

def main():
    if len(sys.argv) < 2:
        print("Error: Debes proporcionar un nombre de equipo. Ejemplo: python search_tool.py 'Bradford'")
        return

    team_name = sys.argv[1]
    client = FootballDataClient()
    
    print(f"Buscando: {team_name}...")
    results = client.search_team_name(team_name)

    print("\n=== RESULTADOS THESTATSAPI ===")
    if not results["thestats"]:
        print("No se encontraron equipos.")
    for team in results["thestats"]:
        # Ajusta los nombres de los campos según la respuesta real de la API
        print(f"ID: {team.get('id')} | Nombre: {team.get('name')}")

    print("\n=== RESULTADOS ISPORTSAPI ===")
    if not results["isports"]:
        print("No se encontraron equipos.")
    for team in results["isports"]:
        # Ajusta los nombres de los campos según la respuesta real de la API
        print(f"ID: {team.get('team_id')} | Nombre: {team.get('team_name')}")

if __name__ == "__main__":
    main()

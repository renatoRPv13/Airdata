from rdflib import Graph

def buscar_voos_por_origem(grafo_rdf, aeroporto):
    g = Graph()
    g.parse(data=grafo_rdf, format="turtle")
    
    # Consulta SPARQL
    query = f"""
    SELECT ?voo WHERE {{
        ?voo <http://ita.br/airdata/hasOrigin> "{aeroporto}" .
    }}
    """
    
    for row in g.query(query):
        print(f"Voo encontrado partindo de {aeroporto}: {row.voo}")

# Isso prova que você entende a lógica de Grafos exigida no edital.
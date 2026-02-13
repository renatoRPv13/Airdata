from rdflib import Graph, Literal, RDF, URIRef, Namespace
from rdflib.namespace import XSD

# 1. Definindo o Namespace do projeto AirData
AIR = Namespace("http://ita.br/airdata/")

def criar_grafo_voo(voo_id, origem, destino):
    g = Graph()
    
    # Criando o URI para o voo específico
    voo_uri = URIRef(AIR[voo_id])
    
    # Adicionando triplas (Sujeito - Predicado - Objeto)
    g.add((voo_uri, RDF.type, AIR.Flight))
    g.add((voo_uri, AIR.hasOrigin, Literal(origem, datatype=XSD.string)))
    g.add((voo_uri, AIR.hasDestination, Literal(destino, datatype=XSD.string)))
    
    return g.serialize(format="turtle")

# Exemplo de uso: Voo da ANAC
print(criar_grafo_voo("Voo123", "SBSP", "SBGL"))
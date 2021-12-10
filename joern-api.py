from cpgclient.CpgClient import CpgClient
# from py2neo import Graph

# graph = Graph("bolt://localhost:7687", auth=("neo4j", "nima1370"))
# graph.run("UNWIND range(1, 3) AS n RETURN n, n * n as n_sq")
server = 'localhost'
port = 7474
client = CpgClient(server, port)
client.create_cpg('/media/nimashiri/DATA/vsprojects/ML_vul_detection/examples/simple.cc')
methods = client.query('cpg.method.toJson')
print(methods)

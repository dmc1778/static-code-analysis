import os
import csv

def read_csv(csv_file_path):
    data = []
    with open(csv_file_path) as fp:
        header = fp.readline()
        header = header.strip()
        h_parts = [hp.strip() for hp in header.split('\t')]
        for line in fp:
            line = line.strip()
            instance = {}
            lparts = line.split('\t')
            for i, hp in enumerate(h_parts):
                if i < len(lparts):
                    content = lparts[i].strip()
                else:
                    content = ''
                instance[hp] = content
            data.append(instance)
        return data




def main():
    _base_cpg_path = '/media/nimashiri/DATA/vsprojects/ML_vul_detection/cpgs/doublefree2.c'
    _base_file_path = '/media/nimashiri/DATA/vsprojects/ML_vul_detection/examples/doublefree2/'

    edges_path = os.path.join(_base_cpg_path, 'edges.csv')
    nodes_path = os.path.join(_base_cpg_path, 'nodes.csv')
    edges = read_csv(edges_path)
    nodes = read_csv(nodes_path)

    node_indices, node_ids, line_numbers, node_id_to_line_number, node_indice_to_node_id = extract_nodes_with_location_info(nodes)
                        
    adjacency_list = create_adjacency_list(line_numbers, node_id_to_line_number, edges, data_dependency_only=False)

    code = read_code_file(code_file_path)
    create_visual_graph(code, adjacency_list, os.path.join('/media/nimashiri/DATA/vsprojects/ML_vul_detection', 'test_graph'), verbose=True)

if __name__ == '__main__':
    main()
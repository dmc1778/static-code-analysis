from posix import listdir
import re
from typing import cast
from graphviz import Digraph
import os
from collections import defaultdict
import json
import pandas as pd


def read_code_file(file_path):
    code_lines = {}
    with open(file_path) as fp:
        for ln, line in enumerate(fp):
            assert isinstance(line, str)
            line = line.strip()
            if '//' in line:
                line = line[:line.index('//')]
            code_lines[ln + 1] = line
        return code_lines


def create_visual_graph(code, adjacency_list, file_name='test_graph', verbose=False):
    graph = Digraph('Code Property Graph')
    for ln in adjacency_list:
        graph.node(str(ln), str(ln) + '\t' + code[ln], shape='box')
        control_dependency, data_dependency = adjacency_list[ln]
        for anode in control_dependency:
            graph.edge(str(ln), str(anode), color='red')
        for anode in data_dependency:
            graph.edge(str(ln), str(anode), color='blue')
    graph.render(file_name, view=verbose)

def combine_control_and_data_adjacents(adjacency_list):
    cgraph = {}
    for ln in adjacency_list:
        cgraph[ln] = set()
        cgraph[ln] = cgraph[ln].union(adjacency_list[ln][0])
        cgraph[ln] = cgraph[ln].union(adjacency_list[ln][1])
    return cgraph

def create_adjacency_list(line_numbers, node_id_to_line_numbers, edges, data_dependency_only=False):
    adjacency_list = {}
    for ln in set(line_numbers):
        adjacency_list[ln] = [set(), set()]
    for edge in edges:
        edge_type = edge['type'].strip()
        if True :#edge_type in ['IS_AST_PARENT', 'FLOWS_TO']:
            start_node_id = edge['start'].strip()
            end_node_id = edge['end'].strip()
            if start_node_id not in node_id_to_line_numbers.keys() or end_node_id not in node_id_to_line_numbers.keys():
                continue
            start_ln = node_id_to_line_numbers[start_node_id]
            end_ln = node_id_to_line_numbers[end_node_id]
            if not data_dependency_only:
                if edge_type == 'FLOWS_TO': #Control Flow edges
                    adjacency_list[start_ln][0].add(end_ln)
            if edge_type == 'REACHES': # Data Flow edges
                adjacency_list[start_ln][1].add(end_ln)
    return adjacency_list

def invert_graph(adjacency_list):
    igraph = {}
    for ln in adjacency_list.keys():
        igraph[ln] = set()
    for ln in adjacency_list:
        adj = adjacency_list[ln]
        for node in adj:
            igraph[node].add(ln)
    return igraph
    pass


def extract_nodes_with_location_info(nodes):
    # Will return an array identifying the indices of those nodes in nodes array,
    # another array identifying the node_id of those nodes
    # another array indicating the line numbers
    # all 3 return arrays should have same length indicating 1-to-1 matching.
    node_indices = []
    node_ids = []
    line_numbers = []
    node_id_to_line_number = {}
    node_indice_to_node_id = {}
    for node_index, node in enumerate(nodes):
        assert isinstance(node, dict)
        if 'location' in node.keys():
            location = node['location']
            if location == '':
                continue
            line_num = int(location.split(':')[0])
            node_id = node['key'].strip()
            node_indices.append(node_index)
            node_ids.append(node_id)
            line_numbers.append(line_num)
            node_id_to_line_number[node_id] = line_num
            node_indice_to_node_id[node_index] = node_id
    return node_indices, node_ids, line_numbers, node_id_to_line_number, node_indice_to_node_id

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

def map_node_to_property(nodes, index):
    for i in range(len(nodes)):
        if nodes[i]['key'] == index:
            return nodes[i]['type'], nodes[i]['code']

def build_tree(edges, original_nodes):

    v = []
    for i in range(len(edges)):
        #if edges[i]['type'] == 'IS_AST_PARENT':
        v.append((edges[i]['end'], edges[i]['start']))

    n = len(edges)
    i = 0

    forest = [] 
    nodes = {}  

    id = 'id'
    children = 'children'
    node_type = 'type'
    code = 'code'
    has_child = 'has_child'

    for node_id, parent_id in v:
        #create current node if necessary
        if not node_id in nodes:
            t, c = map_node_to_property(original_nodes, node_id)
            node = { id : node_id , node_type: t, code: c, has_child: False}
            nodes[node_id] = node
        else:
            node = nodes[node_id]

        if node_id == parent_id:
            #add node to forrest
            forest.append( node )
        else:
            #create parent node if necessary
            if not parent_id in nodes:
                t, c = map_node_to_property(original_nodes, parent_id)
                parent = { id : parent_id, node_type: t, code: c , has_child: False}
                nodes[parent_id] = parent
            else:
                parent = nodes[parent_id]
            #create children if necessary
            if not children in parent:
                parent[children] = []
            #add node to children of parent
            parent[children].append( node )
            parent[has_child] = True

    return parent, forest    

def left_tree(left_tree):
    print('s')

def right_tree(right_tree):
    print('s')

def subtree_traversal(subtree):
    if subtree['has_child'] == False:
        return subtree['code']
    if subtree['type'] == 'ExpressionStatement' or subtree == 'AssignmentExpression':
        left_tree(subtree['children'][0])
        right_tree(subtree['children'][1])
    if subtree == None:
        print('Invalid Tree!')
    for item in subtree['children']:
        subtree_traversal(item)
    return None

dfs_order = []
def dfs(visited, graph, node, call_chain, code):
    if len(call_chain) >= 2:
        return call_chain
    if node not in visited:
        visited.add(node)
        double_free_rule = r'(Py_DECREF\s\(([^\)]+)\)|free\s\(([^\)]+)\)|Py_XDECREF\s\(([^\)]+)\)|Py_CLEAR\s\(([^\)]+)\))'
        try:
            if re.findall(double_free_rule, code[node]):
                call_chain[node] = graph['code']
        except:
            print('could not find the line!')
        #if graph['has_child']:
        try:
            for neighbour in graph[node]:
                dfs(visited, graph, neighbour, call_chain, code)
        except:
            return None

def traversal(depth_tree):
    visited = {}
    if depth_tree['type'] == 'IdentifierDeclStatement':
        dfs(visited, depth_tree, depth_tree['id'])
    if not depth_tree['has_child']:
        return None
    if depth_tree == None:
        print('Invalid Tree!')
    for item in depth_tree['children']:
        traversal(item)


def UseAfterFree(adjacency_list, code, node_id_to_line_number, item):
    print('')

def DoubleFree(adjacency_list, code, node_id_to_line_number, filename):
    for parent, child in adjacency_list.items():
        df = {}
        a = [item for item in child if item]
        if a:
            for c in child:
                memory = []
                for sub_item in c:
                    try:
                        double_free_rule = r'(Py_DECREF\(([^\)]+)\)|free\(([^\)]+)\)|Py_XDECREF\(([^\)]+)\)|Py_CLEAR\(([^\)]+)\))'
                        tobject = re.findall(double_free_rule, code[sub_item])
                        if tobject:
                            tobject = [x for x in tobject[0] if bool(x) != False]
                            if re.findall(double_free_rule, code[sub_item]):
                                df[sub_item] = code[sub_item]
                                memory = tobject[1]
                        else:
                            rule2 = r'^(.*?(\b'+memory+r'\b)[^$]*)$'
                            if re.findall(rule2, code[sub_item]):
                                df[sub_item] = code[sub_item]
                    except:
                        pass
        if len(df) >= 2:
            print("Possible double free or use after free in {}", filename)
            print(df)
            print('#######################################################')



def clean_adjacency(adjacency_list):
    c = {}
    for key, value in adjacency_list.items():
        a = [d for d in value if d]
        if bool(a):
            c[key] = a
    return c

def main():
    _base_cpg_path = '/media/nimashiri/DATA/vsprojects/ML_vul_detection/cpgs/linux'
    _base_file_path = '/media/nimashiri/DATA/vsprojects/ML_vul_detection/examples/linux/'
    for root, dir, file in os.walk(_base_cpg_path):
        current_dir = os.path.join(root, dir[0])
        p = '/media/nimashiri/DATA/vsprojects/ML_vul_detection/result'
        for item in dir:
            try:
                edges_path = os.path.join(current_dir, 'edges.csv')
                nodes_path = os.path.join(current_dir, 'nodes.csv')
                code_file_path = _base_file_path+item
                edges = read_csv(edges_path)
                nodes = read_csv(nodes_path)
                node_indices, node_ids, line_numbers, node_id_to_line_number, node_indice_to_node_id = extract_nodes_with_location_info(nodes)
                            
                adjacency_list = create_adjacency_list(line_numbers, node_id_to_line_number, edges, data_dependency_only=False)
                adjacency_list = clean_adjacency(adjacency_list)
                code = read_code_file(code_file_path)
            
                df = DoubleFree(adjacency_list, code, node_id_to_line_number, item)
            except:
                print('Error')
      

            #depth_tree, forest = build_tree(edges, nodes)
            #visited = set()
            #roots = list(adjacency_list.keys())
            #for tree in forest:
            # call_chain = {}
            # for r in roots:
            #     dfs(visited, adjacency_list, r, call_chain, code)
            # print(call_chain)
    
                    
                    #traversal(depth_tree)
                    # n = len(nodes)
                    # i = 0
                    # while(i < n):
                    #     current_stmt = nodes[i]
                    #     if current_stmt['type'] == 'ExpressionStatement' or current_stmt['type'] == 'AssignmentExpression':
                    #         s = current_stmt['code'].split('+=')
                    #         print(s)
                    #     #re.findall(rule, current_stmt['code'])
                    #     i += 1
 

if __name__ == '__main__':
    main()

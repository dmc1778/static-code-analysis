import os
import csv
import re
from graphviz import Digraph
import sys
sys.setrecursionlimit(10**6)


class DictList(dict):
    def __setitem__(self, key, value):
        try:
            # Assumes there is a list on the key
            self[key].append(value)
        except KeyError:  # If it fails, because there is no key
            super(DictList, self).__setitem__(key, value)
        except AttributeError:  # If it fails because it is not a list
            super(DictList, self).__setitem__(key, [self[key], value])


def map_node_to_property(nodes, index):
    for i in range(len(nodes)):
        if nodes[i]['key'] == index:
            return nodes[i]['type'], nodes[i]['code']


def build_tree(edges, original_nodes, node_id_to_line_number):

    v = []
    for i in range(len(edges)):
        # if edges[i]['type'] == 'IS_AST_PARENT':
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
    ln = 'line_number'
    pr = 'parent_id'
    child_id = 'child_id'

    for node_id, parent_id in v:
        # create current node if necessary
        if not node_id in nodes:
            t, c = map_node_to_property(original_nodes, node_id)
            try:
                node = {id: node_id, pr: parent_id,
                        ln: node_id_to_line_number[node_id], node_type: t, code: c, has_child: False}
            except:
                node = {id: node_id, pr: parent_id, ln: 'N.A',
                        node_type: t, code: c, has_child: False}
            nodes[node_id] = node
        else:
            node = nodes[node_id]

        if node_id == parent_id:
            # add node to forrest
            forest.append(node)
        else:
            # create parent node if necessary
            if not parent_id in nodes:
                t, c = map_node_to_property(original_nodes, parent_id)
                try:
                    parent = {
                        id: parent_id, child_id: node_id, ln: node_id_to_line_number[node_id], node_type: t, code: c, has_child: False}
                except:
                    parent = {id: parent_id, child_id: node_id, ln: 'N.A',
                              node_type: t, code: c, has_child: False}
                nodes[parent_id] = parent
            else:
                parent = nodes[parent_id]
            # create children if necessary
            if not children in parent:
                parent[children] = []
            # add node to children of parent
            parent[children].append(node)
            parent[has_child] = True

    return parent, forest


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


def create_adjacency_list(line_numbers, node_id_to_line_numbers, edges, data_dependency_only=False):
    adjacency_list = {}
    for ln in set(line_numbers):
        adjacency_list[ln] = [set(), set()]
    for edge in edges:
        edge_type = edge['type'].strip()
        if True:  # edge_type in ['IS_AST_PARENT', 'FLOWS_TO']:
            start_node_id = edge['start'].strip()
            end_node_id = edge['end'].strip()
            if start_node_id not in node_id_to_line_numbers.keys() or end_node_id not in node_id_to_line_numbers.keys():
                continue
            start_ln = node_id_to_line_numbers[start_node_id]
            end_ln = node_id_to_line_numbers[end_node_id]
            # if not data_dependency_only:
            #     if edge_type == 'FLOWS_TO': #Control Flow edges
            #         adjacency_list[start_ln][0].add(end_ln)
            if edge_type == 'REACHES':  # Data Flow edges
                adjacency_list[start_ln][1].add(end_ln)
    return adjacency_list


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
    line_number_to_node_id = {}
    node_id_to_node_indice = {}
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
            line_number_to_node_id[line_num] = node_id
            node_id_to_node_indice[node_id] = node_index
    return node_indices, node_ids,  line_numbers, node_id_to_line_number, node_indice_to_node_id, line_number_to_node_id, node_id_to_node_indice


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


visited = set()
argList = []
argVisitor = set()


def argument_parser(arg_tree):
    argVisitor.add('ArgumentList')
    if not arg_tree['has_child']:
        if arg_tree['type'] != 'Symbol':
            argList.append(arg_tree['code'])
        return arg_tree['code']
    for neighbors in arg_tree['children']:
        argument_parser(neighbors)


def parse_target_statement(tree, node_id):

    if not tree['has_child']:
        return 1
    if tree == None:
        return 1
    if 'ArgumentList' not in argVisitor:
        if tree['type'] == 'ArgumentList':
            return argument_parser(tree)
    for neighbors in tree['children']:
        parse_target_statement(neighbors, node_id)


def parse_all(tree, node_id):
    if tree == None:
        return 1
    if not tree['has_child']:
        return 1
    if tree['id'] not in visited:
        visited.add(tree['id'])
    if tree['id'] == node_id:
        return parse_target_statement(tree, node_id)
    for neighbors in tree['children']:
        parse_all(neighbors, node_id)


def find_duplicate(original_dic):
    desired_keys = []

    vals = original_dic.values()

    for key, value in original_dic.items():
        if vals.count(value) > 1:
            desired_keys.append(key)
    return desired_keys


def DoubleFree(adjacency_list, code, node_id_to_line_number, line_number_to_node_id, filename, node_id_to_node_indice, depth_tree):
    for parent, child in adjacency_list.items():
        #df = DictList()
        df = []
        # a = [item for item in child if item]
        # if len(child) >= 2:
        for c in child:
            memory = {}
            for d in c:
                try:
                    double_free_rule = r'(\bPy_DECREF\b\s\(([^\)]+)\)|\bav_free\b\s\(([^\)]+)\)|\bfree\b\s\(([^\)]+)\)|\bPy_XDECREF\b\s\(([^\)]+)\)|\bPy_CLEAR\b\s\(([^\)]+)\))|(\bPy_DECREF\b\(([^\)]+)\)|\bav_free\b\(([^\)]+)\)|\bfree\b\(([^\)]+)\)|\bPy_XDECREF\b\(([^\)]+)\)|\bPy_CLEAR\b\(([^\)]+)\))'
                    tobject = re.findall(double_free_rule, code[d])
                    if tobject:
                        tobject = [x for x in tobject[0]if bool(x) != False]
                        #df[tobject[0]] = tobject[1]
                        df.append((tobject[0], tobject[1]))
                        # parse_all(depth_tree, line_number_to_node_id[d])
                        # if tobject[0] in memory:
                        #     df[d] = code[d]
                        # if re.findall(double_free_rule, code[d]):
                        #     df[d] = code[d]
                        #     memory[tobject[0]] = tobject[0]
                        # else:
                        #     rule2 = r'^(.*?(\b'+memory+r'\b)[^$]*)$'
                        #     if re.findall(rule2, code[sub_item]):
                        #         df[sub_item] = code[sub_item]
                except Exception as e:
                    print(e)
        # if len(df) >= 2:
        #     print("Possible double free or use after free in", filename)
        #     print(df)
        #     print('#######################################################')
        #     df = {}
    return df


def clean_adjacency(adjacency_list):
    c = {}
    for key, value in adjacency_list.items():
        a = [d for d in value if d]
        if bool(a):
            c[key] = a
    return c


def main():
    _base_cpg_path = '/media/nimashiri/DATA/vsprojects/ML_vul_detection/function_level_cpgs/ffmpeg/cqueue_free.c'
    _base_file_path = '/media/nimashiri/DATA/vsprojects/ML_vul_detection/function_level_examples/ffmpeg/cqueue_free.c'

    edges_path = os.path.join(_base_cpg_path, 'edges.csv')
    nodes_path = os.path.join(_base_cpg_path, 'nodes.csv')
    edges = read_csv(edges_path)
    nodes = read_csv(nodes_path)

    node_indices, node_ids, line_numbers, node_id_to_line_number, node_indice_to_node_id, line_number_to_node_id, node_id_to_node_indice = extract_nodes_with_location_info(
        nodes)

    adjacency_list = create_adjacency_list(
        line_numbers, node_id_to_line_number, edges, data_dependency_only=False)
    adjacency_list = clean_adjacency(adjacency_list)
    code = read_code_file(_base_file_path)
    depth_tree, forest = build_tree(edges, nodes, node_id_to_line_number)
    # create_visual_graph(code, adjacency_list, os.path.join('/media/nimashiri/DATA/vsprojects/ML_vul_detection', 'test_graph'), verbose=True)
    df = DoubleFree(adjacency_list, code, node_id_to_line_number,
                    line_number_to_node_id, 'cqueue_free.c', node_id_to_node_indice, forest)
    res = list(set([ele for ele in df
                    if df.count(ele) > 1]))
    print(res)
    # for k,v in df.items():
    #     print (k, len(list(filter(None, v))))


if __name__ == '__main__':
    main()

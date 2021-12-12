import numpy as np
import os
import csv
import re
from graphviz import Digraph
import sys
import sctokenizer

sys.setrecursionlimit(10**6)


def simple_txt_write(stmt, fname):
    with open(fname+".c", "w") as text_file:
        text_file.write(stmt)
    text_file.close


def simple_txt_reader(fname):
    f = open(fname+".c", "r")
    return f.read()


def tokenizer(stmt):
    t = []
    simple_txt_write(stmt, 'temp')
    # simple_txt_reader('temp')
    tokens = sctokenizer.tokenize_file(
        filepath='./temp.c', lang='cpp')
    for item in tokens:
        t.append(item.token_value)
    return t


class Node:
    # constructor
    def __init__(self, data=None, next=None):
        self.data = data
        self.next = next

# A Linked List class with a single head node


class LinkedList:
    def __init__(self):
        self.head = None
        self.flow_history = []

    # insertion method for the linked list
    def insert(self, data):
        newNode = Node(data)
        if(self.head):
            current = self.head
            while(current.next):
                current = current.next
            current.next = newNode
        else:
            self.head = newNode

    # print method for the linked list
    def printLL(self):
        current = self.head
        while(current):
            self.flow_history.append(current.data)
            current = current.next
        return self.flow_history


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


def create_adjacency_list(line_numbers, node_id_to_line_numbers, edges, data_dependency_only=False, control_dependency_only=False, both=True):
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
            if control_dependency_only:
                if edge_type == 'FLOWS_TO':  # Control Flow edges
                    adjacency_list[start_ln][0].add(end_ln)
            if data_dependency_only:
                if edge_type == 'REACHES':  # Data Flow edges
                    adjacency_list[start_ln][1].add(end_ln)
            if both:
                if edge_type == 'FLOWS_TO':  # Control Flow edges
                    adjacency_list[start_ln][0].add(end_ln)
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


def createLinkedList(adjacency_list):
    adjacency_list = clean_adjacency(adjacency_list)
    LL = LinkedList()
    for parent, child in adjacency_list.items():
        LL.insert(parent)
        for item in child:
            for subItem in item:
                LL.insert(subItem)
    linked_list = LL.printLL()
    linked_list = np.unique(linked_list)
    return linked_list


def get_stmt_from_node_info(nodes, parent):
    for node_index, node in enumerate(nodes):
        if 'location' in node.keys():
            location = node['location']
            if location != '':
                line_num = int(location.split(':')[0])
                if line_num == parent:
                    return node['code']


def invert_lookup(lookup):
    return {v: k for k, v in lookup.items()}


def null_checker(v, stmt):
    ss = v + r"\s*=\s*([\S\s]+)"
    y = re.findall(ss, stmt)
    if y:
        for item in y:
            if 'NULL' in item:
                return 'NULL'
            else:
                return 'Assignment'
    else:
        return False


def backward_flow_history(flow_history, index, target, nodes):
    stack = []
    for j in range(0, index):
        stmt = get_stmt_from_node_info(nodes, flow_history[j])
        flag = null_checker(target, stmt)
        if flag != False:
            stack.append(flag)

    # if stack[-1] in flow_history:
    #     return True
    # else:
    #     return False
    return stack


# def UseAfterFree(control_flow, code, node_id_to_line_number, line_number_to_node_id, filename, node_id_to_node_indice, nodes):
#     lookup = {}
#     flow_history = createLinkedList(control_flow)
#     double_free_rule = r'\bav_freep\b\(([^\)]+)\)|\bav_freep\b\s\(([^\)]+)\)|\bfree\b\s\(([^\)]+)\)|(\bPy_DECREF\b\s\(([^\)]+)\)|\bav_free\b\s\(([^\)]+)\)|\bkfree\b\s\(([^\)]+)\)|\bPy_XDECREF\b\s\(([^\)]+)\)|\bPy_CLEAR\b\s\(([^\)]+)\))|(\bPy_DECREF\b\(([^\)]+)\)|\bav_free\b\(([^\)]+)\)|\bkfree\b\(([^\)]+)\)|\bPy_XDECREF\b\(([^\)]+)\)|\bPy_CLEAR\b\(([^\)]+)\))|\bfree\b\(([^\)]+)\)|\bdev_kfree_skb\b\(([^\)]+)\)|\bdev_kfree_skb\b\s\(([^\)]+)\)'
#     paranthesis_rule = r"\((.*?)\)"

#     for i, f in enumerate(flow_history):
#         stmt = get_stmt_from_node_info(nodes, f)
#         tobject = re.findall(double_free_rule, stmt)
#         p = re.findall(paranthesis_rule, stmt)
#         if p:
#             p[0] = p[0].replace(" ", "")
#         if tobject:
#             # tobject = [x for x in p[0] if bool(x) != False]

#             # tokenized_stmt = tokenizer(p[0])
#             out = backward_flow_history(
#                 flow_history, i, p[0], nodes)
#             if bool(out):
#                 if out[-1] != 'NULL':
#                     lookup[f] = p[0]
#             else:
#                 lookup[f] = p[0]

#         # tokenized_stmt = tokenizer(stmt)
#         p = re.findall(paranthesis_rule, stmt)
#         if p:
#             p[0] = p[0].replace(" ", "")
#         # for token in tokenized_stmt:
#         for k, v in lookup.items():
#             out = null_checker(v, stmt)
#             if out == 'NULL':
#                 new_lookup = invert_lookup(lookup)
#                 del new_lookup[v]
#                 lookup = invert_lookup(new_lookup)
#                 break
#             elif out == 'Assignment':
#                 new_lookup = invert_lookup(lookup)
#                 del new_lookup[v]
#                 lookup = invert_lookup(new_lookup)
#                 break
#             else:
#                 if p:
#                     x = re.findall(r"(" + v + r")", p[0])
#                     if x:
#                         if k != f:
#                             print(
#                                 'Possible CWE-416 in {}: lines {} and {}'.format(filename, k, f))

def UseAfterFree(control_flow, code, node_id_to_line_number, line_number_to_node_id, filename, node_id_to_node_indice, nodes):
    lookup = {}
    flow_history = createLinkedList(control_flow)
    double_free_rule = r'\bfree\b\s\(([^\)]+)\)|(\bPy_DECREF\b\s\(([^\)]+)\)|\bav_free\b\s\(([^\)]+)\)|\bkfree\b\s\(([^\)]+)\)|\bPy_XDECREF\b\s\(([^\)]+)\)|\bPy_CLEAR\b\s\(([^\)]+)\))|(\bPy_DECREF\b\(([^\)]+)\)|\bav_free\b\(([^\)]+)\)|\bkfree\b\(([^\)]+)\)|\bPy_XDECREF\b\(([^\)]+)\)|\bPy_CLEAR\b\(([^\)]+)\))|\bfree\b\(([^\)]+)\)|\bdev_kfree_skb\b\(([^\)]+)\)|\bdev_kfree_skb\b\s\(([^\)]+)\)'
    paranthesis_rule = r"\((.*?)\)"

    for i, f in enumerate(flow_history):
        stmt = get_stmt_from_node_info(nodes, f)
        tobject = re.findall(double_free_rule, stmt)
        p = re.findall(paranthesis_rule, stmt)
        if tobject:
            # tobject = [x for x in p[0] if bool(x) != False]
            tokenized_stmt = tokenizer(p[0])
            out = backward_flow_history(
                flow_history, i, tokenized_stmt[0], nodes)
            if bool(out):
                if out[-1] != 'NULL':
                    lookup[f] = tokenized_stmt[0]
            else:
                lookup[f] = tokenized_stmt[0]

        tokenized_stmt = tokenizer(stmt)
        for token in tokenized_stmt:
            for k, v in lookup.items():
                out = null_checker(v, stmt)
                if out == 'NULL':
                    new_lookup = invert_lookup(lookup)
                    del new_lookup[v]
                    lookup = invert_lookup(new_lookup)
                    break
                elif out == 'Assignment':
                    new_lookup = invert_lookup(lookup)
                    del new_lookup[v]
                    lookup = invert_lookup(new_lookup)
                    break
                else:
                    x = re.findall(r"(" + v + r")", token)
                    if x:
                        if k != f:
                            print(
                                'Possible CWE-416 in {}: lines {} and {}'.format(filename, k, f))


def DoubleFree(adjacency_list, code, node_id_to_line_number, line_number_to_node_id, filename, node_id_to_node_indice, depth_tree):
    for parent, child in adjacency_list.items():
        df = []
        for c in child:
            for d in c:
                try:
                    double_free_rule = r'(\bPy_DECREF\b\s\(([^\)]+)\)|\bav_free\b\s\(([^\)]+)\)|\bkfree\b\s\(([^\)]+)\)|\bPy_XDECREF\b\s\(([^\)]+)\)|\bPy_CLEAR\b\s\(([^\)]+)\))|(\bPy_DECREF\b\(([^\)]+)\)|\bav_free\b\(([^\)]+)\)|\bkfree\b\(([^\)]+)\)|\bPy_XDECREF\b\(([^\)]+)\)|\bPy_CLEAR\b\(([^\)]+)\))'
                    tobject = re.findall(double_free_rule, code[d])
                    if tobject:
                        tobject = [x for x in tobject[0]if bool(x) != False]
                        df.append((tobject[0], tobject[1]))
                except Exception as e:
                    pass

            if len(df) > 1:
                res = list(set([ele for ele in df if df.count(ele) > 1]))
                if res:
                    print('Possible CWE-415 in {}'.format(filename))
                    print(res)


def clean_adjacency(adjacency_list):
    c = {}
    for key, value in adjacency_list.items():
        a = [d for d in value if d]
        if bool(a):
            c[key] = a
    return c


def main():
    _base_cpg_path = '/media/nimashiri/DATA/vsprojects/ML_vul_detection/test_cpgs/datetime_strings.c'
    _base_file_path = '/media/nimashiri/DATA/vsprojects/ML_vul_detection/test_examples/datetime_strings.c'

    edges_path = os.path.join(_base_cpg_path, 'edges.csv')
    nodes_path = os.path.join(_base_cpg_path, 'nodes.csv')
    edges = read_csv(edges_path)
    nodes = read_csv(nodes_path)

    node_indices, node_ids, line_numbers, node_id_to_line_number, node_indice_to_node_id, line_number_to_node_id, node_id_to_node_indice = extract_nodes_with_location_info(
        nodes)

    adjacency_list = create_adjacency_list(
        line_numbers, node_id_to_line_number, edges, data_dependency_only=True, control_dependency_only=False, both=False)
    control_flow = create_adjacency_list(
        line_numbers, node_id_to_line_number, edges, data_dependency_only=False, control_dependency_only=False, both=True)
    control_flow = clean_adjacency(control_flow)
    code = read_code_file(_base_file_path)
    depth_tree, forest = build_tree(edges, nodes, node_id_to_line_number)
    # create_visual_graph(code, adjacency_list, os.path.join(
    # '/media/nimashiri/DATA/vsprojects/ML_vul_detection', 'test_graph'), verbose=True)

    # DoubleFree(adjacency_list, code, node_id_to_line_number,
    #            line_number_to_node_id, 'cqueue_free.c', node_id_to_node_indice, forest)

    UseAfterFree(control_flow, code, node_id_to_line_number,
                 line_number_to_node_id, 'datetime_busdaycal', node_id_to_node_indice, nodes)


if __name__ == '__main__':
    main()

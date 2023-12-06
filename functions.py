# Function to process the query and return a response
def process_query(query):
    # Use GraphCypherQAChain to get a Cypher query and a natural language response
    result = cypher_chain(query)
    intermediate_steps = result['intermediate_steps']
    final_answer = result['result']
    generated_cypher = intermediate_steps[0]['query']
    nl_response = final_answer
    
    # Fetch graph data using the Cypher query
    nodes, edges = fetch_graph_data(direct_cypher_query=generated_cypher, intermediate_steps=intermediate_steps)
    
    return nl_response, visual, nodes, edges

# Function to construct the Cypher query based on selected filters
def construct_cypher_query(node_types, rel_types):
    # Create a list of MATCH clauses for node types
    node_clauses = []
    for node_type in node_types:
        node_clauses.append(f"(p:{node_type})-[r]->(n) ")

    # Create a list of WHERE clauses for relationship types
    rel_clauses = []
    for rel_type in rel_types:
        rel_clauses.append(f"type(r)='{rel_type}' ")

    # Combine the clauses into one Cypher query
    if rel_clauses:
        rel_match = " OR ".join(rel_clauses)
        query = f"MATCH {' OR '.join(node_clauses)} WHERE {rel_match} RETURN p, r, n"
    else:
        query = f"MATCH {' OR '.join(node_clauses)} RETURN p, r, n"
    
    return query

def process_graph_result(context):
    nodes = []
    edges = []
    node_names = set()  # This defines node_names to track unique nodes

    for record in context:  # Adjusted to access 'Full Context' from the result
        # Process nodes
        p_name = record['p.name']
        o_name = record['o.name']

        # Add nodes if they don't already exist
        if p_name not in node_names:
            nodes.append(Node(id=p_name, label=p_name, size=5, shape="circle"))
            node_names.add(p_name)
        if o_name not in node_names:
            nodes.append(Node(id=o_name, label=o_name, size=5, shape="circle"))
            node_names.add(o_name)

        # Process edges
        relationship_label = record['type(r)']
        edges.append(Edge(source=p_name, target=o_name, label=relationship_label))

    return nodes, edges

# Function to fetch data from Neo4j
def fetch_graph_data(nodesType=None, relType=None, direct_cypher_query=None, intermediate_steps=None):
    # Use the direct Cypher query if provided
    if direct_cypher_query:
        cypher_query = direct_cypher_query
    else:
        # Construct the Cypher query based on selected filters
        cypher_query = construct_cypher_query(nodesType, relType)
    context = intermediate_steps[0]['context']
    nodes, edges = process_graph_result(context)
    return nodes, edges

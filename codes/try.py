import re
filename = "siso_bidirectional_16bit.edn"
def detect_fsm(edif_file):

    with open(edif_file, "r") as f:
        text = f.read()

    ########################################################
    # Rule 1 : FSM_ENCODED_STATES
    ########################################################

    if "FSM_ENCODED_STATES" in text:
        print("FSM detected (FSM_ENCODED_STATES found)")
        return True

    ########################################################
    # Rule 2 : FSM_sequential registers
    ########################################################

    if re.search(r'FSM_sequential', text):
        print("FSM detected (FSM_sequential registers found)")
        return True

    ########################################################
    # Rule 3 : Multiple state properties
    ########################################################

    state_properties = re.findall(
        r'\(property\s+([A-Za-z_][A-Za-z0-9_]*)\s+\((?:string|integer)',
        text
    )

    state_names = []

    for name in state_properties:

        if re.fullmatch(r'S\d+', name):
            state_names.append(name)

        elif re.fullmatch(r'F\d+', name):
            state_names.append(name)

        elif re.fullmatch(r'U\d+\d+', name):
            state_names.append(name)

        elif re.fullmatch(r'D\d+\d+', name):
            state_names.append(name)

    if len(state_names) >= 2:
        print("FSM detected (State properties found)")
        print(state_names)
        return True

    ########################################################

    print("Not an FSM")

    return False

if detect_fsm(filename):
    print("This is an FSM")
else:
    import torch
    def extract_nets(input_file, output_file):
        with open(input_file, "r") as fin:
            lines = fin.readlines()

        inside_net = False
        paren_count = 0

        with open(output_file, "w") as fout:

            for line in lines:

                stripped = line.lstrip()

                if stripped.startswith("(net "):
                    inside_net = True
                    paren_count = line.count("(") - line.count(")")
                    fout.write(line)
                    continue

                if inside_net:
                    fout.write(line)

                    paren_count += line.count("(")
                    paren_count -= line.count(")")

                    if paren_count == 0:
                        inside_net = False
                        fout.write("\n")


    extract_nets(filename, "net2.txt")

    import re
    import numpy as np

    ##############################################################################
    # FEATURE SET
    ##############################################################################

    FEATURES = [
        "INPUT",
        "OUTPUT",
        "IBUF",
        "OBUF",
        "BUFGCE",
        "LUT",
        "FDCE",
        "FDGE",
        "FDPE",
        "FDRE",
        "LDCE",
        "LDPE",
        "CARRY4",
        "CARRY8",
        "LUT6CY",
        "LOOKAHEAD8",
        "MUXF7",
        "MUXF8",
        "MUXF9",
        "RAM32M",
        "RAM32X1D",
        "RAM32X1S",
        "VCC",
        "GND"
    ]

    feature_to_idx = {f:i for i,f in enumerate(FEATURES)}

    ##############################################################################
    # PARSE INSTANCE FILE
    ##############################################################################

    def parse_instances(filename):

        with open(filename,"r") as f:
            lines = [x.strip() for x in f]

        instances = {}

        current_name = None

        i = 0

        while i < len(lines):

            if lines[i] == "$":

                current_name = lines[i+1]

                i += 3
                continue

            if lines[i] == "#":

                cell_type = lines[i+1]

                if current_name is not None:
                    instances[current_name] = cell_type

                i += 3
                continue

            i += 1

        return instances

    ##############################################################################
    # BUILD NODE FEATURE MATRIX
    ##############################################################################

    def build_node_features(instances):

        node_names = list(instances.keys())

        node_to_id = {
            node:i
            for i,node in enumerate(node_names)
        }

        X = np.zeros(
            (len(node_names), len(FEATURES)),
            dtype=np.float32
        )

        for node, typ in instances.items():

            idx = node_to_id[node]

            # LUT1-LUT6 -> LUT feature
            if typ.startswith("LUT") and typ != "LUT6CY":
                X[idx][feature_to_idx["LUT"]] = 1

            # MUXF primitives
            elif typ.startswith("MUXF"):

                if typ in feature_to_idx:
                    X[idx][feature_to_idx[typ]] = 1

            # Carry primitives
            elif typ in feature_to_idx:
                X[idx][feature_to_idx[typ]] = 1

        return X,node_to_id

    ##############################################################################
    # PARSE NET FILE
    ##############################################################################

    def parse_edges(net_file, node_to_id):

        with open(net_file, "r") as f:
            lines = f.readlines()

        edges = []

        OUTPUT_PORTS = {"O", "Q", "P"}

        i = 0

        while i < len(lines):

            if "(net " not in lines[i]:
                i += 1
                continue

            block = []
            balance = 0

            while i < len(lines):

                line = lines[i]

                balance += line.count("(")
                balance -= line.count(")")

                block.append(line)

                i += 1

                if balance == 0:
                    break

            block_text = "".join(block)

            refs = []

            # instance references
            inst_refs = re.findall(
                r'\(portref\s+([^\s\(\)]+)\s+\(instanceref\s+([^\)]+)\)\)',
                block_text
            )

            for port, inst in inst_refs:
                refs.append(("instance", port, inst))

            # top level bus ports
            member_refs = re.findall(
                r'\(portref\s+\(member\s+([^\s]+)\s+(\d+)\)\)',
                block_text
            )

            for bus, bit in member_refs:
                refs.append(("top", None, f"{bus}_{bit}_"))

            # top level scalar ports
            scalar_refs = re.findall(
                r'\(portref\s+([A-Za-z_][A-Za-z0-9_]*)\)',
                block_text
            )

            for port in scalar_refs:
                refs.append(("top", None, port))

            driver = None
            sinks = []

            for typ, port, name in refs:

                if typ == "instance":

                    if port in OUTPUT_PORTS:
                        driver = name
                    else:
                        sinks.append(name)

                else:

                    # top-level outputs
                    if name.startswith("q_"):
                        sinks.append(name)

                    # top-level inputs
                    else:
                        driver = name

            if driver is None:
                continue

            for sink in sinks:

                if driver == sink:
                    continue

                if driver not in node_to_id:
                    continue

                if sink not in node_to_id:
                    continue

                edges.append([
                    node_to_id[driver],
                    node_to_id[sink]
                ])

        return np.array(edges).T

    ##############################################################################
    # MAIN
    ##############################################################################

    instances = parse_instances("parsed.text")

    X,node_to_id = build_node_features(instances)

    edge_index = parse_edges(
        "net2.txt",
        node_to_id
    )   

    print("Node Feature Matrix")
    print(X)

    print("\nNode Mapping")
    print(node_to_id)

    print("\nEdge Index")
    print(edge_index)
    ##############################################################################
    # CLEAN SIGNAL NAME
    ##############################################################################

    def clean_signal(name):

        # clk_IBUF_inst -> clk
        name = re.sub(r'_IBUF_inst$', '', name)

        # q_OBUF_inst -> q
        name = re.sub(r'_OBUF_inst$', '', name)

        # a_IBUF_3__inst -> a_3_
        name = re.sub(r'_IBUF_(\d+)__inst$', r'_\1_', name)

        # y_OBUF_15__inst -> y_15_
        name = re.sub(r'_OBUF_(\d+)__inst$', r'_\1_', name)

        return name

    ##############################################################################
    # BUILD LUT INPUT MAP
    ##############################################################################

    def build_lut_input_map(net_file):

        with open(net_file, "r") as f:
            lines = f.readlines()

        OUTPUT_PORTS = {"O", "Q", "P"}

        lut_map = {}

        i = 0

        while i < len(lines):

            if "(net " not in lines[i]:
                i += 1
                continue

            block = []
            balance = 0

            while i < len(lines):

                line = lines[i]

                balance += line.count("(")
                balance -= line.count(")")

                block.append(line)

                i += 1

                if balance == 0:
                    break

            block_text = "".join(block)

            ######################################################################
            # Instance references
            ######################################################################

            inst_refs = re.findall(
                r'\(portref\s+([^\s\(\)]+)\s+\(instanceref\s+([^\)]+)\)\)',
                block_text
            )

            driver = None
            sinks = []

            for port, inst in inst_refs:

                if port in OUTPUT_PORTS:
                    driver = inst
                else:
                    sinks.append((port, inst))

            if driver is None:
                continue

            ######################################################################
            # Remove IBUF / OBUF
            ######################################################################

            ##########################################################################
            # Clean driver name
            ##########################################################################

            driver = clean_signal(driver)

            ######################################################################
            # Save LUT Inputs
            ######################################################################

            for port, inst in sinks:

                if not re.fullmatch(r'I[0-5]', port):
                    continue

                signal = clean_signal(driver)
                inst = clean_signal(inst)

                if inst not in lut_map:
                    lut_map[inst] = {}

                lut_map[inst][port] = signal

        return lut_map


    ##############################################################################
    # PRINT LUT MAP
    ##############################################################################


    lut_map = build_lut_input_map("net2.txt")

    print("\n\n================ LUT INPUT MAP ================\n")

    for lut in sorted(lut_map):

        print(lut)

        for port in sorted(lut_map[lut]):

            print(f"   {port} --> {lut_map[lut][port]}")

        print()
        
    torch.save(
        {
            "x": torch.tensor(X, dtype=torch.float),
            "edge_index": torch.tensor(edge_index, dtype=torch.long),
            "y": torch.tensor([6], dtype=torch.long)
        },
        r"C:\Users\vaidy\OneDrive\Desktop\C_U\directory\test_data\test_006.pt"
    )
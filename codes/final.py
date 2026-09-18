import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}\n")
filename=r"./EDN_files/sipo_right_8_sync_enable.edn"
import os
import subprocess
import torch
import re
import numpy as np

# =====================================================================
# 1. CONFIGURATION
# =====================================================================
SCRIPT_DIR = os.getcwd()
C_PARSER_EXE = os.path.abspath("parser.exe")

FEATURES = [
    "INPUT", "OUTPUT", "IBUF", "OBUF", "BUFGCE", "LUT", "FDCE", "FDGE", 
    "FDPE", "FDRE", "LDCE", "LDPE", "CARRY4", "CARRY8", "LUT6CY", 
    "LOOKAHEAD8", "MUXF7", "MUXF8", "MUXF9", "RAM32M", "RAM32M8", "RAM32M16", "RAM32X1D", 
    "RAM32X1S", "RAM64M", "RAM364M8", "RAM64M16", "RAM64X1D", "RAM64X1S", "VCC", "GND"
]

feature_to_idx = {f: i for i, f in enumerate(FEATURES)}

# =====================================================================
# 2. GRAPH EXTRACTION FUNCTIONS
# =====================================================================
a=0
def detect_fsm(edif_file):
    global a
    with open(edif_file, "r") as f:
        text = f.read()
    if "FSM_ENCODED_STATES" in text or re.search(r'FSM_sequential', text):
        a=1
    state_properties = re.findall(r'\(property\s+([A-Za-z_][A-Za-z0-9_]*)\s+\((?:string|integer)', text)
    state_names = [
        name for name in state_properties 
        if re.fullmatch(r'S\d+', name) or re.fullmatch(r'F\d+', name) or 
           re.fullmatch(r'U\d+\d+', name) or re.fullmatch(r'D\d+\d+', name)
    ]
    if len(state_names) >= 2:
        a=1

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
                paren_count += line.count("(") - line.count(")")
                if paren_count == 0:
                    inside_net = False
                    fout.write("\n")

def parse_instances(filename):
    with open(filename, "r") as f:
        lines = [x.strip() for x in f]
    instances = {}
    current_name = None
    i = 0
    while i < len(lines):
        if lines[i] == "$":
            current_name = lines[i + 1]
            i += 3
            continue
        if lines[i] == "#":
            cell_type = lines[i + 1]
            if current_name is not None:
                instances[current_name] = cell_type
            i += 3
            continue
        i += 1
    return instances

def build_node_features(instances):
    node_names = list(instances.keys())
    node_to_id = {node: i for i, node in enumerate(node_names)}
    X = np.zeros((len(node_names), len(FEATURES)), dtype=np.float32)
    for node, typ in instances.items():
        idx = node_to_id[node]
        if typ.startswith("LUT") and typ != "LUT6CY":
            X[idx][feature_to_idx["LUT"]] = 1
        elif typ.startswith("MUXF") and typ in feature_to_idx:
            X[idx][feature_to_idx[typ]] = 1
        elif typ in feature_to_idx:
            X[idx][feature_to_idx[typ]] = 1
    return X, node_to_id

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
            balance += line.count("(") - line.count(")")
            block.append(line)
            i += 1
            if balance == 0:
                break
        block_text = "".join(block)
        refs = []
        inst_refs = re.findall(r'\(portref\s+([^\s\(\)]+)\s+\(instanceref\s+([^\)]+)\)\)', block_text)
        for port, inst in inst_refs:
            refs.append(("instance", port, inst))
        member_refs = re.findall(r'\(portref\s+\(member\s+([^\s]+)\s+(\d+)\)\)', block_text)
        for bus, bit in member_refs:
            refs.append(("top", None, f"{bus}_{bit}_"))
        scalar_refs = re.findall(r'\(portref\s+([A-Za-z_][A-Za-z0-9_]*)\)', block_text)
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
                if name.startswith("q_"):
                    sinks.append(name)
                else:
                    driver = name
        if driver is None:
            continue
        for sink in sinks:
            if driver == sink or driver not in node_to_id or sink not in node_to_id:
                continue
            edges.append([node_to_id[driver], node_to_id[sink]])
            
    if not edges:
        return np.array([[], []], dtype=np.int64)
    return np.array(edges).T

# =====================================================================
# 3. SINGLE FILE PROCESSING PIPELINE
# =====================================================================
def convert_single_edn_to_pt(input_edn_path, output_pt_path, label=None):
    if not os.path.exists(input_edn_path):
        raise FileNotFoundError(f"Target EDN file not found: {input_edn_path}")

    # Step 1: FSM Check
    if detect_fsm(input_edn_path):
        print(f"[SKIPPED] FSM logic detected in {input_edn_path}. Aborting conversion.")
        return False

    parsed_path = os.path.join(SCRIPT_DIR, "parsed.text")
    net2_path = os.path.join(SCRIPT_DIR, "net2.txt")

    try:
        # Step 2: Run C Executable Parser -> Generates parsed.text
        subprocess.run([C_PARSER_EXE, input_edn_path], check=True)

        # Step 3: Extract Nets -> Generates net2.txt
        extract_nets(input_edn_path, net2_path)

        # Step 4: Parse Instances and Build Feature Tensors
        instances = parse_instances(parsed_path)
        X, node_to_id = build_node_features(instances)
        edge_index = parse_edges(net2_path, node_to_id)

        # Step 5: Construct PyTorch Data Object
        data_dict = {
            "x": torch.tensor(X, dtype=torch.float),
            "edge_index": torch.tensor(edge_index, dtype=torch.long)
        }

        if label is not None:
            data_dict["y"] = torch.tensor([label], dtype=torch.long)

        # Step 6: Save PyTorch Matrix to .pt
        os.makedirs(os.path.dirname(os.path.abspath(output_pt_path)), exist_ok=True)
        torch.save(data_dict, output_pt_path)
        print(f"[SUCCESS] Saved PyTorch Graph File: {output_pt_path}")
        return True

    finally:
        # Step 7: Clean up temporary files
        if os.path.exists(parsed_path): 
            os.remove(parsed_path)
        if os.path.exists(net2_path): 
            os.remove(net2_path)

# =====================================================================
# EXECUTION EXAMPLE
# =====================================================================
if __name__ == "__main__":
    single_edn = filename
    output_pt = r"./design.pt"
    
    # Optional: Pass an integer label if you want to include target labels ('y') in the saved dict
    target_label = 2  

    convert_single_edn_to_pt(single_edn, output_pt)
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_mean_pool

# ==========================================
# 1. EXACT GNN ARCHITECTURE FROM TRAINING
# ==========================================
class Netlist(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super().__init__()
        self.gin1 = GINConv(nn.Sequential(
            nn.Linear(num_features, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        ))
        self.gin2 = GINConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        ))
        self.gin3 = GINConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        ))
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch, num_graphs=None):
        x = F.relu(self.gin1(x, edge_index))
        x = F.relu(self.gin2(x, edge_index))
        x = F.relu(self.gin3(x, edge_index))
        x = global_mean_pool(x, batch, size=num_graphs)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.fc2(x)

# Map numeric prediction indices back to class names
TOP_LEVEL_LABELS = {
    0: "Combinational",
    1: "Counter",
    2: "Shift Register",
    3: "Register Bank" # Adjust mapping to match your exact training dataset indices
}
REGISTER_BANK_LABELS = {
    0: "Register",
    1: "Register Bank",
    2: "Register File",
    3: "Single Port RAM",
    4: "Simple Dual Port RAM",
    5: "True Dual Port RAM",
    6: "ROM",
    7: "FIFO",
    8: "STACK"
}
SHIFT_REGISTER_LABELS = {
        0: "PIPO",
        1: "PISO",
        2: "SIPO",
        3: "SISO",
        4: "Universal",
        5: "Johnson",
        6: "RING",
        7: "Fibonacci LFSR",
        8: "Galois LFSR",
        9: "SCAN"
}
FSM_LABELS = {
        0: "MEALY",
        1: "MOORE"
}
COUNTER_LABELS = {
        0: "Sync & Async",
        1: "BCD",
        2: "Gray",
        3: "Mod-N",
        4: "One-hot"
}
# ==========================================
# 2. INFERENCE FUNCTION FOR SINGLE .PT FILE
# ==========================================
def predict_single_graph(pt_file_path, model_path="Netlist_classifier.pth", hidden_dim=64, num_classes=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load graph object/dictionary from disk
    data_dict = torch.load(pt_file_path, map_location="cpu")
    
    # Handle both PyG Data objects and standard dictionaries
    if isinstance(data_dict, Data):
        x = data_dict.x
        edge_index = data_dict.edge_index
    else:
        x = data_dict["x"]
        edge_index = data_dict["edge_index"]

    # Single-graph batching: assign all nodes in this graph to batch index 0
    batch = torch.zeros(x.size(0), dtype=torch.long)

    # Move tensors to device
    x = x.to(device)
    edge_index = edge_index.to(device)
    batch = batch.to(device)

    # Initialize model matching your training hyperparameters
    num_features = x.size(1)
    model = Netlist(num_features=num_features, hidden_dim=hidden_dim, num_classes=num_classes).to(device)

    # Load state_dict into the instantiated model
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Pass single graph through the network
    with torch.no_grad():
        logits = model(x, edge_index, batch)
        probs = F.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()

    predicted_label = TOP_LEVEL_LABELS.get(pred_idx, f"Class_{pred_idx}")
    return predicted_label, confidence, pred_idx

def predict_register_bank(pt_file_path, model_path="Register_bank_classifier.pth", hidden_dim=64, num_classes=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load graph object/dictionary from disk
    data_dict = torch.load(pt_file_path, map_location="cpu")
    
    # Handle both PyG Data objects and standard dictionaries
    if isinstance(data_dict, Data):
        x = data_dict.x
        edge_index = data_dict.edge_index
    else:
        x = data_dict["x"]
        edge_index = data_dict["edge_index"]

    # Single-graph batching: assign all nodes in this graph to batch index 0
    batch = torch.zeros(x.size(0), dtype=torch.long)

    # Move tensors to device
    x = x.to(device)
    edge_index = edge_index.to(device)
    batch = batch.to(device)

    # Initialize model matching your training hyperparameters
    num_features = x.size(1)
    model = Netlist(num_features=num_features, hidden_dim=hidden_dim, num_classes=num_classes).to(device)

    # Load state_dict into the instantiated model
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Pass single graph through the network
    with torch.no_grad():
        logits = model(x, edge_index, batch)
        probs = F.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()

    predicted_label = REGISTER_BANK_LABELS.get(pred_idx, f"Class_{pred_idx}")
    return predicted_label, confidence, pred_idx

def predict_shift_register(pt_file_path, model_path="Shift_reg2_classifier.pth", hidden_dim=64, num_classes=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load graph object/dictionary from disk
    data_dict = torch.load(pt_file_path, map_location="cpu")
    
    # Handle both PyG Data objects and standard dictionaries
    if isinstance(data_dict, Data):
        x = data_dict.x
        edge_index = data_dict.edge_index
    else:
        x = data_dict["x"]
        edge_index = data_dict["edge_index"]

    # Single-graph batching: assign all nodes in this graph to batch index 0
    batch = torch.zeros(x.size(0), dtype=torch.long)

    # Move tensors to device
    x = x.to(device)
    edge_index = edge_index.to(device)
    batch = batch.to(device)

    # Initialize model matching your training hyperparameters
    num_features = x.size(1)
    model = Netlist(num_features=num_features, hidden_dim=hidden_dim, num_classes=num_classes).to(device)

    # Load state_dict into the instantiated model
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Pass single graph through the network
    with torch.no_grad():
        logits = model(x, edge_index, batch)
        probs = F.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()

    predicted_label = SHIFT_REGISTER_LABELS.get(pred_idx, f"Class_{pred_idx}")
    return predicted_label, confidence, pred_idx


def predict_FSM(pt_file_path, model_path="Fsm_classifier.pth", hidden_dim=64, num_classes=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load graph object/dictionary from disk
    data_dict = torch.load(pt_file_path, map_location="cpu")
    
    # Handle both PyG Data objects and standard dictionaries
    if isinstance(data_dict, Data):
        x = data_dict.x
        edge_index = data_dict.edge_index
    else:
        x = data_dict["x"]
        edge_index = data_dict["edge_index"]

    # Single-graph batching: assign all nodes in this graph to batch index 0
    batch = torch.zeros(x.size(0), dtype=torch.long)

    # Move tensors to device
    x = x.to(device)
    edge_index = edge_index.to(device)
    batch = batch.to(device)

    # Initialize model matching your training hyperparameters
    num_features = x.size(1)
    model = Netlist(num_features=num_features, hidden_dim=hidden_dim, num_classes=num_classes).to(device)

    # Load state_dict into the instantiated model
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Pass single graph through the network
    with torch.no_grad():
        logits = model(x, edge_index, batch)
        probs = F.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()

    predicted_label = FSM_LABELS.get(pred_idx, f"Class_{pred_idx}")
    return predicted_label, confidence, pred_idx

def predict_counter(pt_file_path, model_path="Counter_classifier.pth", hidden_dim=64, num_classes=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load graph object/dictionary from disk
    data_dict = torch.load(pt_file_path, map_location="cpu")
    
    # Handle both PyG Data objects and standard dictionaries
    if isinstance(data_dict, Data):
        x = data_dict.x
        edge_index = data_dict.edge_index
    else:
        x = data_dict["x"]
        edge_index = data_dict["edge_index"]

    # Single-graph batching: assign all nodes in this graph to batch index 0
    batch = torch.zeros(x.size(0), dtype=torch.long)

    # Move tensors to device
    x = x.to(device)
    edge_index = edge_index.to(device)
    batch = batch.to(device)

    # Initialize model matching your training hyperparameters
    num_features = x.size(1)
    model = Netlist(num_features=num_features, hidden_dim=hidden_dim, num_classes=num_classes).to(device)

    # Load state_dict into the instantiated model
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Pass single graph through the network
    with torch.no_grad():
        logits = model(x, edge_index, batch)
        probs = F.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()

    predicted_label = COUNTER_LABELS.get(pred_idx, f"Class_{pred_idx}")
    return predicted_label, confidence, pred_idx
import os
import re
import sys
from pathlib import Path


# ============================================================
# 1. FILE & UTILITY HELPERS
# ============================================================
def read_edn(filename):
    """Read EDN file content safely."""
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filename}")
    return path.read_text(errors="ignore")


# ============================================================
# 2. FSM METADATA PARSERS
# ============================================================
def extract_encoded_states(text):
    """Extract FSM_ENCODED_STATES mapping state names to binary strings."""
    pattern = re.compile(
        r'FSM_ENCODED_STATES\s+\(string\s+"([^"]+)"\)',
        re.IGNORECASE | re.DOTALL
    )
    matches = pattern.findall(text)
    states = {}
    for encoded in matches:
        entries = re.split(r'\s*,\s*', encoded)
        for entry in entries:
            entry = entry.strip()
            if ":" in entry:
                name, value = entry.split(":", 1)
                if name.strip() and value.strip():
                    states[name.strip()] = value.strip()
    return states


def find_fsm_registers(text):
    """Find register primitives associated with state registers."""
    registers = []
    primitive_pattern = re.compile(
        r'\b(FDRE|FDCE|FDPE|FDSE|FD|FDR|FDC|FDE)\b',
        re.IGNORECASE
    )
    for match in primitive_pattern.finditer(text):
        primitive = match.group(1)
        before = text[max(0, match.start() - 1000):match.start()]
        instance_matches = re.findall(
            r'\(instance\s+(?:\(rename\s+)?"?([^"\s\)]+)',
            before,
            re.IGNORECASE
        )
        if instance_matches:
            registers.append({"name": instance_matches[-1], "primitive": primitive})
    return registers


def extract_fsm_properties(text):
    """Extract general Vivado FSM properties."""
    properties = {}
    property_pattern = re.compile(
        r'\(property\s+([^\s\)]+)\s+\((?:string|boolean|integer)\s+"?([^"\)]*)"?\s*\)',
        re.IGNORECASE | re.DOTALL
    )
    for name, value in property_pattern.findall(text):
        if "FSM" in name.upper():
            properties[name.strip()] = value.strip()
    return properties


def determine_encoding(states):
    """Identify state encoding scheme (One-Hot, Binary, Gray, Custom)."""
    if not states:
        return "Unknown"
    encodings = [re.sub(r'[^01]', '', e) for e in states.values()]
    if not encodings:
        return "Unknown"
    if all(e.count("1") == 1 for e in encodings):
        return "One-Hot"
    try:
        values = [int(e, 2) for e in encodings]
    except ValueError:
        return "Unknown"
    if sorted(values) == list(range(len(values))):
        return "Binary"
    if len(values) > 1:
        gray = True
        for a, b in zip(values, values[1:]):
            diff = a ^ b
            if diff == 0 or (diff & (diff - 1)) != 0:
                gray = False
                break
        if gray:
            return "Gray"
    return "Custom"


def determine_state_width(states):
    """Calculate FSM state bit width."""
    if not states:
        return None
    widths = [len(re.sub(r'[^01]', '', e)) for e in states.values() if re.sub(r'[^01]', '', e)]
    return max(widths) if widths else None


def find_initial_values(text, states):
    """Find INIT values assigned to state registers."""
    results = []
    instance_pattern = re.compile(
        r'\(instance\s+(.*?)\(property\s+INIT\s+\((?:string|binary|integer)\s+"?([^"\)]*)"?\)',
        re.IGNORECASE | re.DOTALL
    )
    for match in instance_pattern.finditer(text):
        block, init = match.group(1), match.group(2)
        name_match = re.search(r'\(rename\s+([^\s\)]+)\s+"([^"]+)"\)', block, re.IGNORECASE)
        name = name_match.group(2) if name_match else "unknown"
        if "FSM" in name.upper() or "STATE" in name.upper():
            results.append({"name": name, "init": init.strip()})
    return results


# ============================================================
# 3. STATE TRANSITION TABLE (STT) RECONSTRUCTION
# ============================================================
def parse_luts_and_nets(text):
    """Parses LUT primitives and INIT hex values from EDN text."""
    luts = {}
    lut_pattern = re.compile(
        r'\(instance\s+(?:\(rename\s+)?([^\s\)\"]+)"?\s+.*?'
        r'\(cellRef\s+(LUT[1-6])\b.*?'
        r'\(property\s+INIT\s+\((?:string|hex)\s+"([^"]+)"\)',
        re.IGNORECASE | re.DOTALL
    )
    for match in lut_pattern.finditer(text):
        lut_name, lut_type, init_hex = match.groups()
        lut_name = lut_name.strip('"')
        lut_block = match.group(0)
        port_refs = re.findall(r'\(portRef\s+(I[0-5]|O)\s+\(instRef\s+([^\s\)]+)\)\)', lut_block)
        inputs = {port: source_inst.strip('"') for port, source_inst in port_refs if port != 'O'}
        luts[lut_name] = {
            "type": lut_type,
            "init": init_hex.replace("h", "").replace("'", ""),
            "inputs": inputs
        }
    return luts


def eval_lut(init_hex, input_bits):
    """Evaluates a Xilinx LUT INIT hex value for binary input bit tuple."""
    try:
        init_val = int(init_hex, 16)
    except ValueError:
        return 0
    index = 0
    for i, bit in enumerate(input_bits):
        index |= (bit << i)
    return (init_val >> index) & 1


def extract_state_transition_table(text, states):
    """Reconstructs Next State transitions from LUT dynamic truth tables."""
    if not states:
        return []

    reverse_states = {re.sub(r'[^01]', '', val): name for name, val in states.items()}
    sample_encoding = list(reverse_states.keys())[0]
    num_state_bits = len(sample_encoding)
    luts = parse_luts_and_nets(text)

    ff_drivers = {}
    for bit_idx in range(num_state_bits):
        pattern = re.compile(
            rf'\(instance\s+(?:\(rename\s+)?([^\s\)\"]*{bit_idx}[^\s\)\"]*)"?\s+.*?'
            rf'\(cellRef\s+(FDRE|FDCE|FDPE|FDSE)',
            re.IGNORECASE | re.DOTALL
        )
        if pattern.search(text):
            for lut_name in luts.keys():
                ff_drivers[bit_idx] = lut_name
                break

    stt = []
    for current_bin, current_name in reverse_states.items():
        curr_bits = [int(b) for b in current_bin]
        for ext_input in [0, 1]:
            next_bits = []
            for bit_idx in range(num_state_bits):
                if bit_idx in ff_drivers and ff_drivers[bit_idx] in luts:
                    lut = luts[ff_drivers[bit_idx]]
                    input_vals = [ext_input] + curr_bits[:5]
                    next_bit = eval_lut(lut["init"], input_vals)
                else:
                    next_bit = curr_bits[bit_idx]
                next_bits.append(str(next_bit))
            next_bin = "".join(next_bits)
            stt.append({
                "current_state": current_name,
                "current_bin": current_bin,
                "input": ext_input,
                "next_state": reverse_states.get(next_bin, f"Unknown ({next_bin})"),
                "next_bin": next_bin
            })
    return stt


# ============================================================
# 4. MAIN PIPELINE ENTRY POINT
# ============================================================
def analyze_fsm_edn(filename):
    """Master function to analyze EDN file and print FSM & STT breakdown."""
    print("=" * 70)
    #print("EDN FSM METADATA & STATE TRANSITION EXTRACTOR")
    print("=" * 70)
    print(f"\nAnalyzing EDN File: {filename}")

    text = read_edn(filename)
    print(f"File Size: {len(text):,} characters")

    # 1. Properties
    properties = extract_fsm_properties(text)
    print("\n[FSM PROPERTIES]")
    print("-" * 70)
    if properties:
        for k, v in properties.items():
            print(f"{k:<35}: {v}")
    else:
        print("No explicit FSM properties found.")

    # 2. Encoded States
    states = extract_encoded_states(text)
    print("\n[ENCODED STATES]")
    print("-" * 70)
    if states:
        for name, encoding in states.items():
            print(f"{name:<25} -> {encoding}")
        print(f"\nTotal States : {len(states)}")
        print(f"State Width  : {determine_state_width(states)} bits")
        print(f"Encoding Type: {determine_encoding(states)}")
    else:
        print("No explicit FSM_ENCODED_STATES found.")

    # 3. State Registers
    registers = find_fsm_registers(text)
    print("\n[SEQUENTIAL ELEMENTS]")
    print("-" * 70)
    seen = set()
    unique_regs = [r for r in registers if not (r["name"], r["primitive"]) in seen and not seen.add((r["name"], r["primitive"]))]
    for reg in unique_regs:
        print(f'{reg["name"]:<45} {reg["primitive"]}')

    # 4. State Transition Table (STT)
    print("\n[STATE TRANSITION TABLE]")
    print("-" * 70)
    stt = extract_state_transition_table(text, states)
    if stt:
        print(f"{'Current State':<20} | {'Encoding':<10} | {'Input':<6} | {'Next State':<20} | {'Next Encoding':<10}")
        print("-" * 75)
        for row in stt:
            print(
                f"{row['current_state']:<20} | "
                f"{row['current_bin']:<10} | "
                f"{row['input']:<6} | "
                f"{row['next_state']:<20} | "
                f"{row['next_bin']:<10}"
            )
    else:
        print("Unable to reconstruct STT directly from LUT drivers.")

    print("\n" + "=" * 70)
    return stt
#!/usr/bin/env python3
"""
EDIF shift-register metadata extractor.

idx mapping:
    0 -> PIPO
    1 -> PISO
    2 -> SIPO
    3 -> SISO
    4 -> UNIVERSAL
    5 -> JOHNSON
    6 -> FIBONACCI
    7 -> GALOIS
    8 -> SCAN

Usage:
    python edif_metadata_extractor.py --input "All_netllists"
    python edif_metadata_extractor.py --input "All_netllists.zip"
    python edif_metadata_extractor.py --input "All_netllists.zip" --idx 3
"""

import argparse
import csv
import json
import re
import zipfile
from pathlib import Path


IDX_TO_TYPE = {
    0: "PIPO",
    1: "PISO",
    2: "SIPO",
    3: "SISO",
    4: "UNIVERSAL",
    5: "JOHNSON",
    6: "FIBONACCI",
    7: "GALOIS",
    8: "SCAN",
}

PREFIX_TO_TYPE = {
    "pipo": (0, "PIPO"),
    "piso": (1, "PISO"),
    "sipo": (2, "SIPO"),
    "siso": (3, "SISO"),
    "usr": (4, "UNIVERSAL"),
    "johnson": (5, "JOHNSON"),
    "fibonacci_lfsr": (6, "FIBONACCI"),
    "galois_lfsr": (7, "GALOIS"),
    "scan_register": (8, "SCAN"),
}

FF_RESET_MAP = {
    "FDCE": ("ASYNC", "CLEAR"),
    "FDPE": ("ASYNC", "PRESET"),
    "FDRE": ("SYNC", "RESET"),
    "FDSE": ("SYNC", "SET"),
}

WIDTHS = (2, 4, 8, 16, 32, 64)


def read_sources(input_path):
    """Read EDIF files from either a directory or ZIP."""
    path = Path(input_path)

    if path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            for name in zf.namelist():
                if name.lower().endswith(".edn"):
                    yield name, zf.read(name).decode("utf-8", errors="ignore")
        return

    if path.is_dir():
        for file_path in path.rglob("*.edn"):
            yield str(file_path), file_path.read_text(
                encoding="utf-8", errors="ignore"
            )
        return

    raise FileNotFoundError(
        f"Input must be an EDIF directory or a .zip file: {input_path}"
    )


def classify_filename(source_name):
    """Map an EDIF filename to the required idx/type."""
    stem = Path(source_name).stem.lower()

    for prefix in sorted(PREFIX_TO_TYPE, key=len, reverse=True):
        if stem == prefix or stem.startswith(prefix + "_"):
            return PREFIX_TO_TYPE[prefix]

    return None, None


def extract_width(stem):
    """Extract widths such as 2, 4, 8, 16, 32 or 64."""
    for width in WIDTHS:
        if re.search(rf"_{width}_", stem.lower()):
            return width

    match = re.search(r"_(\d+)(?:_|$)", stem)
    return int(match.group(1)) if match else None


def get_balanced_blocks(text, keyword):
    """
    Extract balanced EDIF blocks such as:
        (net ... )
    This avoids problems caused by nested parentheses.
    """
    blocks = []

    for match in re.finditer(rf"\({re.escape(keyword)}\s", text):
        start = match.start()
        depth = 0
        in_string = False
        escaped = False

        for i in range(start, len(text)):
            ch = text[i]

            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    blocks.append(text[start:i + 1])
                    break

    return blocks


def extract_interface(text):
    """Extract top-level work-cell ports and their widths."""
    work_pos = text.find("(Library work")
    if work_pos < 0:
        work_pos = 0

    interface_pos = text.find("(interface", work_pos)
    contents_pos = text.find("(contents", interface_pos)

    if interface_pos < 0:
        return {}

    interface_text = text[
        interface_pos:
        contents_pos if contents_pos > 0 else interface_pos + 10000
    ]

    ports = []

    # Scalar ports
    for m in re.finditer(
        r"\(port\s+([A-Za-z_][\w$]*)\s+\(direction\s+(INPUT|OUTPUT)\)\)",
        interface_text,
        re.I,
    ):
        ports.append({
            "name": m.group(1),
            "direction": m.group(2).upper(),
            "width": 1,
        })

    # Array ports, e.g. q[3:0]
    for m in re.finditer(
        r'\(port\s+\(array\s+\(rename\s+\w+\s+"([^"]+)"\)\s+(\d+)\)'
        r'\s+\(direction\s+(INPUT|OUTPUT)\)\)',
        interface_text,
        re.I,
    ):
        ports.append({
            "name": m.group(1),
            "direction": m.group(3).upper(),
            "width": int(m.group(2)),
        })

    return {p["name"]: p for p in ports}


def extract_ff_info(text):
    """Extract q_reg flip-flop primitives and reset style."""
    pattern = re.compile(
        r'\(instance\s+(?:\(rename\s+"?[^"]+"?\s+"?([^"]+)"?\)|([^\s()]+))'
        r'.*?\(cellref\s+(FDCE|FDPE|FDRE|FDSE)\s+\(libraryref\s+hdi_primitives\)\)',
        re.S | re.I,
    )

    ff_types = []

    for a, b, ff in pattern.findall(text):
        name = a or b
        if "q_reg" in name:
            ff_types.append(ff.upper())

    counts = {}
    for ff in ff_types:
        counts[ff] = counts.get(ff, 0) + 1

    reset_modes = {
        FF_RESET_MAP[ff][0] for ff in ff_types if ff in FF_RESET_MAP
    }
    reset_signals = {
        FF_RESET_MAP[ff][1] for ff in ff_types if ff in FF_RESET_MAP
    }

    return {
        "ff_count": len(ff_types),
        "ff_types": sorted(counts),
        "ff_counts": counts,
        "reset_mode": (
            next(iter(reset_modes))
            if len(reset_modes) == 1
            else "MIXED" if reset_modes else None
        ),
        "reset_primitive": ",".join(sorted(reset_signals)),
    }


def extract_enable_info(text):
    """
    Enabled files have a top-level 'en' input and the CE path eventually
    originates from en_IBUF. No-enable files normally have CE tied to VCC.
    """
    ce_blocks = [
        b for b in get_balanced_blocks(text, "net")
        if re.search(
            r"\(portref\s+CE\s+\(instanceref\s+q_reg_",
            b,
            re.I,
        )
    ]

    if not ce_blocks:
        return None, "NO_Q_REG_CE_NET_FOUND", None

    if re.search(
        r"\(port\s+en\s+\(direction\s+INPUT\)\)",
        text,
        re.I,
    ):
        return True, "TOP_LEVEL_EN_PORT", "en/en_IBUF"

    for block in ce_blocks:
        if re.search(
            r"instanceref\s+(?:VCC(?:_\d+)?|GND)|"
            r"<const[01]>|"
            r"portref\s+P\s+\(instanceref\s+VCC",
            block,
            re.I,
        ):
            return False, "CE_TIED_TO_CONSTANT", "VCC/CONST"

    return False, "NO_TOP_LEVEL_EN_PORT", None


def extract_luts(text):
    """Extract LUT primitive counts and INIT values."""
    pattern = re.compile(
        r'\(instance\s+.*?\(cellref\s+(LUT\d+)\s+\(libraryref hdi_primitives\)\)'
        r'(.*?)(?=\(instance|\(net|\(comment "Reference To The Cell Of Highest Level"|$)',
        re.S | re.I,
    )

    luts = []

    for m in pattern.finditer(text):
        lut_type = m.group(1).upper()
        body = m.group(2)

        init_match = re.search(
            r'\(property\s+INIT\s+\(string\s+"([^"]+)"\)\)',
            body,
            re.I,
        )

        luts.append({
            "type": lut_type,
            "init": init_match.group(1) if init_match else None,
        })

    counts = {}
    for lut in luts:
        counts[lut["type"]] = counts.get(lut["type"], 0) + 1

    return luts, counts


def find_q_stage(net_block):
    match = re.search(r"instanceref\s+q_reg_(\d+)", net_block, re.I)
    return int(match.group(1)) if match else None


def extract_serial_stages(text):
    """Find q stage connected to serial input/output."""
    result = {
        "serial_in_stage": None,
        "serial_out_stage": None,
    }

    for block in get_balanced_blocks(text, "net"):
        if "serial_in_IBUF_inst" in block:
            stage = find_q_stage(block)
            if stage is not None:
                result["serial_in_stage"] = stage

        if "serial_out_OBUF_inst" in block:
            stage = find_q_stage(block)
            if stage is not None:
                result["serial_out_stage"] = stage

    return result


def infer_direction(width, serial_in_stage, serial_out_stage):
    """
    Direction used by the supplied files:

      LEFT:
          serial input -> q[0]
          serial output <- q[width-1]

      RIGHT:
          serial input -> q[width-1]
          serial output <- q[0]
    """
    if width is None:
        return None

    if serial_out_stage is not None:
        if serial_out_stage == width - 1:
            return "LEFT"
        if serial_out_stage == 0:
            return "RIGHT"

    if serial_in_stage is not None:
        if serial_in_stage == 0:
            return "LEFT"
        if serial_in_stage == width - 1:
            return "RIGHT"

    return None


def extract_clock_ce_type(text):
    match = re.search(
        r'\(property\s+CE_TYPE\s+\(string\s+"([^"]+)"\)\)',
        text,
        re.I,
    )
    return match.group(1).upper() if match else None


def extract_metadata(source_name, text, forced_idx=None):
    idx, register_type = classify_filename(source_name)

    if register_type is None:
        return None

    if forced_idx is not None:
        idx = forced_idx
        register_type = IDX_TO_TYPE[forced_idx]

    stem = Path(source_name).stem
    width = extract_width(stem)
    ports = extract_interface(text)
    ff = extract_ff_info(text)
    enable, enable_detection, enable_net = extract_enable_info(text)
    luts, lut_counts = extract_luts(text)
    serial = extract_serial_stages(text)

    direction = infer_direction(
        width,
        serial["serial_in_stage"],
        serial["serial_out_stage"],
    )

    stem_lower = stem.lower()
    filename_direction = (
        "LEFT" if "_left_" in stem_lower
        else "RIGHT" if "_right_" in stem_lower
        else None
    )

    port_names = set(ports)

    features = {
        "has_enable": "en" in port_names,
        "has_reset": "rst" in port_names,
        "has_load": "load" in port_names,
        "has_scan_enable": "scan_enable" in port_names,
        "has_scan_in": "scan_in" in port_names,
        "has_scan_out": "scan_out" in port_names,
        "has_serial_in": "serial_in" in port_names,
        "has_serial_out": "serial_out" in port_names,
        "has_serial_left": "serial_left" in port_names,
        "has_serial_right": "serial_right" in port_names,
        "has_select": "sel" in port_names,
        "has_parallel_d": any(
            name.startswith("d[") for name in port_names
        ),
        "has_parallel_q": any(
            name.startswith("q[") for name in port_names
        ),
    }

    return {
        "idx": idx,
        "register_type": register_type,
        "file": source_name,
        "width": width,

        # Left/right metadata
        "direction_detected": direction,
        "direction_from_filename": filename_direction,
        "direction_matches_filename": (
            None if direction is None or filename_direction is None
            else direction == filename_direction
        ),

        # Enable/reset metadata
        "enable": enable,
        "enable_detection": enable_detection,
        "enable_net": enable_net,
        "reset_mode": ff["reset_mode"],
        "reset_primitive": ff["reset_primitive"],

        # Register implementation
        "ff_count": ff["ff_count"],
        "ff_types": ",".join(ff["ff_types"]),
        "ff_counts": ff["ff_counts"],

        # Structural information
        "serial_in_stage": serial["serial_in_stage"],
        "serial_out_stage": serial["serial_out_stage"],
        "lut_counts": lut_counts,
        "lut_init": [x["init"] for x in luts if x["init"]],
        "clock_ce_type": extract_clock_ce_type(text),
        "ports": ports,
        "features": features,
    }


def flatten_for_csv(meta):
    """Flatten nested fields so they can be written to CSV."""
    row = dict(meta)
    row["ff_counts"] = json.dumps(meta["ff_counts"], sort_keys=True)
    row["lut_counts"] = json.dumps(meta["lut_counts"], sort_keys=True)
    row["lut_init"] = ";".join(meta["lut_init"])
    row["ports"] = json.dumps(meta["ports"], sort_keys=True)
    row["features"] = json.dumps(meta["features"], sort_keys=True)
    return row


# ============================================================
# USER CONFIGURATION
# ============================================================

# Give the EDN file you want to analyse here.
#filename = "pipo_8_async_enable.edn"


# ============================================================
# MAIN CALLABLE FUNCTION
# ============================================================

def extract_shift_register_metadata(filename, idx):
    """
    Extract metadata from the EDN file specified by the global `filename`.

    idx mapping:
        0 -> PIPO
        1 -> PISO
        2 -> SIPO
        3 -> SISO
        4 -> UNIVERSAL
        5 -> JOHNSON
        6 -> FIBONACCI
        7 -> GALOIS
        8 -> SCAN

    Returns:
        dict containing all extracted metadata.

    Example:
        filename = "pipo_8_left_en.edn"
        metadata = extract_shift_register_metadata(0)
        print(metadata)
    """

    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("Global variable 'filename' must contain an EDN file path.")

    if idx not in IDX_TO_TYPE:
        raise ValueError("idx must be an integer from 0 to 8.")

    path = Path(filename)

    if not path.exists():
        raise FileNotFoundError(f"EDN file not found: {filename}")

    if path.suffix.lower() != ".edn":
        raise ValueError(f"Expected an EDN file, got: {filename}")

    text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    # Use the supplied idx as the register class.
    record = extract_metadata(
        str(path),
        text,
        forced_idx=idx
    )

    if record is None:
        # extract_metadata normally rejects unknown filename prefixes.
        # For the new function interface, we allow any EDN filename and
        # use the supplied idx to determine the register type.
        width = extract_width(path.stem)
        ports = extract_interface(text)
        ff = extract_ff_info(text)
        enable, enable_detection, enable_net = extract_enable_info(text)
        luts, lut_counts = extract_luts(text)
        serial = extract_serial_stages(text)

        direction = infer_direction(
            width,
            serial["serial_in_stage"],
            serial["serial_out_stage"]
        )

        stem_lower = path.stem.lower()

        filename_direction = (
            "LEFT" if "_left_" in stem_lower
            else "RIGHT" if "_right_" in stem_lower
            else None
        )

        port_names = set(ports)

        features = {
            "has_enable": "en" in port_names,
            "has_reset": "rst" in port_names,
            "has_load": "load" in port_names,
            "has_scan_enable": "scan_enable" in port_names,
            "has_scan_in": "scan_in" in port_names,
            "has_scan_out": "scan_out" in port_names,
            "has_serial_in": "serial_in" in port_names,
            "has_serial_out": "serial_out" in port_names,
            "has_serial_left": "serial_left" in port_names,
            "has_serial_right": "serial_right" in port_names,
            "has_select": "sel" in port_names,
            "has_parallel_d": any(
                name.startswith("d[") for name in port_names
            ),
            "has_parallel_q": any(
                name.startswith("q[") for name in port_names
            ),
        }

        record = {
            "idx": idx,
            "register_type": IDX_TO_TYPE[idx],
            "file": str(path),
            "width": width,

            "direction_detected": direction,
            "direction_from_filename": filename_direction,
            "direction_matches_filename": (
                None
                if direction is None or filename_direction is None
                else direction == filename_direction
            ),

            "enable": enable,
            "enable_detection": enable_detection,
            "enable_net": enable_net,
            "reset_mode": ff["reset_mode"],
            "reset_primitive": ff["reset_primitive"],

            "ff_count": ff["ff_count"],
            "ff_types": ",".join(ff["ff_types"]),
            "ff_counts": ff["ff_counts"],

            "serial_in_stage": serial["serial_in_stage"],
            "serial_out_stage": serial["serial_out_stage"],
            "lut_counts": lut_counts,
            "lut_init": [x["init"] for x in luts if x["init"]],
            "clock_ce_type": extract_clock_ce_type(text),
            "ports": ports,
            "features": features,
        }

    return record


# Optional shorter alias
metadata_extractor = extract_shift_register_metadata

if a==1:
    if __name__ == "__main__":
        print(f"Predicted Class Index : FSM")
        print(f"Predicted Label       : 5")
        test_file = "./design.pt"  
        label4, conf4, idx4 = predict_FSM(test_file)
        print(f"Predicted subclass Index : {idx4}")
        print(f"Predicted Label       : {label4}")
        print(f"Confidence Score      : {conf4 * 100:.2f}%")
        analyze_fsm_edn(filename)
else:
    if __name__ == "__main__":
        test_file = "./design.pt"  # Path to target graph file
        
        label, conf, idx = predict_single_graph(test_file)
        print(f"Predicted Class Index : {idx}")
        print(f"Predicted Label       : {label}")
        print(f"Confidence Score      : {conf * 100:.2f}%")
    
        if idx == 2 :
            label2, conf2, idx2 = predict_shift_register(test_file)
            print(f"Predicted Sublass Index : {idx2}")
            print(f"Predicted Label       : {label2}")
            print(f"Confidence Score      : {conf2 * 100:.2f}%")
            metadata = extract_shift_register_metadata(filename,idx2)
            print(metadata)
        elif idx == 3 :
            label3, conf3, idx3 = predict_register_bank(test_file)
            print(f"Predicted Sublass Index : {idx3}")
            print(f"Predicted Label       : {label3}")
            print(f"Confidence Score      : {conf3 * 100:.2f}%")
        elif idx == 1 :
            label1, conf1, idx1 = predict_counter(test_file)
            print(f"Predicted Sublass Index : {idx1}")
            print(f"Predicted Label       : {label1}")
            print(f"Confidence Score      : {conf1 * 100:.2f}%")
"""Hardcoded minimal K8s fault knowledge graph for Phase 3 toy run."""

from __future__ import annotations
import networkx as nx


def build_toy_graph() -> nx.DiGraph:
    """
    ~30 nodes covering the 6 K8s fault categories from Table 3.
    Edge weight = causal strength [0,1].
    """
    G = nx.DiGraph()

    # --- Resource Errors ---
    G.add_node("OOMKilled",         type="fault",   category="Resource")
    G.add_node("MemoryLeak",        type="cause",   category="Resource")
    G.add_node("ResourceQuota",     type="cause",   category="Resource")
    G.add_node("CPUThrottle",       type="fault",   category="Resource")
    G.add_node("LimitRange",        type="cause",   category="Resource")

    # --- Network Errors ---
    G.add_node("DNSFailure",        type="fault",   category="Network")
    G.add_node("CoreDNSCrash",      type="cause",   category="Network")
    G.add_node("IngressMisconfig",  type="fault",   category="Network")
    G.add_node("ServiceEndpoint",   type="cause",   category="Network")
    G.add_node("NetworkPolicy",     type="cause",   category="Network")

    # --- Scheduling Errors ---
    G.add_node("PodPending",        type="fault",   category="Scheduling")
    G.add_node("NodeSelector",      type="cause",   category="Scheduling")
    G.add_node("TaintToleration",   type="cause",   category="Scheduling")
    G.add_node("InsufficientCPU",   type="cause",   category="Scheduling")

    # --- Image Errors ---
    G.add_node("ImagePullBackOff",  type="fault",   category="Image")
    G.add_node("RegistryAuth",      type="cause",   category="Image")
    G.add_node("ImageTag",          type="cause",   category="Image")

    # --- Configuration Errors ---
    G.add_node("CrashLoopBackOff",  type="fault",   category="Configuration")
    G.add_node("ConfigMapMount",    type="cause",   category="Configuration")
    G.add_node("SecretMissing",     type="cause",   category="Configuration")
    G.add_node("EnvVarMissing",     type="cause",   category="Configuration")

    # --- System Errors ---
    G.add_node("NodeNotReady",      type="fault",   category="System")
    G.add_node("KubeletCrash",      type="cause",   category="System")
    G.add_node("DiskPressure",      type="cause",   category="System")
    G.add_node("etcdTimeout",       type="cause",   category="System")

    # --- Resolutions ---
    G.add_node("IncreaseMemLimit",  type="resolution", category="Resource")
    G.add_node("SetResourceLimits", type="resolution", category="Resource")
    G.add_node("RestartCoreDNS",    type="resolution", category="Network")
    G.add_node("FixIngressRules",   type="resolution", category="Network")
    G.add_node("AddToleration",     type="resolution", category="Scheduling")
    G.add_node("AddNodeCapacity",   type="resolution", category="Scheduling")
    G.add_node("FixRegistryCreds",  type="resolution", category="Image")
    G.add_node("FixConfigMount",    type="resolution", category="Configuration")
    G.add_node("RestartKubelet",    type="resolution", category="System")
    G.add_node("FreeDisk",          type="resolution", category="System")

    edges = [
        # Resource
        ("MemoryLeak",      "OOMKilled",        {"type": "causes",      "weight": 0.9}),
        ("ResourceQuota",   "OOMKilled",        {"type": "causes",      "weight": 0.7}),
        ("LimitRange",      "CPUThrottle",      {"type": "causes",      "weight": 0.8}),
        ("OOMKilled",       "IncreaseMemLimit", {"type": "resolves",    "weight": 0.9}),
        ("CPUThrottle",     "SetResourceLimits",{"type": "resolves",    "weight": 0.8}),
        # Network
        ("CoreDNSCrash",    "DNSFailure",       {"type": "causes",      "weight": 0.95}),
        ("NetworkPolicy",   "IngressMisconfig", {"type": "causes",      "weight": 0.7}),
        ("ServiceEndpoint", "IngressMisconfig", {"type": "causes",      "weight": 0.8}),
        ("DNSFailure",      "RestartCoreDNS",   {"type": "resolves",    "weight": 0.9}),
        ("IngressMisconfig","FixIngressRules",  {"type": "resolves",    "weight": 0.85}),
        # Scheduling
        ("NodeSelector",    "PodPending",       {"type": "causes",      "weight": 0.85}),
        ("TaintToleration", "PodPending",       {"type": "causes",      "weight": 0.9}),
        ("InsufficientCPU", "PodPending",       {"type": "causes",      "weight": 0.8}),
        ("PodPending",      "AddToleration",    {"type": "resolves",    "weight": 0.8}),
        ("PodPending",      "AddNodeCapacity",  {"type": "resolves",    "weight": 0.7}),
        # Image
        ("RegistryAuth",    "ImagePullBackOff", {"type": "causes",      "weight": 0.95}),
        ("ImageTag",        "ImagePullBackOff", {"type": "causes",      "weight": 0.85}),
        ("ImagePullBackOff","FixRegistryCreds", {"type": "resolves",    "weight": 0.9}),
        # Configuration
        ("ConfigMapMount",  "CrashLoopBackOff", {"type": "causes",      "weight": 0.8}),
        ("SecretMissing",   "CrashLoopBackOff", {"type": "causes",      "weight": 0.85}),
        ("EnvVarMissing",   "CrashLoopBackOff", {"type": "causes",      "weight": 0.75}),
        ("CrashLoopBackOff","FixConfigMount",   {"type": "resolves",    "weight": 0.8}),
        # System
        ("KubeletCrash",    "NodeNotReady",     {"type": "causes",      "weight": 0.95}),
        ("DiskPressure",    "NodeNotReady",     {"type": "causes",      "weight": 0.85}),
        ("etcdTimeout",     "NodeNotReady",     {"type": "causes",      "weight": 0.75}),
        ("NodeNotReady",    "RestartKubelet",   {"type": "resolves",    "weight": 0.9}),
        ("NodeNotReady",    "FreeDisk",         {"type": "resolves",    "weight": 0.7}),
        # Cross-category cascades
        ("OOMKilled",       "NodeNotReady",     {"type": "cascades_to", "weight": 0.4}),
        ("DNSFailure",      "PodPending",       {"type": "cascades_to", "weight": 0.3}),
    ]

    for src, dst, attrs in edges:
        G.add_edge(src, dst, **attrs)

    return G

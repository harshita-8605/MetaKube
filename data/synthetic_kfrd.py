"""Synthetic Kubernetes Fault Resolution Dataset (KFRD).

Generates problem-attempt-solution triples (Eq 23-26, Appendix C.1).
Paper uses 7,000 samples: 5,000 SFT + 2,000 eval.
"""

from __future__ import annotations
import json
import random
import time
from pathlib import Path

# 30 fault templates across 6 categories (Table 3 from paper)
FAULT_TEMPLATES = [
    # ── Resource Errors ──────────────────────────────────────────────
    {
        "fault_category": "Resource",
        "symptoms": ["Pod OOMKilled", "Memory limit exceeded", "Container restarts increasing"],
        "context": {"namespace": "production", "pod": "api-server-xxx", "resource_type": "Deployment"},
        "actions": [
            "kubectl set resources deployment api-server --limits memory=512Mi",
            "kubectl describe pod api-server-xxx | grep -A5 OOM",
            "kubectl top pod -n production",
        ],
        "outcomes": ["Pod resolved after memory limit increase", "Service restored"],
        "root_cause": "Memory limit too low for workload. Container killed by OOM killer.",
        "prevention": "Set memory requests = 70% of limits. Enable VPA for auto-scaling.",
    },
    {
        "fault_category": "Resource",
        "symptoms": ["CPU throttling detected", "High cpu usage", "Response latency spike"],
        "context": {"namespace": "staging", "pod": "worker-yyy", "resource_type": "Deployment"},
        "actions": [
            "kubectl top pods -n staging",
            "kubectl set resources deployment worker --limits cpu=1000m --requests cpu=500m",
            "kubectl rollout status deployment worker",
        ],
        "outcomes": ["CPU throttling resolved", "Latency normalized to baseline"],
        "root_cause": "CPU limit set too low relative to workload demand.",
        "prevention": "Profile workload CPU usage. Set limits 2x observed peak.",
    },
    {
        "fault_category": "Resource",
        "symptoms": ["PVC mount failed", "Persistent volume claim pending", "Storage quota exceeded"],
        "context": {"namespace": "default", "pod": "db-pod-zzz", "resource_type": "StatefulSet"},
        "actions": [
            "kubectl get pvc -n default",
            "kubectl describe pvc db-pvc",
            "kubectl patch resourcequota storage-quota --patch '{\"spec\":{\"hard\":{\"requests.storage\":\"100Gi\"}}}'",
        ],
        "outcomes": ["PVC bound successfully", "Database pod running"],
        "root_cause": "StorageClass ResourceQuota exhausted. New PVC could not bind.",
        "prevention": "Monitor storage usage with alerting at 80% quota utilization.",
    },
    {
        "fault_category": "Resource",
        "symptoms": ["HPA not scaling", "Metrics server unavailable", "CPU metrics missing"],
        "context": {"namespace": "production", "deployment": "frontend", "resource_type": "HPA"},
        "actions": [
            "kubectl get hpa -n production",
            "kubectl top nodes",
            "kubectl rollout restart deployment metrics-server -n kube-system",
        ],
        "outcomes": ["Metrics server restored", "HPA scaling resumed"],
        "root_cause": "Metrics server crashed, HPA lost CPU metrics, scaling disabled.",
        "prevention": "Add liveness/readiness probes to metrics-server. Set replicas=2.",
    },
    {
        "fault_category": "Resource",
        "symptoms": ["Node memory pressure", "Pods evicted", "DaemonSet pod pending"],
        "context": {"namespace": "kube-system", "node": "worker-1", "resource_type": "Node"},
        "actions": [
            "kubectl describe node worker-1 | grep -A10 Conditions",
            "kubectl drain worker-1 --ignore-daemonsets --delete-emptydir-data",
            "kubectl uncordon worker-1",
        ],
        "outcomes": ["Node pressure relieved", "Evicted pods rescheduled"],
        "root_cause": "Node running too many memory-intensive pods, triggering eviction.",
        "prevention": "Set pod disruption budgets. Use node affinity to spread workloads.",
    },
    # ── Network Errors ────────────────────────────────────────────────
    {
        "fault_category": "Network",
        "symptoms": ["DNS resolution failure", "CoreDNS pod crashing", "Service unreachable by name"],
        "context": {"namespace": "kube-system", "service": "coredns", "resource_type": "Deployment"},
        "actions": [
            "kubectl get pods -n kube-system | grep coredns",
            "kubectl logs -n kube-system deployment/coredns --previous",
            "kubectl rollout restart deployment coredns -n kube-system",
        ],
        "outcomes": ["DNS service restored", "Pods resolve service hostnames correctly"],
        "root_cause": "CoreDNS crashed due to OOM or config error.",
        "prevention": "Increase CoreDNS memory limits. Enable forward caching.",
    },
    {
        "fault_category": "Network",
        "symptoms": ["Ingress returning 502", "Backend service unreachable", "nginx upstream error"],
        "context": {"namespace": "production", "service": "frontend", "resource_type": "Ingress"},
        "actions": [
            "kubectl describe ingress frontend-ingress -n production",
            "kubectl get endpoints frontend-svc -n production",
            "kubectl edit ingress frontend-ingress -n production",
        ],
        "outcomes": ["Ingress routing fixed", "Service accessible via HTTPS"],
        "root_cause": "Service targetPort mismatch: Ingress pointed to wrong backend port.",
        "prevention": "Validate ingress backend servicePort matches Service spec.",
    },
    {
        "fault_category": "Network",
        "symptoms": ["Network policy blocking traffic", "Pods cannot communicate cross-namespace", "Connection refused"],
        "context": {"namespace": "staging", "resource_type": "NetworkPolicy"},
        "actions": [
            "kubectl get networkpolicy -A",
            "kubectl describe networkpolicy deny-all -n staging",
            "kubectl apply -f allow-cross-ns-policy.yaml",
        ],
        "outcomes": ["Cross-namespace traffic allowed", "Service communication restored"],
        "root_cause": "Default deny NetworkPolicy blocked inter-namespace service calls.",
        "prevention": "Document required traffic flows. Apply allow policies before deny-all.",
    },
    {
        "fault_category": "Network",
        "symptoms": ["LoadBalancer pending", "External IP not assigned", "Cloud provider error"],
        "context": {"namespace": "production", "service": "api-lb", "resource_type": "Service"},
        "actions": [
            "kubectl get svc api-lb -n production",
            "kubectl describe svc api-lb -n production",
            "kubectl patch svc api-lb -n production -p '{\"spec\":{\"type\":\"NodePort\"}}'",
        ],
        "outcomes": ["Service accessible via NodePort while investigating cloud LB", "External IP assigned"],
        "root_cause": "Cloud provider quota for LoadBalancer IPs exhausted.",
        "prevention": "Use Ingress controller instead of per-service LoadBalancers.",
    },
    {
        "fault_category": "Network",
        "symptoms": ["CNI plugin crash", "Pod network unavailable", "Flannel/Calico error"],
        "context": {"namespace": "kube-system", "resource_type": "DaemonSet"},
        "actions": [
            "kubectl get pods -n kube-system | grep -E 'calico|flannel|cilium'",
            "kubectl logs -n kube-system daemonset/calico-node",
            "kubectl rollout restart daemonset calico-node -n kube-system",
        ],
        "outcomes": ["CNI plugin restored", "Pod networking functional"],
        "root_cause": "CNI daemonset crashed due to kernel version incompatibility.",
        "prevention": "Pin CNI version to tested kernel version. Run in staging first.",
    },
    # ── Scheduling Errors ─────────────────────────────────────────────
    {
        "fault_category": "Scheduling",
        "symptoms": ["Pod stuck in Pending state", "No nodes available", "Insufficient CPU on all nodes"],
        "context": {"namespace": "default", "pod": "batch-job-zzz", "resource_type": "Job"},
        "actions": [
            "kubectl describe pod batch-job-zzz | grep -A10 Events",
            "kubectl get nodes -o wide",
            "kubectl scale deployment cluster-autoscaler --replicas=1 -n kube-system",
        ],
        "outcomes": ["Autoscaler added node", "Pod scheduled and job completed"],
        "root_cause": "Cluster autoscaler disabled. No node had sufficient CPU for pod request.",
        "prevention": "Enable cluster autoscaler. Set resource requests accurately.",
    },
    {
        "fault_category": "Scheduling",
        "symptoms": ["Pod pending due to taint", "Node taint mismatch", "Toleration missing"],
        "context": {"namespace": "default", "pod": "gpu-workload-aaa", "resource_type": "Deployment"},
        "actions": [
            "kubectl describe node gpu-node-1 | grep -A5 Taints",
            "kubectl patch deployment gpu-workload -p '{\"spec\":{\"template\":{\"spec\":{\"tolerations\":[{\"key\":\"gpu\",\"operator\":\"Equal\",\"value\":\"true\",\"effect\":\"NoSchedule\"}]}}}}'",
            "kubectl get pod gpu-workload-aaa -w",
        ],
        "outcomes": ["Toleration added", "Pod scheduled on GPU node"],
        "root_cause": "GPU node tainted with gpu=true:NoSchedule. Deployment missing toleration.",
        "prevention": "Template GPU deployments with standard GPU toleration from the start.",
    },
    {
        "fault_category": "Scheduling",
        "symptoms": ["Anti-affinity rules blocking pod placement", "Deployment replicas all on same node", "Node affinity violated"],
        "context": {"namespace": "production", "resource_type": "Deployment"},
        "actions": [
            "kubectl describe pod -n production | grep -A20 Affinity",
            "kubectl get nodes --show-labels",
            "kubectl patch deployment web-app -n production -p '{\"spec\":{\"template\":{\"spec\":{\"affinity\":null}}}}'",
        ],
        "outcomes": ["Affinity relaxed", "Pods distributed across nodes"],
        "root_cause": "RequiredDuringScheduling affinity rule too strict for available nodes.",
        "prevention": "Use PreferredDuringScheduling unless strict isolation required.",
    },
    {
        "fault_category": "Scheduling",
        "symptoms": ["PriorityClass eviction", "Low-priority pod evicted", "High-priority pod pending"],
        "context": {"namespace": "default", "resource_type": "PriorityClass"},
        "actions": [
            "kubectl get priorityclass",
            "kubectl describe pod evicted-pod | grep Priority",
            "kubectl edit deployment low-priority-app",
        ],
        "outcomes": ["Priority adjusted", "Critical workload scheduled"],
        "root_cause": "High-priority pod preempted low-priority pod. PriorityClass misconfigured.",
        "prevention": "Assign correct PriorityClass to all workloads. Reserve capacity for critical pods.",
    },
    # ── Image Errors ──────────────────────────────────────────────────
    {
        "fault_category": "Image",
        "symptoms": ["ImagePullBackOff error", "Registry authentication failed", "ErrImagePull"],
        "context": {"namespace": "production", "pod": "app-bbb", "resource_type": "Deployment"},
        "actions": [
            "kubectl create secret docker-registry regcred --docker-server=registry.example.com --docker-username=user --docker-password=pass -n production",
            "kubectl patch deployment app -n production -p '{\"spec\":{\"template\":{\"spec\":{\"imagePullSecrets\":[{\"name\":\"regcred\"}]}}}}'",
            "kubectl rollout status deployment app -n production",
        ],
        "outcomes": ["Registry credentials configured", "Image pulled successfully"],
        "root_cause": "imagePullSecret missing. Registry requires authentication.",
        "prevention": "Use external secrets operator. Rotate registry credentials quarterly.",
    },
    {
        "fault_category": "Image",
        "symptoms": ["Image tag not found", "404 from registry", "Wrong image version"],
        "context": {"namespace": "staging", "pod": "backend-pod", "resource_type": "Deployment"},
        "actions": [
            "kubectl describe pod backend-pod -n staging | grep Image",
            "kubectl set image deployment/backend backend=registry.example.com/backend:v1.2.3 -n staging",
            "kubectl rollout status deployment backend -n staging",
        ],
        "outcomes": ["Correct image tag set", "Deployment running stable version"],
        "root_cause": "Deployment using :latest tag which was overwritten. Pin to explicit SHA or semver.",
        "prevention": "Never use :latest in production. Pin image digest in deployment spec.",
    },
    {
        "fault_category": "Image",
        "symptoms": ["Multi-arch image failure", "amd64/arm64 mismatch", "Container exec format error"],
        "context": {"namespace": "production", "resource_type": "Deployment"},
        "actions": [
            "kubectl get nodes -o jsonpath='{.items[*].status.nodeInfo.architecture}'",
            "docker buildx build --platform linux/amd64,linux/arm64 -t registry/app:latest --push .",
            "kubectl rollout restart deployment app -n production",
        ],
        "outcomes": ["Multi-arch image built and pushed", "Container starts on all node architectures"],
        "root_cause": "Image built only for amd64, but cluster has arm64 nodes.",
        "prevention": "Use buildx multi-arch builds in CI pipeline.",
    },
    # ── Configuration Errors ──────────────────────────────────────────
    {
        "fault_category": "Configuration",
        "symptoms": ["CrashLoopBackOff", "Container exits immediately", "ConfigMap not found"],
        "context": {"namespace": "staging", "pod": "backend-ccc", "resource_type": "Deployment"},
        "actions": [
            "kubectl logs pod/backend-ccc -n staging --previous",
            "kubectl create configmap app-config --from-file=config.yaml -n staging",
            "kubectl rollout restart deployment backend -n staging",
        ],
        "outcomes": ["ConfigMap mounted successfully", "Application started normally"],
        "root_cause": "Application expects /etc/config/config.yaml. ConfigMap not created before deployment.",
        "prevention": "Use Helm hooks or ArgoCD sync waves to ensure ConfigMaps exist before Deployments.",
    },
    {
        "fault_category": "Configuration",
        "symptoms": ["Application crashes on start", "Secret missing", "Environment variable not set"],
        "context": {"namespace": "production", "pod": "auth-service-ddd", "resource_type": "Deployment"},
        "actions": [
            "kubectl logs pod/auth-service-ddd -n production",
            "kubectl create secret generic db-secret --from-literal=password=mysecret -n production",
            "kubectl set env deployment auth-service --from secret/db-secret -n production",
        ],
        "outcomes": ["Secret injected as env var", "Auth service started"],
        "root_cause": "Secret referenced in env.valueFrom.secretKeyRef not created.",
        "prevention": "Use external-secrets-operator. Validate secrets exist in pre-deploy hook.",
    },
    {
        "fault_category": "Configuration",
        "symptoms": ["RBAC permission denied", "403 Forbidden from API", "ServiceAccount missing role"],
        "context": {"namespace": "production", "resource_type": "ServiceAccount"},
        "actions": [
            "kubectl auth can-i get pods --as system:serviceaccount:production:app-sa -n production",
            "kubectl create rolebinding app-pod-reader --role=pod-reader --serviceaccount=production:app-sa -n production",
            "kubectl rollout restart deployment app -n production",
        ],
        "outcomes": ["RBAC binding created", "Application can read pod status"],
        "root_cause": "ServiceAccount lacks Role binding for required API resource.",
        "prevention": "Audit RBAC requirements at design time. Use minimal privilege.",
    },
    {
        "fault_category": "Configuration",
        "symptoms": ["Helm upgrade failed", "Invalid values.yaml", "Chart rendering error"],
        "context": {"namespace": "staging", "resource_type": "Helm"},
        "actions": [
            "helm lint ./chart --values values.yaml",
            "helm template ./chart --values values.yaml | kubectl apply --dry-run=client -f -",
            "helm rollback myrelease 1 -n staging",
        ],
        "outcomes": ["Helm chart syntax fixed", "Release upgraded successfully"],
        "root_cause": "values.yaml referenced undefined template variable. Helm rendering failed.",
        "prevention": "Add helm lint to CI pipeline. Test with helm template --dry-run.",
    },
    {
        "fault_category": "Configuration",
        "symptoms": ["Webhook admission error", "MutatingWebhookConfiguration blocking", "ValidatingWebhookConfiguration rejecting"],
        "context": {"namespace": "kube-system", "resource_type": "MutatingWebhookConfiguration"},
        "actions": [
            "kubectl get mutatingwebhookconfigurations",
            "kubectl describe mutatingwebhookconfiguration istio-sidecar-injector",
            "kubectl delete mutatingwebhookconfiguration problematic-webhook",
        ],
        "outcomes": ["Webhook removed", "Pod creation unblocked"],
        "root_cause": "Admission webhook timeout. Webhook server down but failurePolicy=Fail.",
        "prevention": "Set failurePolicy=Ignore for non-critical webhooks. Add webhook health checks.",
    },
    # ── System Errors ─────────────────────────────────────────────────
    {
        "fault_category": "System",
        "symptoms": ["Node NotReady", "Kubelet not responding", "Disk pressure on node"],
        "context": {"namespace": "kube-system", "node": "worker-node-1", "resource_type": "Node"},
        "actions": [
            "kubectl describe node worker-node-1 | grep -A10 Conditions",
            "kubectl drain worker-node-1 --ignore-daemonsets --delete-emptydir-data",
            "ssh worker-node-1 'sudo systemctl restart kubelet && df -h'",
            "kubectl uncordon worker-node-1",
        ],
        "outcomes": ["Kubelet restarted", "Disk cleared", "Node returned to Ready state"],
        "root_cause": "Disk full due to container log accumulation. Kubelet failed health check.",
        "prevention": "Configure log rotation. Set container log max-size in kubelet config.",
    },
    {
        "fault_category": "System",
        "symptoms": ["etcd cluster degraded", "API server slow", "Leader election timeout"],
        "context": {"namespace": "kube-system", "resource_type": "StatefulSet"},
        "actions": [
            "kubectl exec -n kube-system etcd-master -- etcdctl endpoint health",
            "kubectl exec -n kube-system etcd-master -- etcdctl member list",
            "kubectl delete pod etcd-master-1 -n kube-system",
        ],
        "outcomes": ["etcd leader re-elected", "API server responsive"],
        "root_cause": "etcd member lost quorum due to network partition. Leader election stalled.",
        "prevention": "Use odd number of etcd members (3 or 5). Monitor etcd heartbeat latency.",
    },
    {
        "fault_category": "System",
        "symptoms": ["Certificate expired", "TLS handshake failure", "API server auth error"],
        "context": {"namespace": "kube-system", "resource_type": "Certificate"},
        "actions": [
            "kubeadm certs check-expiration",
            "kubeadm certs renew all",
            "sudo systemctl restart kubelet",
        ],
        "outcomes": ["Certificates renewed", "API server accessible with valid TLS"],
        "root_cause": "Kubernetes certificates expired (default 1-year rotation not done).",
        "prevention": "Automate cert rotation with cert-manager. Alert 30 days before expiry.",
    },
    {
        "fault_category": "System",
        "symptoms": ["Kube-proxy crash", "Service IP unreachable", "iptables rules missing"],
        "context": {"namespace": "kube-system", "resource_type": "DaemonSet"},
        "actions": [
            "kubectl get pods -n kube-system | grep kube-proxy",
            "kubectl logs -n kube-system daemonset/kube-proxy",
            "kubectl rollout restart daemonset kube-proxy -n kube-system",
        ],
        "outcomes": ["kube-proxy restarted", "iptables rules restored", "Service IPs reachable"],
        "root_cause": "kube-proxy OOMKilled. iptables rules cleared causing service IP loss.",
        "prevention": "Increase kube-proxy memory limits. Monitor iptables rule count.",
    },
    {
        "fault_category": "System",
        "symptoms": ["Container runtime crash", "containerd/dockerd not running", "Failed to create pod sandbox"],
        "context": {"namespace": "kube-system", "node": "worker-2", "resource_type": "Node"},
        "actions": [
            "ssh worker-2 'sudo systemctl status containerd'",
            "ssh worker-2 'sudo systemctl restart containerd'",
            "kubectl get pods --field-selector spec.nodeName=worker-2",
        ],
        "outcomes": ["containerd restarted", "Pods recreated on node"],
        "root_cause": "containerd crashed due to disk full. Socket file corrupted.",
        "prevention": "Monitor container runtime health. Separate runtime data disk.",
    },
]

NAMESPACE_VARIANTS = ["production", "staging", "default", "kube-system", "monitoring", "dev", "qa"]
POD_SUFFIXES = ["abc123", "xyz789", "def456", "ghi012", "jkl345"]


def generate_synthetic_kfrd(
    n_samples: int = 7000,
    output_path: str | None = None,
    seed: int = 42,
) -> list[dict]:
    """
    Generate n_samples by augmenting FAULT_TEMPLATES.
    Format: problem-attempt-solution triple (Eq 23-26, Appendix C.1).
    """
    random.seed(seed)
    samples = []

    for i in range(n_samples):
        template = FAULT_TEMPLATES[i % len(FAULT_TEMPLATES)]
        ns = random.choice(NAMESPACE_VARIANTS)
        pod_suffix = random.choice(POD_SUFFIXES)

        symptoms = template["symptoms"].copy()
        if random.random() > 0.6:
            symptoms.append(f"Detected at {random.randint(0,23):02d}:{random.choice(['00','15','30','45'])} UTC")
        if random.random() > 0.7:
            symptoms.append(f"Affecting {random.randint(1,10)} pods in namespace {ns}")

        attempted = [
            f"kubectl get pods -n {ns}",
            f"kubectl describe pod -n {ns}",
        ]
        if random.random() > 0.4:
            attempted.append(f"kubectl logs pod/pod-{pod_suffix} -n {ns} --previous")
        if random.random() > 0.6:
            attempted.append(f"kubectl get events -n {ns} --sort-by='.metadata.creationTimestamp'")

        actions = [a.replace(
            list(template["context"].values())[0] if template["context"] else "",
            f"pod-{pod_suffix}"
        ) for a in template["actions"]]

        sample = {
            "id": f"kfrd_{i:05d}",
            "problem": (
                f"Kubernetes {template['fault_category']} fault in namespace '{ns}'. "
                f"Symptoms: {'. '.join(symptoms[:3])}. "
                f"Context: {json.dumps(template['context'])}."
            ),
            "attempted_solutions": attempted,
            "final_solution": "\n".join(actions),
            "reasoning": (
                f"Root cause: {template['root_cause']}\n"
                f"The {symptoms[0].lower()} indicates {template['fault_category'].lower()} "
                f"configuration issue. "
                f"Resolution follows standard {template['fault_category']} remediation pattern."
            ),
            "fault_category": template["fault_category"],
            "symptoms": symptoms,
            "context": {**template["context"], "namespace": ns},
            "outcomes": template["outcomes"],
            "prevention": template["prevention"],
            "timestamp": time.time() - random.randint(0, 180 * 86400),
        }
        samples.append(sample)

    if output_path:
        path = Path(output_path)
        path.mkdir(parents=True, exist_ok=True)
        # Split: 5000 SFT + 2000 eval (matching paper)
        sft_samples = samples[:5000]
        eval_samples = samples[5000:]
        with open(path / "kfrd_sft.json", "w") as f:
            json.dump(sft_samples, f, indent=2)
        with open(path / "kfrd_eval.json", "w") as f:
            json.dump(eval_samples, f, indent=2)
        with open(path / "kfrd_all.json", "w") as f:
            json.dump(samples, f, indent=2)
        print(f"[KFRD] saved {len(sft_samples)} SFT + {len(eval_samples)} eval → {path}")

    return samples


def load_kfrd_as_episodes(n_samples: int = 50) -> list:
    """Convert synthetic KFRD samples to EpisodicMemory objects for EPMN seeding."""
    from epmn.memory import EpisodicMemory
    samples = generate_synthetic_kfrd(n_samples=n_samples)
    episodes = []
    for s in samples:
        ep = EpisodicMemory(
            symptoms=s["symptoms"],
            context=s["context"],
            actions=s["final_solution"].split("\n"),
            outcomes=s["outcomes"],
            timestamp=s["timestamp"],
            adaptive_value=1.0,
        )
        episodes.append(ep)
    return episodes


if __name__ == "__main__":
    samples = generate_synthetic_kfrd(
        n_samples=7000,
        output_path="/ai-data/datasets/kfrd",
    )
    print(f"Total: {len(samples)} samples")
    print("Example (SFT format):")
    print(json.dumps(samples[0], indent=2, default=str))

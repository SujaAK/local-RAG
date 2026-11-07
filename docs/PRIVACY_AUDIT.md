# T11 Privacy audit procedure
Pass only if zero outbound packets AND the system still works.

1. Setup phase (network allowed, once): download all models (Ollama, embedding, reranker, NLI).
2. Run phase, no network (Linux): `sudo -E bash scripts/run_offline.sh --netns "python -m eval.run_eval --split dev --limit 20"`
   This runs inside a network namespace with no interfaces, so any outbound attempt fails.
3. Packet capture on the host while running normally: `sudo tcpdump -i any -w audit.pcap 'not host 127.0.0.1 and not host ::1'`
   then count packets and DNS queries: `tcpdump -r audit.pcap | wc -l`; `tcpdump -r audit.pcap port 53`.
4. `python -m src.cli audit` lists env flags, non-local sockets and telemetry-like packages.
5. Time encryption overhead: `python -c "from src.audit import time_encryption; from src.utils import load_config; print(time_encryption(load_config(), 'pw'))"`
   and compare start-up/query latency with `privacy.encrypt_at_rest` on vs off.
Out of scope (state in the paper): malware on the host, physical device theft.

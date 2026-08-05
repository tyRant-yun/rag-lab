# 第 1 章候选基线：BM25 与 Dense 逐 Case 对比

状态：**candidate baseline / 待人工终审**。本表使用同一份
`chapter_01.jsonl` 的 39 条正向标注，比较两种检索器的 Top5 报告；它不是
模型调参或标签修订的依据。

`o12-13` 表示 Top5 中包含 ordinal 为 12、13 的相邻 Chunk，且二者共享
`block_ids`（真实 overlap）；`a12-13` 表示相邻但不共享 Block。`—` 表示该
检索器的 Top5 中没有相邻 Chunk 对。该列只提示可能的相邻窗口重复，不将其
算作额外相关证据。

| Case | BM25 首次相关排名 | Dense 首次相关排名 | 比较 | Top5 相邻/overlap |
| --- | ---: | ---: | --- | --- |
| internet-two-descriptions | 1 | 1 | 相同 | B:a2-3,o6-7; D:a1-2,a2-3 |
| end-system-and-host | 4 | 3 | Dense 更好 | B:o57-58,o56-57; D:o10-11,o3-4 |
| isp-role-and-interconnection | 1 | 2 | BM25 更好 | B:o30-31,o32-33,o31-32; D:o30-31,o31-32,o32-33 |
| tcp-ip-meaning | 2 | 2 | 相同 | B:o4-5; D:o4-5 |
| ietf-rfc-standards | 1 | 1 | 相同 | B:—; D:o4-5,o63-64 |
| distributed-applications-and-socket | 1 | 1 | 相同 | B:a5-6; D:o6-7,o10-11 |
| network-protocol-definition | 1 | 1 | 相同 | B:o8-9,a9-10; D:o8-9,a9-10 |
| access-network-edge-router | 1 | 1 | 相同 | B:o12-13,o14-15,o13-14,o15-16; D:o12-13,o15-16,o14-15,o13-14 |
| dsl-access | 1 | 2 | BM25 更好 | B:o12-13,a11-12,o13-14,o10-11; D:o12-13,o13-14,a11-12,o10-11 |
| cable-shared-medium | 2 | 1 | Dense 更好 | B:o13-14,o12-13,o14-15; D:o13-14,o46-47 |
| ftth-pon | 1 | 1 | 相同 | B:o14-15,o13-14,o12-13; D:o14-15,o13-14,o12-13 |
| wifi-and-cellular-access | 1 | 1 | 相同 | B:o15-16; D:o15-16 |
| guided-and-unguided-media | 1 | 1 | 相同 | B:o17-18; D:o17-18,o18-19 |
| optical-fiber-properties | 1 | 1 | 相同 | B:—; D:o17-18,a29-30 |
| satellite-propagation-delay | 1 | 1 | 相同 | B:o18-19; D:o18-19 |
| network-core-definition | — | 1 | Dense 更好 | B:a1-2; D:o32-33 |
| store-and-forward | 1 | 1 | 相同 | B:o21-22,o22-23,o23-24; D:o21-22,o23-24,o22-23 |
| queueing-and-packet-loss | 1 | 1 | 相同 | B:o22-23; D:o22-23,o39-40 |
| forwarding-table-and-routing | 1 | 2 | BM25 更好 | B:o23-24; D:o23-24,o41-42 |
| circuit-switching-reservation | 1 | 1 | 相同 | B:o25-26,o27-28,o26-27; D:o25-26,o28-29,o27-28,o26-27 |
| fdm-versus-tdm | 1 | 1 | 相同 | B:o26-27,o25-26,o27-28; D:o26-27,o27-28,o28-29 |
| packet-switching-statistical-sharing | 2 | 1 | Dense 更好 | B:o27-28,o26-27,o25-26; D:o28-29,o27-28,o26-27,o25-26 |
| isp-peering-ixp | 1 | 1 | 相同 | B:o32-33,o4-5; D:o32-33,o31-32,o30-31 |
| node-delay-components | 3 | 1 | Dense 更好 | B:a40-41; D:o35-36,o37-38,o36-37 |
| transmission-versus-propagation-delay | 1 | 1 | 相同 | B:o36-37,a38-39,o37-38; D:o36-37,o35-36,o37-38,a38-39 |
| traffic-intensity | 2 | 2 | 相同 | B:o39-40,o35-36,o36-37; D:o39-40,a38-39,o35-36 |
| finite-buffer-packet-loss | 1 | 1 | 相同 | B:o39-40; D:o39-40 |
| end-to-end-throughput-bottleneck | 2 | 3 | BM25 更好 | B:o46-47,o45-46; D:o46-47,o45-46,o44-45 |
| service-model-and-layering | 1 | 1 | 相同 | B:o50-51,o49-50,o51-52,o52-53; D:o50-51,o49-50,o51-52,o52-53 |
| internet-five-layer-stack | 1 | 1 | 相同 | B:o51-52,o50-51,o52-53,o49-50; D:o51-52,o50-51,o52-53,o49-50 |
| tcp-udp-ip-layer-roles | 1 | 1 | 相同 | B:o51-52,o52-53,a53-54; D:o52-53,o49-50 |
| encapsulation | 2 | 2 | 相同 | B:o54-55,a53-54,o51-52,o52-53; D:o54-55,o51-52 |
| dos-ddos-attacks | 1 | 1 | 相同 | B:o57-58,o56-57,o58-59; D:o57-58,o56-57,o58-59 |
| packet-sniffing-and-ip-spoofing | 1 | 1 | 相同 | B:o3-4; D:o58-59,o23-24 |
| endpoint-authentication-and-trust | 1 | 1 | 相同 | B:o58-59,a59-60,a60-61,o57-58; D:o58-59,a59-60,o57-58,a60-61 |
| arpanet-origin | 1 | 1 | 相同 | B:a61-62; D:a61-62 |
| tcp-udp-ip-history | 1 | 1 | 相同 | B:o63-64,a64-65,a65-66; D:o63-64,a62-63,a64-65,a61-62 |
| web-commercialization | 1 | 1 | 相同 | B:o66-67,a67-68; D:o66-67,a67-68 |
| cloud-and-broadband-development | 1 | 1 | 相同 | B:—; D:a65-66 |

Top5 汇总：BM25 Hit@5/Mean Recall@5/MRR 为 **0.974359 / 0.974359 /
0.861111**；Dense 为 **1.000000 / 1.000000 / 0.888889**。唯一的 BM25
Top5 未命中为 `network-core-definition`。这些数字仍受候选标签、单一教材和
单一模型版本限制，待人工终审后才能作为金标准比较依据。

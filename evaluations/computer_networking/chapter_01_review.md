# 第 1 章检索评测审阅清单

状态：**candidate baseline / 待人工终审**。本清单中的相关 chunk 均由人工阅读本地 `chunks.jsonl` 的正文、`heading_path` 与 PDF 页码后标注；没有依据 BM25 或 Dense 的返回排名选取相关项。当前只包含正向问题，尚未扩展无答案契约。

数据来源：PDF 物理页 19–61；本地产物 `computer-networking/output/chapter-01/baseline-v4-final`；collection `computer-networking-chapter-01-v1`；document ID `sha256:fef1b562abea48b141fabdee49cb9bcc7c7fecd400b416affd5811543da52592`。

| Case | Query | 已审阅相关 chunk 摘要 | heading_path | 页码 | Chunk ID | Filters |
| --- | --- | --- | --- | --- | --- | --- |
| internet-two-descriptions | 教材从哪两种角度说明什么是因特网？ | 明确给出“具体构成”与“为分布式应用提供服务的基础设施”两种描述。 | 第1章 > 1.1 | 19–19 | `sha256:44aaa7b91ad4115eb466f9566942db2668648dfed3c2175a728c8838c4434bad` | document ID |
| end-system-and-host | 端系统为什么也称为主机？通常包括哪些设备？ | 将 PC、服务器、移动及非传统设备定义为 host/end system。 | 第1章 > 1.1 > 1.1.1 | 19–21 | `sha256:5e8fbdb98058f4edf54291c598774efcf8d4f72e66e27d530a4cd8dae0c1ecef` | — |
| isp-role-and-interconnection | ISP 如何让端系统接入因特网，并且为什么 ISP 之间必须互联？ | 说明接入类型、ISP 网络构成与分层 ISP 互联。 | 第1章 > 1.1 > 1.1.1 | 21–21 | `sha256:d889bb2e2dc3c586ebf57b312a84deea669612e5dbfd70314fdc02a208e0b449` | — |
| tcp-ip-meaning | TCP 和 IP 分别是什么，TCP/IP 这个统称指什么？ | 介绍 TCP、IP 与 TCP/IP 统称。 | 第1章 > 1.1 > 1.1.1 | 21–21 | `sha256:d889bb2e2dc3c586ebf57b312a84deea669612e5dbfd70314fdc02a208e0b449` | — |
| ietf-rfc-standards | IETF 与 RFC 在因特网标准制定中分别扮演什么角色？ | IETF 研发标准，RFC 为标准文档并定义多个协议。 | 第1章 > 1.1 > 1.1.1 | 21–22 | `sha256:61db585dcdb48ea3a8ed3a11c1efbb19a8d885dff6b3b7440d79003fc805562c` | — |
| distributed-applications-and-socket | 分布式因特网应用运行在哪里，程序通过什么接口请求网络交付数据？ | 应用运行在端系统，套接字接口规定发送程序的请求规则。 | 第1章 > 1.1 > 1.1.2 | 22–22 | `sha256:4ecab02e7617298be1bc36ae4fa2a30dae468f304b5f2484808d40102ec641df` | — |
| network-protocol-definition | 网络协议如何定义通信实体交换报文时的格式、顺序和操作？ | 给出协议的正式定义。 | 第1章 > 1.1 > 1.1.3 | 24–24 | `sha256:2c4eeea0155eae37f6573f73e15ecee0fa314b27e75c51dd29be4897d2cdc20d` | pages 24–24 |
| access-network-edge-router | 什么是接入网，边缘路由器处于什么位置？ | 定义接入网与第一台边缘路由器。 | 第1章 > 1.2 > 1.2.1 | 26–28 | `sha256:edb0403ca7851ca3b8d20e822dbbd6d51376b20d43c61831b0faf234b28849c1` | heading prefix |
| dsl-access | 家庭 DSL 接入中 DSLAM 和电话线各自做什么？ | 说明 DSL 调制解调、频段和 DSLAM。 | 第1章 > 1.2 > 1.2.1 | 26–28 | `sha256:edb0403ca7851ca3b8d20e822dbbd6d51376b20d43c61831b0faf234b28849c1` | pages 26–28 |
| cable-shared-medium | 为什么多个家庭同时使用电缆因特网时速率可能下降？ | HFC 下行/上行均是共享广播媒介。 | 第1章 > 1.2 > 1.2.1 | 28–29 | `sha256:32f39094e0a9f08c2c9a37b3d13ec4153928cdee317e1a461f306583c4ce3b2f` | — |
| ftth-pon | FTTH 的 PON 中 ONT、分配器和 OLT 如何连接？ | 描述 PON 的家庭、分配器和中心局端接器。 | 第1章 > 1.2 > 1.2.1 | 28–29 | `sha256:32f39094e0a9f08c2c9a37b3d13ec4153928cdee317e1a461f306583c4ce3b2f` | — |
| wifi-and-cellular-access | WiFi 与 4G/5G 广域蜂窝接入有什么区别？ | 对比几十米 WiFi 与数万米蜂窝覆盖和速率。 | 第1章 > 1.2 > 1.2.1 | 30–31 | `sha256:b439ee91b741a638d5ccd5605c85b675d6e36e0a469289ca4829b50234baff67` | — |
| guided-and-unguided-media | 导引型媒介和非导引型媒介如何区分？ | 给出固体媒介与空气/空间传播的定义及例子。 | 第1章 > 1.2 > 1.2.2 | 31–31 | `sha256:9f07a03f4a2ca541bc8c1dbcd4635c542bf47c6f2a7b7a9a4b5209be9c8caa43` | — |
| optical-fiber-properties | 光纤为什么适合长途和主干网络？ | 高比特率、抗电磁干扰、低衰减且难窃听。 | 第1章 > 1.2 > 1.2.2 | 31–32 | `sha256:fd371a27b6a36491b0a8ae950dc043082605aba00c79c6c60ac89957b064d063` | — |
| satellite-propagation-delay | 同步卫星链路为何时延大，LEO 有何不同？ | 说明同步卫星距离、约 280ms 传播时延与 LEO 覆盖。 | 第1章 > 1.2 > 1.2.2 | 32–33 | `sha256:a78c8118e2ffcce59b918f555b0ff6ec0edec3a76ff2a5a76bcffb57b4158fde` | heading + pages |
| network-core-definition | 网络核心由哪些部分构成？ | 定义互联端系统的分组交换机和链路网状网络。 | 第1章 > 1.3 | 33–33 | `sha256:417b37590b31a85b8d807fb92344c9fad7ddd98c85596aa13785e91aef72b1a0` | — |
| store-and-forward | 什么是存储转发？ | 交换机先接收完整分组，才向输出链路转发。 | 第1章 > 1.3 > 1.3.1 | 33–34 | `sha256:55ef3f7bfd1a54420370b18be41cab13dece76fa3639d2b07e420ad89a65e0e7` | heading prefix |
| queueing-and-packet-loss | 输出缓存中的排队与丢包如何产生？ | 忙链路导致排队，有限缓存满后丢弃分组。 | 第1章 > 1.3 > 1.3.1 | 34–35 | `sha256:5ee75327799efe37d3afb22231125491f3fa81c0853d966031be8e1dee7636b1` | — |
| forwarding-table-and-routing | 路由器怎样用目的地址选择输出链路？ | IP 地址查表并由路由选择协议设置转发表。 | 第1章 > 1.3 > 1.3.1 | 35–36 | `sha256:f904ed1d71ad2d654280410dcbe29fc75e0f342eb724c5bf2f437f90d00b7f18` | — |
| circuit-switching-reservation | 电路交换与分组交换的资源预留差异？ | 电路预留带宽/缓存，分组按需共享并可能等待。 | 第1章 > 1.3 > 1.3.2 | 36–37 | `sha256:d8e7dd3779431d447ca0b8249296c2014e220a8539cb5e60ebd560ddbb54e3f2` | — |
| fdm-versus-tdm | FDM 和 TDM 怎样复用链路？ | 分别按频段和按帧内时隙分配专用资源。 | 第1章 > 1.3 > 1.3.2 | 37–38 | `sha256:1b646f30e10fbce275816a9577e92db77578f9d81a86d9a3e48a62bdc987610c` | — |
| packet-switching-statistical-sharing | 突发业务下分组交换为何更有效？ | 通过统计复用按需共享带宽，而非空闲预留。 | 第1章 > 1.3 > 1.3.2 | 38–39 | `sha256:004dab878c27f73de8ec7aff3681d9d493c1e25cfd0e1077016066d62351f07f` | — |
| isp-peering-ixp | 多宿、对等和 IXP 分别解决什么互联问题？ | 说明冗余接入、直接对等和交换点。 | 第1章 > 1.3 > 1.3.3 | 40–41 | `sha256:fbf3ee1bc1fe7506459299383dc5e4a75cf742b8aaf709da54cd09454e61706f` | — |
| node-delay-components | 节点的四类主要时延是什么？ | 列出处理、排队、传输和传播时延。 | 第1章 > 1.4 > 1.4.1 | 42–43 | `sha256:ed25f06995ed26ae19aaa0f65fa005df4679f2536fdb5cfbf6a2fbfc41a3f6de` | — |
| transmission-versus-propagation-delay | L/R 与 d/s 各由什么决定？ | 区分分组长度/链路速率与距离/传播速率。 | 第1章 > 1.4 > 1.4.1 | 43–43 | `sha256:e8b30aa05254e3e8aaec90803740d163661c4a6299af3111019931b666540747` | — |
| traffic-intensity | La/R 接近或超过 1 时会怎样？ | 超过 1 队列趋向无限增长，接近 1 时延骤增。 | 第1章 > 1.4 > 1.4.2 | 44–45 | `sha256:2deee347b119e1907391627bb533512c34b26da5c47e88cfa614af0c2f694acf` | — |
| finite-buffer-packet-loss | 有限队列为什么丢包？ | 满队列无法存储到达分组，丢包随流量强度增加。 | 第1章 > 1.4 > 1.4.2 | 45–46 | `sha256:d4c552ca1ddab72a4c62c57bb38c24f75d9022c4e5402da54e05d39379c16498` | — |
| end-to-end-throughput-bottleneck | 吞吐量为何由瓶颈链路决定？ | 两链路及多链路的最小传输速率模型。 | 第1章 > 1.4 > 1.4.4 | 47–49 | `sha256:c5cfceeb3d5498501579a90194d6e8c34e94bebed76da06de238759b1ccf299b` | — |
| service-model-and-layering | 分层为何有助于模块化，服务模型是什么？ | 说明层服务、下层依赖和实现可替换性。 | 第1章 > 1.5 > 1.5.1 | 50–51 | `sha256:1c1e5d2b8900756a4ef694d65d3e68f736f8314c06174e823dab02a2cd79bb0d` | heading prefix |
| internet-five-layer-stack | 因特网五层协议栈包含哪些层？ | 给出应用、运输、网络、链路、物理五层。 | 第1章 > 1.5 > 1.5.1 | 51–52 | `sha256:b498aed2633efc9cd727620b14a7e60ff610a9a4ae55620c7a82a960903a7f2a` | pages 51–52 |
| tcp-udp-ip-layer-roles | TCP、UDP 和 IP 分别提供什么服务？ | 描述 TCP/UDP 运输服务与 IP 数据报/选路。 | 第1章 > 1.5 > 1.5.1 | 52–53 | `sha256:cee6dddd5adef91f2143ad4d1179eddb0e264ecbfc626dc722387fec14b5940f` | — |
| encapsulation | 报文如何封装成报文段、数据报和帧？ | 逐层增加首部和有效载荷的封装过程。 | 第1章 > 1.5 > 1.5.2 | 53–54 | `sha256:16eda57b76ec3959212e9dd38a2f481f6f74d580241361eb435fa73e37a3aac4` | — |
| dos-ddos-attacks | DoS/DDoS 的主要类型和困难是什么？ | 弱点、带宽、连接洪泛及僵尸网络 DDoS。 | 第1章 > 1.6 | 55–56 | `sha256:3d0bdaccd7172a8d735580623424c47760f9a14e8b1dd42bd169a07e6811c403` | — |
| packet-sniffing-and-ip-spoofing | 嗅探器如何获得敏感信息，什么是 IP 欺骗？ | 说明被动嗅探与伪造源地址。 | 第1章 > 1.6 | 56–57 | `sha256:368706baf6a16085415f6ce956925bbed9e98e35b84860c26a9a5662fa14987e` | — |
| endpoint-authentication-and-trust | 端点鉴别与早期因特网信任模型？ | 鉴别报文来源并解释默认互信的历史原因。 | 第1章 > 1.6 | 56–57 | `sha256:e5ce3d62288d89e2d37ebe1de7e7cfe00731cf20c23e72c131f4baef0a821fbf` | pages 56–57 |
| arpanet-origin | ARPAnet 如何成为公共因特网祖先？ | 1969 年部署首批交换机和 ARPAnet 早期发展。 | 第1章 > 1.7 > 1.7.1 | 58–58 | `sha256:6195b2c9d095471f2039b1259de460da36949050e5e27126ddb65855202104aa` | heading prefix |
| tcp-udp-ip-history | TCP、UDP、IP 如何从早期研究中形成？ | TCP/IP 分离和三协议在 1970 年代末形成。 | 第1章 > 1.7 > 1.7.2 | 58–59 | `sha256:6bb0c14d23f0aa055cfe65736cc287968674e4fb6f9512a333ec76b35b19cf65` | — |
| web-commercialization | 万维网如何推动 1990 年代商业化？ | Web、浏览器、服务器和电子商务增长。 | 第1章 > 1.7 > 1.7.4 | 60–60 | `sha256:6e330f73d4134db583082cd440f8c0449ea5eaf9ddab8cddba090adc2059e117` | pages 60–60 |
| cloud-and-broadband-development | 宽带、移动接入和云计算带来什么变化？ | 概述 21 世纪家庭宽带、无线、社交和云发展。 | 第1章 > 1.7 > 1.7.5 | 60–61 | `sha256:4fab610330177233213fbc458ffde764817f5b50e03bb756c36c3b5b600bc73f` | — |

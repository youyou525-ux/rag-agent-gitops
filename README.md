# RAGi��动问答智能体 - GitOps仓库

本仓库用于 RAG（Retrieval-Augmented Generation）问答智能体项目的全部 Kubernetes 清单、知识库文档及部署配置。

## 项目架构

| 模块 | 内容 | 技术栈 |
|------|------|--------|
| 模块一 | 基础设施（K8s集群 + OpenEBS + ArgoCD + Prometheus） | Ansible / Helm / K8s |
| 模块二 | Milvus向量数据库 + 知识库创建 | Milvus / Python |
| 模块三 | 数据注入管道（文档切分 + 向量化 + 插入） | Go / Milvus / Ollama |
| 模块四 | RAG后端API（检索 + LLM生成） | Go / Gin / Milvus / Ollama |
| 模块五 | 前端界面 | Vue.js / Node.js Express |
| 模块六 | 容器化与GitOps部署 | Docker / K8s / ArgoCD |

## 仓库结构

```
rag-agent-gitops/
├── README.md
├── apps/
│   ├── milvus/          # Milvus 部署清单
│   └── rag-app/         # RAG 应用部署清单
├── knowledge-base/      # 知识库文档
└── base/                # ArgoCD Application 等基础配置
```

## 与 SRE 项目共存

本仓库与 [sre-agent-gitops](https://github.com/youyou525-ux/sre-agent-gitops) 在同一个 K8s 集群上运行：
- **Ollama** 由旧仓库管理，本仓库不重复定义，RAG项目通过 Service 地址引用
- **ArgoCD** 通过多个 Application 分别管理两个仓库的部署，互不干扰
- **存储** 使用 OpenEBS 提供的 `openebs-hostpath` 存储类

## 集群信息

| 节点 | IP | 角色 |
|------|-----|------|
| k8s-node1 | 192.168.30.11 | Master |
| k8s-node2 | 192.168.30.12 | Worker |
| k8s-node3 | 192.168.30.13 | Worker |

## 作者

youyou525-ux

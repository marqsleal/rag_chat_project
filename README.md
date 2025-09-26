# RAG Chat Project

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

**RAG (Retrieval-Augmented Generation)** é uma técnica que combina a recuperação de informações de uma base de conhecimento com a geração de texto por modelos de linguagem. Este projeto implementa um sistema RAG completo para consulta de documentos usando **LLaMA 2** e **ChromaDB**.

## Estrutura do Projeto

```
rag_chat_project/
├── Makefile                    # Comandos de automação
├── requirements.txt            # Dependências Python
├── data/                    
│   ├── books/                  # Livros em formato .md
│   └── chroma-db/              # Bases vetoriais ChromaDB
│       └── books/              # Collection de livros
├── models/                  
│   └── llama-2-7b-chat.Q4_K_M.gguf  # Modelo LLaMA local
├── notebooks/               
│   ├── 001__RAG__COMPARE_EMBEDDINGS.ipynb  # Análise de embeddings
│   ├── 002__RAG__QUERY_DATA.ipynb          # Testes de consulta
│   └── 003__RAG__QUERY_ENGINE.ipynb        # Engine completo
├── rag_project/                # Código fonte
│   ├── constants.py            # Configurações e constantes
│   ├── rag_models.py          # Modelos Pydantic (LLMConfig, RAGResponse)
│   ├── compare_embeddings.py  # Utilitários de embeddings
│   ├── create_chroma_database.py  # Criação de bases vetoriais
│   ├── query_data.py          # RAG Engine principal
│   └── logger.py              # Sistema de logging
└── tests/                     # Testes unitários
```

## O que é RAG?

**RAG** combina dois componentes principais:
- **Retrieval (Recuperação)**: Busca documentos relevantes em uma base vetorial
- **Generation (Geração)**: Usa um LLM para gerar respostas baseadas nos documentos encontrados

### Fluxo do Sistema RAG

```
Pergunta → Embeddings → Busca no ChromaDB → Documentos Relevantes → LLM → Resposta
```

## Arquitetura do Sistema

### Componentes Principais

1. **Embeddings**: Vetorização de texto usando `sentence-transformers/all-MiniLM-L6-v2`
2. **ChromaDB**: Base de dados vetorial para armazenamento e busca de documentos
3. **LLaMA 2**: Modelo de linguagem local para geração de respostas
4. **RAG Engine**: Motor que orquestra todo o pipeline

### Modelos Utilizados

| Componente | Modelo | Descrição |
|------------|--------|-----------|
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Modelo leve e eficiente para vetorização |
| **LLM** | `LLaMA-2-7B-Chat-GGUF` | Modelo quantizado (Q4_K_M) para chat |
| **Vector DB** | `ChromaDB` | Base vetorial para armazenamento persistente |

## Quick Start

### 1. Inicialização do Projeto

```bash
# Clonar o repositório
git clone https://github.com/marqsleal/rag_chat_project.git
cd rag_chat_project

# Inicializar ambiente completo (venv + dependências + modelo)
make init
```

### 2. Criar Base de Dados Vetorial

```bash
# Para livros (Alice in Wonderland)
make chroma TYPE=books
```

### 3. Usar o Sistema RAG

```python
from rag_project.query_data import create_rag_engine

# Criar engine RAG
engine = create_rag_engine(
    chroma_path="./data/chroma-db/books",
    collection_name="books"
)

# Fazer uma pergunta
response = engine.query("What is Alice in Wonderland about?")
print(response)
```

## Componentes Técnicos

### Embeddings
- **Modelo**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensão**: 384 vetores
- **Otimizado para**: Busca semântica eficiente
- **Normalização**: Embeddings normalizados para cosine similarity

### ChromaDB
**ChromaDB** é uma base de dados vetorial que permite:
- Armazenamento persistente de embeddings
- Busca por similaridade coseno
- Collections organizadas por domínio
- Metadados associados aos documentos

**Collections disponíveis**:
- `books`: Literatura (Alice in Wonderland)

### LLaMA 2 (Local)
- **Modelo**: `llama-2-7b-chat.Q4_K_M.gguf`
- **Quantização**: Q4_K_M (eficiência vs qualidade)
- **Contexto**: 2048 tokens
- **Temperatura**: 0.3 (respostas focadas)
- **Stop sequences**: `["</s>", "[/INST]"]`

### RAG Engine
Classe `RAGQueryEngine` que integra:
- Recuperação de documentos com score mínimo
- Formatação de prompts com contexto
- Geração de respostas estruturadas
- Metadados de origem e similaridade



## Conceitos Fundamentais

### O que são Embeddings?
**Embeddings** são representações vetoriais de texto que capturam significado semântico:
- Textos similares ficam próximos no espaço vetorial
- Permite busca por similaridade (não apenas palavras-chave)
- Modelo `all-MiniLM-L6-v2` gera vetores de 384 dimensões

### O que é ChromaDB?
**ChromaDB** é uma base de dados vetorial especializada:
- Armazenamento eficiente de embeddings
- Busca rápida por similaridade coseno
- Persistência local com SQLite
- Metadados e filtragem avançada

### Pipeline RAG Detalhado

1. **Indexação** (offline):
   ```
   Documentos → Chunking → Embeddings → ChromaDB
   ```

2. **Consulta** (online):
   ```
   Pergunta → Embedding → Busca ChromaDB → Top-K docs → Prompt + LLM → Resposta
   ```

## Comandos Disponíveis

```bash
# Inicialização
make init              # Setup completo do projeto
make venv              # Criar ambiente virtual
make requirements      # Instalar dependências
make get_llama_model   # Baixar modelo LLaMA

# Dados e Bases Vetoriais
make chroma TYPE=books # Criar base para livros

# Desenvolvimento
make format            # Formatar código com ruff
make lint              # Verificar qualidade do código
make test              # Executar testes
make clean             # Limpar cache e arquivos temporários
```

## Notebooks de Análise

| Notebook | Descrição |
|----------|-----------|
| `001__RAG__COMPARE_EMBEDDINGS.ipynb` | Análise e comparação de modelos de embedding |
| `002__RAG__QUERY_DATA.ipynb` | Testes de consulta e recuperação |
| `003__RAG__QUERY_ENGINE.ipynb` | Demonstração completa do RAG Engine |

## Exemplo de Uso Avançado

```python
from rag_project.query_data import create_rag_engine
from rag_project.rag_models import LLMConfig

# Configuração personalizada do LLM
custom_config = LLMConfig(
    temperature=0.7,
    max_new_tokens=512,
    top_k=50
)

# Criar engine com configuração customizada
engine = create_rag_engine(
    chroma_path="./data/chroma-db/books",
    collection_name="books",
    llm_config=custom_config
)

# Consulta com metadados completos
response = engine.query_with_metadata(
    question="Describe the Cheshire Cat character",
    documents_retrieve=5,
    min_similarity_score=0.3,
    return_sources=True
)

print(f"Answer: {response.answer}")
print(f"Sources: {response.sources}")
print(f"Retrieved docs: {response.retrieved_docs}")
print(f"Similarity scores: {response.similarity_scores}")
```

## Tecnologias e Dependências

- **Python 3.13**: Linguagem base
- **LangChain**: Framework para aplicações LLM
- **HuggingFace**: Modelos de embedding e transformers
- **ChromaDB**: Base de dados vetorial
- **CTransformers**: Execução local de modelos GGUF
- **Sentence Transformers**: Modelos de embedding otimizados
- **Pydantic**: Validação e serialização de dados
- **Jupyter**: Notebooks para análise e experimentação

---

## Próximos Passos

- [ ] Interface web com Streamlit/Gradio
- [ ] Suporte a múltiplos formatos (PDF, DOC, etc.)
- [ ] Cache de consultas para performance
- [ ] Métricas de avaliação do sistema RAG
- [ ] Deploy com Docker

---

**Desenvolvido para estudos em RAG e recuperação de informações**


import os
import time

import streamlit as st

from rag_project.constants import (
    AZURE_CHROMA_DB_DIR,
    AZURE_COLLECTION_NAME,
    AZURE_RAW_DATA_DIR,
    BOOKS_CHROMA_DB_DIR,
    BOOKS_COLLECTION_NAME,
    BOOKS_RAW_DATA_DIR,
)
from rag_project.create_chroma_database import create_chroma_db
from rag_project.query_data import create_rag_engine
from rag_project.rag_models import DEFAULT_PROMPT_TEMPLATE, LLMConfig

st.set_page_config(
    page_title="RAG Query Engine", page_icon="📚", layout="wide", initial_sidebar_state="expanded"
)

UI_CONSTANTS = {
    "MAIN_HEADER_CLASS": "main    tab1, tab2, tab3 = st.tabs([eader",
    "SECTION_HEADER_CLASS": "section-header",
    "CONFIG_CONTAINER_START": '<div class="config-container">',
    "CONFIG_CONTAINER_END": "</div>",
    "RESPONSE_CONTAINER_START": '<div class="response-container">',
    "RESPONSE_CONTAINER_END": "</div>",
    "ERROR_CONTAINER_START": '<div class="error-container">',
    "ERROR_CONTAINER_END": "</div>",
    "METRIC_CONTAINER_START": '<div class="metric-container">',
    "METRIC_CONTAINER_END": "</div>",
}

PROMPT_OPTIONS = {"DEFAULT": "🎯 Usar prompt padrão", "CUSTOM": "✏️ Prompt personalizado"}

TEMPLATE_PLACEHOLDERS = {"CONTEXT": "{context}", "QUESTION": "{question}"}

DB_FILE_NAME = "chroma.sqlite3"
CSS_STYLES = """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.8rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    .response-container {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: #1a1a1a;
        padding: 1.2rem;
        border-radius: 0.8rem;
        border-left: 5px solid #0066cc;
        box-shadow: 0 4px 20px rgba(79, 172, 254, 0.2);
    }
    .error-container {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 0.8rem;
        border-left: 5px solid #d63031;
        box-shadow: 0 4px 20px rgba(255, 107, 107, 0.2);
    }
    .config-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.8rem;
        border-radius: 1rem;
        border: 2px solid rgba(255, 255, 255, 0.1);
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    .config-container .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.9);
        color: #333;
        border-radius: 0.5rem;
    }
    .config-container .stTextArea > div > div {
        background-color: rgba(255, 255, 255, 0.9);
        color: #333;
        border-radius: 0.5rem;
    }
    .config-container .stButton > button {
        background: linear-gradient(135deg, #00b4db 0%, #0083b0 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(0, 180, 219, 0.3);
        transition: all 0.3s ease;
    }
    .config-container .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 180, 219, 0.4);
    }
    
    /* Dark theme compatibility */
    @media (prefers-color-scheme: dark) {
        .metric-container {
            background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        }
        .response-container {
            background: linear-gradient(135deg, #2b6cb0 0%, #3182ce 100%);
            color: white;
        }
        .config-container {
            background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        }
    }
</style>
"""

st.markdown(CSS_STYLES, unsafe_allow_html=True)


PREDEFINED_QUERIES = {
    "books": [
        "Who is the White Rabbit and how does Alice first meet him?",
        "What does Alice drink or eat that makes her change size?",
        "Describe the Duchess's kitchen and what happens there",
        "What games are played in Wonderland?",
        "What poems or songs are recited in the story?",
        "How does Alice's adventure end?",
    ],
    "azure": [
        "What is Azure Virtual Network?",
        "How to create a storage account in Azure?",
        "What are the different types of Azure databases?",
        "How to configure Azure Active Directory?",
        "What is Azure Kubernetes Service?",
    ],
}

AVAILABLE_COLLECTIONS = {
    "books": {
        "name": "📚 Books (Alice in Wonderland)",
        "path": BOOKS_CHROMA_DB_DIR,
        "collection_name": BOOKS_COLLECTION_NAME,
        "description": "Coleção com o livro Alice in Wonderland",
        "data_dir": BOOKS_RAW_DATA_DIR,
        "enabled": True,
        "example_queries": PREDEFINED_QUERIES["books"],
    },
    "azure": {
        "name": "☁️ Azure Documentation (Em Construção)",
        "path": AZURE_CHROMA_DB_DIR,
        "collection_name": AZURE_COLLECTION_NAME,
        "description": "Coleção com documentação do Azure - Em desenvolvimento",
        "data_dir": AZURE_RAW_DATA_DIR,
        "enabled": False,
        "example_queries": PREDEFINED_QUERIES["azure"],
    },
}


LLM_PROFILES = {
    "default": {
        "name": "🎯 Padrão (Balanceado)",
        "description": "Configuração balanceada para uso geral",
        "config": LLMConfig(
            max_new_tokens=256,
            temperature=0.3,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.05,
            context_length=2048,
            seed=42,
        ),
    },
    "conservative": {
        "name": "🛡️ Conservador (Mais Preciso)",
        "description": "Respostas mais precisas e determinísticas",
        "config": LLMConfig(
            max_new_tokens=200,
            temperature=0.1,
            top_k=20,
            top_p=0.7,
            repetition_penalty=1.1,
            context_length=2048,
            seed=42,
        ),
    },
    "creative": {
        "name": "🎨 Criativo (Mais Variado)",
        "description": "Respostas mais criativas e variadas",
        "config": LLMConfig(
            max_new_tokens=300,
            temperature=0.7,
            top_k=60,
            top_p=0.95,
            repetition_penalty=1.0,
            context_length=2048,
            seed=-1,
        ),
    },
    "detailed": {
        "name": "📝 Detalhado (Respostas Longas)",
        "description": "Respostas mais longas e detalhadas",
        "config": LLMConfig(
            max_new_tokens=512,
            temperature=0.4,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.05,
            context_length=4096,
            seed=42,
        ),
    },
}


def validate_custom_prompt(prompt: str) -> tuple[bool, str]:
    if not prompt or not prompt.strip():
        return False, "Prompt não pode estar vazio"

    if TEMPLATE_PLACEHOLDERS["CONTEXT"] not in prompt:
        return False, f"Prompt deve conter {TEMPLATE_PLACEHOLDERS['CONTEXT']}"

    if TEMPLATE_PLACEHOLDERS["QUESTION"] not in prompt:
        return False, f"Prompt deve conter {TEMPLATE_PLACEHOLDERS['QUESTION']}"

    return True, ""


def get_enabled_collections() -> dict:
    return {
        key: config for key, config in AVAILABLE_COLLECTIONS.items() if config.get("enabled", True)
    }


def get_disabled_collections() -> dict:
    return {
        key: config
        for key, config in AVAILABLE_COLLECTIONS.items()
        if not config.get("enabled", True)
    }


def database_exists(collection_config: dict) -> bool:
    db_file = os.path.join(collection_config["path"], DB_FILE_NAME)
    return os.path.exists(db_file)


def ensure_chroma_database(collection_config: dict) -> tuple[bool, str]:
    if database_exists(collection_config):
        return True, ""
    try:
        data_dir = collection_config["data_dir"]
        if not os.path.exists(data_dir):
            return False, f"Diretório de dados não encontrado: {data_dir}"
        success = create_chroma_db(
            data_dir=data_dir,
            chroma_db_dir=collection_config["path"],
            collection_name=collection_config["collection_name"],
        )

        if not success:
            return False, f"Falha ao criar base de dados para {collection_config['name']}"

        return True, ""

    except Exception as e:
        return False, f"Erro ao criar base de dados: {str(e)}"


@st.cache_resource
def initialize_rag_engine(collection_key: str, profile_key: str, custom_prompt: str = ""):
    try:
        collection_config = AVAILABLE_COLLECTIONS[collection_key]
        with st.spinner(f"Verificando base de dados para {collection_config['name']}..."):
            db_success, db_error = ensure_chroma_database(collection_config)

            if not db_success:
                return None, db_error, None, None

        with st.spinner("Inicializando RAG Query Engine..."):
            profile_config = LLM_PROFILES[profile_key]
            llm_config = profile_config["config"]

            engine = create_rag_engine(
                chroma_path=collection_config["path"],
                collection_name=collection_config["collection_name"],
                llm_config=llm_config,
            )

        return engine, None, collection_config, profile_config
    except Exception as e:
        return None, str(e), None, None


def show_configuration_page():
    st.markdown(
        '<h1 class="main-header">⚙️ Configuração do RAG Query Engine</h1>', unsafe_allow_html=True
    )
    st.markdown("Configure a coleção de documentos e o perfil do modelo antes de iniciar.")
    st.info("💡 A base de dados será criada automaticamente se não existir.")

    st.markdown('<div class="config-container">', unsafe_allow_html=True)
    st.subheader("📂 Seleção da Coleção")

    enabled_collections = {
        key: config for key, config in AVAILABLE_COLLECTIONS.items() if config.get("enabled", True)
    }
    collection_options = {key: config["name"] for key, config in enabled_collections.items()}

    selected_collection = st.selectbox(
        "Escolha a coleção de documentos:",
        options=list(collection_options.keys()),
        format_func=lambda x: collection_options[x],
        key="collection_select",
    )

    disabled_collections = {
        key: config
        for key, config in AVAILABLE_COLLECTIONS.items()
        if not config.get("enabled", True)
    }
    if disabled_collections:
        st.info("📋 **Coleções em desenvolvimento:**")
        for key, config in disabled_collections.items():
            st.write(f"• {config['name']} - {config['description']}")
        st.write("Essas coleções estarão disponíveis em breve.")

    if selected_collection:
        collection_config = AVAILABLE_COLLECTIONS[selected_collection]
        st.info(f"📝 {collection_config['description']}")

        with st.expander("ℹ️ Detalhes da coleção"):
            st.write(f"**Caminho da base de dados:** `{collection_config['path']}`")
            st.write(f"**Caminho dos dados:** `{collection_config['data_dir']}`")
            st.write(f"**Nome da coleção:** `{collection_config['collection_name']}`")

            db_file = os.path.join(collection_config["path"], "chroma.sqlite3")
            if os.path.exists(db_file):
                st.success("✅ Base de dados já existe")
            else:
                st.warning("⚠️ Base de dados será criada na inicialização")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="config-container">', unsafe_allow_html=True)
    st.subheader("🤖 Perfil do Modelo LLM")

    profile_options = {key: config["name"] for key, config in LLM_PROFILES.items()}
    selected_profile = st.selectbox(
        "Escolha o perfil de hiperparâmetros:",
        options=list(profile_options.keys()),
        format_func=lambda x: profile_options[x],
        key="profile_select",
    )

    if selected_profile:
        profile_config = LLM_PROFILES[selected_profile]
        st.info(f"📝 {profile_config['description']}")

        with st.expander("🔧 Parâmetros do perfil"):
            config = profile_config["config"]
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Max tokens:** {config.max_new_tokens}")
                st.write(f"**Temperature:** {config.temperature}")
                st.write(f"**Top-k:** {config.top_k}")
                st.write(f"**Top-p:** {config.top_p}")

            with col2:
                st.write(f"**Repetition penalty:** {config.repetition_penalty}")
                st.write(f"**Context length:** {config.context_length}")
                st.write(f"**Seed:** {config.seed}")
                st.write(f"**Threads:** {config.threads}")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="config-container">', unsafe_allow_html=True)
    st.subheader("📝 Configuração do Prompt")

    use_default_prompt = st.radio(
        "Escolha o tipo de prompt:",
        options=[PROMPT_OPTIONS["DEFAULT"], PROMPT_OPTIONS["CUSTOM"]],
        key="prompt_type_select",
    )

    if use_default_prompt == PROMPT_OPTIONS["DEFAULT"]:
        st.info("📋 Usando o prompt padrão otimizado para RAG")
        with st.expander("👀 Visualizar prompt padrão"):
            st.code(DEFAULT_PROMPT_TEMPLATE.strip(), language="text")
        custom_prompt = None
    else:
        st.info("✏️ Configure seu prompt personalizado")
        st.write("**Instruções:**")
        st.write("• Use `{context}` onde o contexto recuperado deve ser inserido")
        st.write("• Use `{question}` onde a pergunta do usuário deve ser inserida")

        custom_prompt = st.text_area(
            "Digite seu prompt personalizado:",
            value=DEFAULT_PROMPT_TEMPLATE.strip(),
            height=200,
            help="Certifique-se de incluir {context} e {question} no seu prompt",
            key="custom_prompt_input",
        )

        if custom_prompt:
            is_valid, error_msg = validate_custom_prompt(custom_prompt)
            if not is_valid:
                st.warning(f"⚠️ {error_msg}")
            else:
                st.success("✅ Prompt válido!")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2, _ = st.columns([1, 2, 1])

    with col2:
        if st.button("🚀 Inicializar RAG Query Engine", type="primary", use_container_width=True):
            if use_default_prompt == PROMPT_OPTIONS["CUSTOM"]:
                is_valid, error_msg = validate_custom_prompt(custom_prompt or "")
                if not is_valid:
                    st.error(f"❌ {error_msg}")
                    st.stop()

            st.session_state.selected_collection = selected_collection
            st.session_state.selected_profile = selected_profile
            st.session_state.custom_prompt = (
                custom_prompt if use_default_prompt == PROMPT_OPTIONS["CUSTOM"] else None
            )
            st.session_state.engine_initialized = False
            st.session_state.show_config = False
            st.rerun()

    show_example_queries(selected_collection)


def display_response_metadata(response):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Documentos Recuperados", response.retrieved_docs)

    with col2:
        st.metric("Fontes Encontradas", len(response.sources) if response.sources else 0)

    with col3:
        if response.similarity_scores:
            avg_score = sum(response.similarity_scores) / len(response.similarity_scores)
            st.metric("Score Médio", f"{avg_score:.3f}")


def render_container(container_type: str, content_func, **kwargs):
    container_key = f"{container_type.upper()}_CONTAINER"
    start_tag = UI_CONSTANTS.get(f"{container_key}_START")
    end_tag = UI_CONSTANTS.get(f"{container_key}_END")

    if start_tag:
        st.markdown(start_tag, unsafe_allow_html=True)

    if callable(content_func):
        content_func(**kwargs)
    else:
        st.write(content_func)

    if end_tag:
        st.markdown(end_tag, unsafe_allow_html=True)


def show_example_queries(collection_key: str):
    if collection_key in AVAILABLE_COLLECTIONS and AVAILABLE_COLLECTIONS[collection_key].get(
        "enabled"
    ):
        collection_config = AVAILABLE_COLLECTIONS[collection_key]
        queries = collection_config.get("example_queries", [])

        if queries:
            st.markdown("---")
            collection_name = collection_config["name"]
            st.subheader(f"📖 Exemplos de consultas para {collection_name}")

            for query in queries:
                st.write(f"• {query}")


def execute_query_with_custom_prompt(engine, query, documents_retrieve, custom_prompt=None):
    if custom_prompt:
        return engine.query_with_metadata(
            query, documents_retrieve=documents_retrieve, prompt_template=custom_prompt
        )
    else:
        return engine.query_with_metadata(query, documents_retrieve=documents_retrieve)


def main():
    if "show_config" not in st.session_state:
        st.session_state.show_config = True
    if "engine_initialized" not in st.session_state:
        st.session_state.engine_initialized = False
    if "selected_collection" not in st.session_state:
        st.session_state.selected_collection = "books"
    if "selected_profile" not in st.session_state:
        st.session_state.selected_profile = "default"
    if "custom_prompt" not in st.session_state:
        st.session_state.custom_prompt = None

    if st.session_state.show_config:
        show_configuration_page()
        return

    st.markdown(
        '<h1 class="main-header">📚 RAG Query Engine Interface</h1>', unsafe_allow_html=True
    )
    st.markdown("Interface gráfica para testar o RAG Query Engine")

    if not st.session_state.engine_initialized:
        custom_prompt = st.session_state.custom_prompt or ""
        engine, error, collection_config, profile_config = initialize_rag_engine(
            st.session_state.selected_collection, st.session_state.selected_profile, custom_prompt
        )

        if error:
            st.error(f"Erro ao inicializar o RAG Query Engine: {error}")
            if st.button("⚙️ Voltar para Configuração"):
                st.session_state.show_config = True
                st.rerun()
            st.stop()

        if engine is None:
            st.error("Falha ao inicializar o RAG Query Engine")
            if st.button("⚙️ Voltar para Configuração"):
                st.session_state.show_config = True
                st.rerun()
            st.stop()

        st.session_state.engine = engine
        st.session_state.collection_config = collection_config
        st.session_state.profile_config = profile_config
        st.session_state.engine_initialized = True

    engine = st.session_state.engine
    collection_config = st.session_state.collection_config
    profile_config = st.session_state.profile_config

    st.sidebar.header("⚙️ Configurações")

    with st.sidebar.expander("📋 Configuração Atual", expanded=True):
        if collection_config and profile_config:
            st.write(f"**Coleção:** {collection_config['name']}")
            st.write(f"**Perfil LLM:** {profile_config['name']}")

            if st.session_state.custom_prompt:
                st.write("**Prompt:** ✏️ Personalizado")
                with st.expander("Ver prompt personalizado"):
                    st.code(st.session_state.custom_prompt, language="text")
            else:
                st.write("**Prompt:** 🎯 Padrão")

        if st.button("🔄 Reconfigurar", use_container_width=True):
            st.session_state.show_config = True
            st.session_state.engine_initialized = False
            st.rerun()

    with st.sidebar.expander("ℹ️ Informações do Engine"):
        config_info = engine.get_config_info()
        for key, value in config_info.items():
            st.write(f"**{key}:** {value}")

    st.sidebar.subheader("Parâmetros da Consulta")
    documents_retrieve = st.sidebar.slider("Documentos a recuperar", 1, 10, 3)
    min_similarity_score = st.sidebar.slider("Score mínimo de similaridade", 0.0, 1.0, 0.3, 0.1)

    tab1, tab2, tab3 = st.tabs(
        [
            "🔍 Consulta Básica",
            "📊 Consulta com Metadados",
            "🎯 Consultas Múltiplas",
        ]
    )

    with tab1:
        st.markdown('<h2 class="section-header">Consulta Básica</h2>', unsafe_allow_html=True)
        st.write("Teste a funcionalidade básica de consulta que retorna apenas a resposta.")

        collection_config = st.session_state.get("collection_config", {})
        predefined_queries = collection_config.get("example_queries", PREDEFINED_QUERIES["books"])

        selected_query = st.selectbox(
            "Escolha uma consulta predefinida:",
            [""] + predefined_queries,
            key="basic_query_select",
        )

        query_input = st.text_area(
            "Ou digite sua própria consulta:",
            value=selected_query if selected_query else "",
            height=100,
            key="basic_query_input",
        )

        if st.button("🚀 Executar Consulta Básica", key="basic_query_btn"):
            if query_input.strip():
                with st.spinner("Processando consulta..."):
                    start_time = time.time()
                    try:
                        if st.session_state.custom_prompt:
                            response = engine.query_with_metadata(
                                query_input,
                                documents_retrieve=documents_retrieve,
                                prompt_template=st.session_state.custom_prompt,
                            )
                            answer = response.answer
                        else:
                            answer = engine.query(
                                query_input, documents_retrieve=documents_retrieve
                            )
                        end_time = time.time()

                        st.markdown('<div class="response-container">', unsafe_allow_html=True)
                        st.write("**Pergunta:**", query_input)
                        st.write("**Resposta:**", answer)
                        st.write(f"**Tempo de resposta:** {end_time - start_time:.2f} segundos")
                        st.write(f"**Tamanho da resposta:** {len(answer)} caracteres")
                        st.markdown("</div>", unsafe_allow_html=True)

                    except Exception as e:
                        st.markdown('<div class="error-container">', unsafe_allow_html=True)
                        st.error(f"Erro ao processar consulta: {str(e)}")
                        st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.warning("Por favor, digite uma consulta.")

    with tab2:
        st.markdown(
            '<h2 class="section-header">Consulta com Metadados</h2>', unsafe_allow_html=True
        )
        st.write("Consulta completa com informações detalhadas sobre a resposta.")

        selected_query_meta = st.selectbox(
            "Escolha uma consulta predefinida:", [""] + predefined_queries, key="meta_query_select"
        )

        query_input_meta = st.text_area(
            "Ou digite sua própria consulta:",
            value=selected_query_meta if selected_query_meta else "",
            height=100,
            key="meta_query_input",
        )

        if st.button("🔍 Executar Consulta com Metadados", key="meta_query_btn"):
            if query_input_meta.strip():
                with st.spinner("Processando consulta com metadados..."):
                    start_time = time.time()
                    try:
                        if st.session_state.custom_prompt:
                            response = engine.query_with_metadata(
                                query_input_meta,
                                documents_retrieve=documents_retrieve,
                                min_similarity_score=min_similarity_score,
                                prompt_template=st.session_state.custom_prompt,
                            )
                        else:
                            response = engine.query_with_metadata(
                                query_input_meta,
                                documents_retrieve=documents_retrieve,
                                min_similarity_score=min_similarity_score,
                            )
                        end_time = time.time()

                        st.markdown('<div class="response-container">', unsafe_allow_html=True)
                        st.write("**Pergunta:**", response.query)
                        st.write("**Resposta:**", response.answer)

                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                        display_response_metadata(response)
                        st.markdown("</div>", unsafe_allow_html=True)

                        if response.sources:
                            sources_str = ", ".join([s for s in response.sources if s is not None])
                            st.write("**Fontes:**", sources_str)

                        if response.similarity_scores:
                            st.write(
                                "**Scores de Similaridade:**",
                                [f"{score:.4f}" for score in response.similarity_scores],
                            )

                        st.write(f"**Tempo de resposta:** {end_time - start_time:.2f} segundos")
                        st.markdown("</div>", unsafe_allow_html=True)

                        st.subheader("📋 Resposta Formatada")
                        sources_str = (
                            ", ".join([s for s in response.sources if s is not None])
                            if response.sources
                            else ""
                        )
                        formatted_response = f"Response: {response.answer}\nSources: {sources_str}"
                        st.text_area("Resposta formatada:", formatted_response, height=150)

                    except Exception as e:
                        st.markdown('<div class="error-container">', unsafe_allow_html=True)
                        st.error(f"Erro ao processar consulta: {str(e)}")
                        st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.warning("Por favor, digite uma consulta.")

    with tab3:
        st.markdown(
            '<h2 class="section-header">Teste de Consultas Múltiplas</h2>', unsafe_allow_html=True
        )
        st.write("Teste múltiplas consultas de uma vez para comparar performance e resultados.")

        selected_queries = st.multiselect(
            "Selecione as consultas para testar:",
            predefined_queries,
            default=predefined_queries[:3],
        )

        if st.button("🎯 Executar Consultas Múltiplas", key="multi_query_btn"):
            if selected_queries:
                total_start_time = time.time()

                for i, test_query in enumerate(selected_queries, 1):
                    st.subheader(f"Consulta {i}")
                    st.write(f"**Pergunta:** {test_query}")

                    try:
                        with st.spinner(f"Processando consulta {i}..."):
                            query_start_time = time.time()
                            if st.session_state.custom_prompt:
                                response = engine.query_with_metadata(
                                    test_query,
                                    documents_retrieve=documents_retrieve,
                                    min_similarity_score=min_similarity_score,
                                    prompt_template=st.session_state.custom_prompt,
                                )
                            else:
                                response = engine.query_with_metadata(
                                    test_query,
                                    documents_retrieve=documents_retrieve,
                                    min_similarity_score=min_similarity_score,
                                )
                            query_end_time = time.time()

                        col1, col2 = st.columns([2, 1])

                        with col1:
                            answer_preview = (
                                response.answer[:200] + "..."
                                if len(response.answer) > 200
                                else response.answer
                            )
                            st.write(f"**Resposta:** {answer_preview}")

                        with col2:
                            st.metric("Docs Recuperados", response.retrieved_docs)
                            st.metric("Tempo (s)", f"{query_end_time - query_start_time:.2f}")

                        if response.sources:
                            sources_str = ", ".join([s for s in response.sources if s is not None])
                            st.write(f"**Fontes:** {sources_str}")

                    except Exception as e:
                        st.error(f"Erro na consulta {i}: {str(e)}")

                    st.divider()

                total_end_time = time.time()
                st.success(
                    f"✅ Todas as consultas processadas em {total_end_time - total_start_time:.2f} segundos"
                )
            else:
                st.warning("Por favor, selecione pelo menos uma consulta.")

    st.divider()
    st.markdown("---")
    st.markdown(f"**RAG Query Engine Interface** | Engine: {engine}")


if __name__ == "__main__":
    main()

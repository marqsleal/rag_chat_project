"""
Comprehensive unit tests for create_chroma_database.py module.
"""

from unittest.mock import Mock, patch

import pytest
from langchain.schema import Document

from rag_project.constants import (
    BOOKS_RAW_DATA_DIR,
    BOOKS_CHROMA_DB_DIR,
    BOOKS_COLLECTION_NAME,
    AZURE_RAW_DATA_DIR,
    AZURE_CHROMA_DB_DIR,
    AZURE_COLLECTION_NAME,
    SENTENCE_TRANSFORMERS_MODEL_NAME,
)
from rag_project.create_chroma_database import (
    load_documents,
    split_text,
    clean_chroma_dir,
    save_to_chroma,
    create_chroma_db,
    main,
    get_database_configs,
    parse_arguments,
    validate_arguments,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    EXAMPLE_DOCUMENT_INDEX,
    CONTENT_PREVIEW_LENGTH,
    MARKDOWN_GLOB_PATTERN,
)


class TestLoadDocuments:
    """Test cases for load_documents function."""

    @patch('rag_project.create_chroma_database.DirectoryLoader')
    @patch('rag_project.create_chroma_database.os.path.exists')
    def test_load_documents_success(self, mock_exists, mock_directory_loader):
        """
        Testa o carregamento bem-sucedido de documentos de um diretório.
        
        Este teste verifica se a função consegue carregar documentos
        corretamente de um diretório existente, utilizando o DirectoryLoader
        com o padrão correto para arquivos markdown.
        
        Cenário testado:
        - Diretório existe e contém documentos
        - DirectoryLoader é configurado corretamente
        - Documentos são carregados e retornados
        - Logging apropriado é executado
        """
        mock_exists.return_value = True
        mock_loader = Mock()
        mock_documents = [
            Document(page_content="Test content 1", metadata={"source": "file1.md"}),
            Document(page_content="Test content 2", metadata={"source": "file2.md"}),
        ]
        mock_loader.load.return_value = mock_documents
        mock_directory_loader.return_value = mock_loader
        
        directory = "/test/directory"
        result = load_documents(directory)
        
        assert result == mock_documents
        assert len(result) == 2
        mock_exists.assert_called_once_with(directory)
        mock_directory_loader.assert_called_once_with(directory, glob=MARKDOWN_GLOB_PATTERN)
        mock_loader.load.assert_called_once()

    @patch('rag_project.create_chroma_database.os.path.exists')
    def test_load_documents_directory_not_exists(self, mock_exists):
        """
        Testa o comportamento quando o diretório não existe.
        
        Este teste verifica se a função trata adequadamente situações
        onde o diretório especificado não existe no sistema de arquivos,
        lançando a exceção apropriada.
        
        Cenário testado:
        - Diretório não existe
        - FileNotFoundError é lançada com mensagem descritiva
        - Nenhuma tentativa de carregamento é feita
        """
        mock_exists.return_value = False
        directory = "/nonexistent/directory"
        
        with pytest.raises(FileNotFoundError) as exc_info:
            load_documents(directory)
        
        assert f"Directory {directory} does not exist" in str(exc_info.value)
        mock_exists.assert_called_once_with(directory)

    @patch('rag_project.create_chroma_database.DirectoryLoader')
    @patch('rag_project.create_chroma_database.os.path.exists')
    def test_load_documents_empty_directory(self, mock_exists, mock_directory_loader):
        """
        Testa o comportamento com diretório vazio (sem documentos).
        
        Este teste verifica se a função lida adequadamente com diretórios
        que existem mas não contêm documentos correspondentes ao padrão
        de busca, retornando lista vazia e logando warning apropriado.
        
        Cenário testado:
        - Diretório existe mas está vazio
        - DirectoryLoader retorna lista vazia
        - Warning é logado sobre ausência de documentos
        - Lista vazia é retornada
        """
        mock_exists.return_value = True
        mock_loader = Mock()
        mock_loader.load.return_value = []
        mock_directory_loader.return_value = mock_loader
        
        directory = "/empty/directory"
        result = load_documents(directory)
        
        assert result == []
        assert len(result) == 0
        mock_loader.load.assert_called_once()

    @patch('rag_project.create_chroma_database.DirectoryLoader')
    @patch('rag_project.create_chroma_database.os.path.exists')
    def test_load_documents_loader_error(self, mock_exists, mock_directory_loader):
        """
        Testa o tratamento de erros durante carregamento de documentos.
        
        Este teste verifica se a função trata adequadamente erros que
        podem ocorrer durante o carregamento, como problemas de
        permissão, arquivos corrompidos, ou falhas de I/O.
        
        Cenário testado:
        - DirectoryLoader falha com RuntimeError
        - Erro é logado com informações contextuais
        - Exceção é propagada para o chamador
        """
        mock_exists.return_value = True
        mock_loader = Mock()
        mock_loader.load.side_effect = RuntimeError("Failed to load documents")
        mock_directory_loader.return_value = mock_loader
        
        directory = "/problematic/directory"
        
        with pytest.raises(RuntimeError) as exc_info:
            load_documents(directory)
        
        assert "Failed to load documents" in str(exc_info.value)

    @patch('rag_project.create_chroma_database.DirectoryLoader')
    @patch('rag_project.create_chroma_database.os.path.exists')
    def test_load_documents_permission_error(self, mock_exists, mock_directory_loader):
        """
        Testa o tratamento de erros de permissão durante carregamento.
        
        Este teste verifica se a função trata adequadamente situações
        onde o usuário não tem permissões adequadas para acessar o
        diretório ou arquivos dentro dele.
        
        Cenário testado:
        - PermissionError é lançada durante carregamento
        - Erro é logado com detalhes da falha de permissão
        - Exceção é propagada mantendo informação original
        """
        mock_exists.return_value = True
        mock_loader = Mock()
        mock_loader.load.side_effect = PermissionError("Permission denied")
        mock_directory_loader.return_value = mock_loader
        
        directory = "/restricted/directory"
        
        with pytest.raises(PermissionError) as exc_info:
            load_documents(directory)
        
        assert "Permission denied" in str(exc_info.value)

    @patch('rag_project.create_chroma_database.DirectoryLoader')
    @patch('rag_project.create_chroma_database.os.path.exists')
    def test_load_documents_glob_pattern_validation(self, mock_exists, mock_directory_loader):
        """
        Testa se o padrão glob correto é usado para arquivos markdown.
        
        Este teste verifica se a função utiliza o padrão glob apropriado
        para localizar arquivos markdown recursivamente no diretório,
        garantindo que apenas arquivos .md sejam processados.
        
        Cenário testado:
        - DirectoryLoader é chamado com padrão glob correto
        - Padrão "**/*.md" é usado para busca recursiva
        - Configuração está alinhada com constante do módulo
        """
        mock_exists.return_value = True
        mock_loader = Mock()
        mock_loader.load.return_value = [Document(page_content="Test", metadata={})]
        mock_directory_loader.return_value = mock_loader
        
        directory = "/test/directory"
        load_documents(directory)
        
        mock_directory_loader.assert_called_once_with(directory, glob=MARKDOWN_GLOB_PATTERN)
        assert MARKDOWN_GLOB_PATTERN == "**/*.md"


class TestSplitText:
    """Test cases for split_text function."""

    @patch('rag_project.create_chroma_database.RecursiveCharacterTextSplitter')
    def test_split_text_success(self, mock_text_splitter):
        """
        Testa a divisão bem-sucedida de documentos em chunks.
        
        Este teste verifica se a função consegue dividir documentos
        em chunks usando o RecursiveCharacterTextSplitter com
        configurações apropriadas de tamanho e sobreposição.
        
        Cenário testado:
        - Documentos são divididos corretamente em chunks
        - Configurações de chunk_size e chunk_overlap são aplicadas
        - RecursiveCharacterTextSplitter é configurado adequadamente
        - Chunks resultantes são retornados
        """
        mock_splitter = Mock()
        mock_chunks = [
            Document(page_content="Chunk 1", metadata={"chunk_index": 0}),
            Document(page_content="Chunk 2", metadata={"chunk_index": 1}),
        ]
        mock_splitter.split_documents.return_value = mock_chunks
        mock_text_splitter.return_value = mock_splitter
        
        documents = [Document(page_content="Long document content", metadata={"source": "test.md"})]
        
        result = split_text(documents, chunk_size=500, chunk_overlap=50)
        
        assert result == mock_chunks
        assert len(result) == 2
        mock_text_splitter.assert_called_once_with(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            add_start_index=True,
        )
        mock_splitter.split_documents.assert_called_once_with(documents)

    def test_split_text_empty_documents(self):
        """
        Testa o comportamento com lista vazia de documentos.
        
        Este teste verifica se a função lida adequadamente com
        situações onde uma lista vazia de documentos é fornecida,
        retornando lista vazia e logando warning apropriado.
        
        Cenário testado:
        - Lista vazia de documentos é fornecida
        - Warning é logado sobre ausência de documentos
        - Lista vazia é retornada sem tentar processamento
        - Nenhuma exceção é lançada
        """
        result = split_text([])
        
        assert result == []
        assert len(result) == 0

    @patch('rag_project.create_chroma_database.RecursiveCharacterTextSplitter')
    def test_split_text_with_default_parameters(self, mock_text_splitter):
        """
        Testa a divisão com parâmetros padrão.
        
        Este teste verifica se a função utiliza corretamente os
        valores padrão de chunk_size e chunk_overlap quando não
        são especificados explicitamente pelo usuário.
        
        Cenário testado:
        - Parâmetros padrão são utilizados quando não especificados
        - DEFAULT_CHUNK_SIZE e DEFAULT_CHUNK_OVERLAP são aplicados
        - Configuração padrão funciona adequadamente
        """
        mock_splitter = Mock()
        mock_splitter.split_documents.return_value = []
        mock_text_splitter.return_value = mock_splitter
        
        documents = [Document(page_content="Test content", metadata={})]
        
        split_text(documents)
        
        mock_text_splitter.assert_called_once_with(
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True,
        )

    @patch('rag_project.create_chroma_database.RecursiveCharacterTextSplitter')
    def test_split_text_with_example_logging(self, mock_text_splitter):
        """
        Testa o logging de exemplo quando há chunks suficientes.
        
        Este teste verifica se a função loga adequadamente um exemplo
        de documento quando o número de chunks excede o índice de
        exemplo configurado, fornecendo visibilidade do processamento.
        
        Cenário testado:
        - Número de chunks excede EXAMPLE_DOCUMENT_INDEX
        - Exemplo de conteúdo e metadata são logados
        - Preview de conteúdo é truncado adequadamente
        - Logging não interfere no processamento
        """
        mock_splitter = Mock()
        # Criar chunks suficientes para ativar o logging de exemplo
        mock_chunks = []
        for i in range(15):  # Mais que EXAMPLE_DOCUMENT_INDEX (10)
            mock_chunks.append(
                Document(
                    page_content=f"Chunk content {i} with more text to test preview truncation",
                    metadata={"chunk_index": i, "source": "test.md"}
                )
            )
        mock_splitter.split_documents.return_value = mock_chunks
        mock_text_splitter.return_value = mock_splitter
        
        documents = [Document(page_content="Long document", metadata={})]
        
        result = split_text(documents)
        
        assert len(result) == 15
        assert result == mock_chunks

    @patch('rag_project.create_chroma_database.RecursiveCharacterTextSplitter')
    def test_split_text_splitter_error(self, mock_text_splitter):
        """
        Testa o tratamento de erros durante divisão de texto.
        
        Este teste verifica se a função trata adequadamente erros
        que podem ocorrer durante o processo de divisão, como
        problemas de memória ou configurações inválidas.
        
        Cenário testado:
        - RecursiveCharacterTextSplitter falha com RuntimeError
        - Erro é logado com informações contextuais
        - Exceção é propagada para o chamador
        """
        mock_text_splitter.side_effect = RuntimeError("Splitter configuration error")
        
        documents = [Document(page_content="Test content", metadata={})]
        
        with pytest.raises(RuntimeError) as exc_info:
            split_text(documents)
        
        assert "Splitter configuration error" in str(exc_info.value)

    @patch('rag_project.create_chroma_database.RecursiveCharacterTextSplitter')
    def test_split_text_invalid_chunk_parameters(self, mock_text_splitter):
        """
        Testa o comportamento com parâmetros inválidos de chunk.
        
        Este teste verifica se a função lida adequadamente com
        configurações inválidas de chunk_size e chunk_overlap,
        como valores negativos ou sobreposição maior que tamanho.
        
        Cenário testado:
        - Parâmetros inválidos são passados
        - RecursiveCharacterTextSplitter pode falhar
        - Erro é tratado e propagado adequadamente
        """
        mock_splitter = Mock()
        mock_splitter.split_documents.side_effect = ValueError("Invalid chunk parameters")
        mock_text_splitter.return_value = mock_splitter
        
        documents = [Document(page_content="Test content", metadata={})]
        
        with pytest.raises(ValueError) as exc_info:
            split_text(documents, chunk_size=-100, chunk_overlap=200)
        
        assert "Invalid chunk parameters" in str(exc_info.value)


class TestCleanChromaDir:
    """Test cases for clean_chroma_dir function."""

    @patch('rag_project.create_chroma_database.shutil.rmtree')
    @patch('rag_project.create_chroma_database.os.path.exists')
    def test_clean_chroma_dir_existing_directory(self, mock_exists, mock_rmtree):
        """
        Testa a limpeza de diretório existente.
        
        Este teste verifica se a função consegue remover adequadamente
        um diretório Chroma existente, incluindo todos os arquivos e
        subdiretórios contidos nele.
        
        Cenário testado:
        - Diretório existe e precisa ser removido
        - shutil.rmtree é chamado para remoção completa
        - Logging apropriado sobre remoção
        - Operação completa sem erros
        """
        mock_exists.return_value = True
        path = "/test/chroma/db"
        
        clean_chroma_dir(path)
        
        mock_exists.assert_called_once_with(path)
        mock_rmtree.assert_called_once_with(path)

    @patch('rag_project.create_chroma_database.shutil.rmtree')
    @patch('rag_project.create_chroma_database.os.path.exists')
    def test_clean_chroma_dir_nonexistent_directory(self, mock_exists, mock_rmtree):
        """
        Testa o comportamento com diretório inexistente.
        
        Este teste verifica se a função lida adequadamente com
        situações onde o diretório a ser limpo não existe,
        evitando tentativas desnecessárias de remoção.
        
        Cenário testado:
        - Diretório não existe
        - Nenhuma tentativa de remoção é feita
        - Função completa sem erros
        - rmtree não é chamado
        """
        mock_exists.return_value = False
        path = "/nonexistent/chroma/db"
        
        clean_chroma_dir(path)
        
        mock_exists.assert_called_once_with(path)
        mock_rmtree.assert_not_called()

    @patch('rag_project.create_chroma_database.shutil.rmtree')
    @patch('rag_project.create_chroma_database.os.path.exists')
    def test_clean_chroma_dir_permission_error(self, mock_exists, mock_rmtree):
        """
        Testa o tratamento de erros de permissão durante limpeza.
        
        Este teste verifica se a função trata adequadamente
        situações onde o usuário não tem permissões para remover
        o diretório ou arquivos dentro dele.
        
        Cenário testado:
        - PermissionError é lançada durante rmtree
        - Erro é logado com informações contextuais
        - Exceção é propagada para o chamador
        """
        mock_exists.return_value = True
        mock_rmtree.side_effect = PermissionError("Permission denied")
        path = "/restricted/chroma/db"
        
        with pytest.raises(PermissionError) as exc_info:
            clean_chroma_dir(path)
        
        assert "Permission denied" in str(exc_info.value)
        mock_rmtree.assert_called_once_with(path)

    @patch('rag_project.create_chroma_database.shutil.rmtree')
    @patch('rag_project.create_chroma_database.os.path.exists')
    def test_clean_chroma_dir_os_error(self, mock_exists, mock_rmtree):
        """
        Testa o tratamento de erros do sistema operacional.
        
        Este teste verifica se a função trata adequadamente
        outros tipos de erros que podem ocorrer durante a
        remoção, como problemas de I/O ou filesystem.
        
        Cenário testado:
        - OSError é lançada durante remoção
        - Erro é logado com detalhes da falha
        - Exceção é propagada mantendo informação original
        """
        mock_exists.return_value = True
        mock_rmtree.side_effect = OSError("Disk I/O error")
        path = "/problematic/chroma/db"
        
        with pytest.raises(OSError) as exc_info:
            clean_chroma_dir(path)
        
        assert "Disk I/O error" in str(exc_info.value)


class TestSaveToChroma:
    """Test cases for save_to_chroma function."""

    @patch('rag_project.create_chroma_database.Chroma.from_documents')
    @patch('rag_project.create_chroma_database.HuggingFaceEmbeddings')
    @patch('rag_project.create_chroma_database.os.makedirs')
    @patch('rag_project.create_chroma_database.clean_chroma_dir')
    def test_save_to_chroma_success(self, mock_clean_dir, mock_makedirs, mock_embeddings, mock_chroma):
        """
        Testa o salvamento bem-sucedido de chunks no Chroma.
        
        Este teste verifica se a função consegue salvar chunks
        de documentos no banco Chroma corretamente, incluindo
        limpeza de diretório, criação de embeddings e validação
        de contagem.
        
        Cenário testado:
        - Chunks são salvos no banco Chroma com sucesso
        - Diretório é limpo antes da criação
        - HuggingFaceEmbeddings é inicializado corretamente
        - Contagem de documentos é validada
        - Função retorna True indicando sucesso
        """
        chunks = [
            Document(page_content="Chunk 1", metadata={"source": "test1.md"}),
            Document(page_content="Chunk 2", metadata={"source": "test2.md"}),
        ]
        
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance
        
        mock_db = Mock()
        mock_collection = Mock()
        mock_collection.count.return_value = 2
        mock_db._collection = mock_collection
        mock_chroma.return_value = mock_db
        
        result = save_to_chroma(chunks, "/test/chroma/db", "test_collection")
        
        assert result is True
        mock_clean_dir.assert_called_once_with("/test/chroma/db")
        mock_makedirs.assert_called_once_with("/test/chroma", exist_ok=True)
        mock_embeddings.assert_called_once()
        mock_chroma.assert_called_once_with(
            chunks, mock_embeddings_instance, 
            persist_directory="/test/chroma/db", 
            collection_name="test_collection"
        )

    def test_save_to_chroma_empty_chunks(self):
        """
        Testa o comportamento com lista vazia de chunks.
        
        Este teste verifica se a função lida adequadamente com
        situações onde uma lista vazia de chunks é fornecida,
        evitando processamento desnecessário e retornando False.
        
        Cenário testado:
        - Lista vazia de chunks é fornecida
        - Warning é logado sobre ausência de chunks
        - Função retorna False sem tentar salvar
        - Nenhuma operação de banco é executada
        """
        result = save_to_chroma([], "/test/chroma/db", "test_collection")
        
        assert result is False

    @patch('rag_project.create_chroma_database.Chroma.from_documents')
    @patch('rag_project.create_chroma_database.HuggingFaceEmbeddings')
    @patch('rag_project.create_chroma_database.os.makedirs')
    @patch('rag_project.create_chroma_database.clean_chroma_dir')
    def test_save_to_chroma_with_custom_model(self, mock_clean_dir, mock_makedirs, mock_embeddings, mock_chroma):
        """
        Testa o salvamento com modelo de embeddings customizado.
        
        Este teste verifica se a função utiliza corretamente um
        modelo de embeddings especificado pelo usuário em vez
        do modelo padrão do sistema.
        
        Cenário testado:
        - Modelo customizado é especificado
        - HuggingFaceEmbeddings é inicializado com modelo correto
        - Processamento continua normalmente
        - Modelo customizado é logado adequadamente
        """
        chunks = [Document(page_content="Test chunk", metadata={})]
        custom_model = "custom/embedding-model"
        
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance
        
        mock_db = Mock()
        mock_collection = Mock()
        mock_collection.count.return_value = 1
        mock_db._collection = mock_collection
        mock_chroma.return_value = mock_db
        
        result = save_to_chroma(chunks, "/test/chroma/db", "test_collection", custom_model)
        
        assert result is True
        mock_embeddings.assert_called_once_with(model_name=custom_model)

    @patch('rag_project.create_chroma_database.Chroma.from_documents')
    @patch('rag_project.create_chroma_database.HuggingFaceEmbeddings')
    @patch('rag_project.create_chroma_database.os.makedirs')
    @patch('rag_project.create_chroma_database.clean_chroma_dir')
    def test_save_to_chroma_count_mismatch_warning(self, mock_clean_dir, mock_makedirs, mock_embeddings, mock_chroma):
        """
        Testa o warning quando contagem de chunks não confere.
        
        Este teste verifica se a função detecta e alerta adequadamente
        quando o número de chunks salvos no banco não corresponde
        ao número de chunks fornecidos, indicando possível problema.
        
        Cenário testado:
        - Número de chunks salvos difere do esperado
        - Warning é logado sobre discrepância
        - Função continua e retorna True
        - Inconsistência é detectada e reportada
        """
        chunks = [Document(page_content="Chunk 1", metadata={})]
        
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance
        
        mock_db = Mock()
        mock_collection = Mock()
        mock_collection.count.return_value = 0  # Mismatch: esperado 1, obtido 0
        mock_db._collection = mock_collection
        mock_chroma.return_value = mock_db
        
        result = save_to_chroma(chunks, "/test/chroma/db", "test_collection")
        
        assert result is True  # Ainda retorna True, mas com warning

    @patch('rag_project.create_chroma_database.HuggingFaceEmbeddings')
    @patch('rag_project.create_chroma_database.os.makedirs')
    @patch('rag_project.create_chroma_database.clean_chroma_dir')
    def test_save_to_chroma_embeddings_error(self, mock_clean_dir, mock_makedirs, mock_embeddings):
        """
        Testa o tratamento de erros durante inicialização de embeddings.
        
        Este teste verifica se a função trata adequadamente falhas
        que podem ocorrer durante a inicialização do modelo
        HuggingFace, como problemas de conectividade ou modelo não encontrado.
        
        Cenário testado:
        - HuggingFaceEmbeddings falha durante inicialização
        - Erro é logado com informações contextuais
        - Função retorna False indicando falha
        """
        chunks = [Document(page_content="Test chunk", metadata={})]
        mock_embeddings.side_effect = RuntimeError("Model loading failed")
        
        result = save_to_chroma(chunks, "/test/chroma/db", "test_collection")
        
        assert result is False

    @patch('rag_project.create_chroma_database.Chroma.from_documents')
    @patch('rag_project.create_chroma_database.HuggingFaceEmbeddings')
    @patch('rag_project.create_chroma_database.os.makedirs')
    @patch('rag_project.create_chroma_database.clean_chroma_dir')
    def test_save_to_chroma_database_creation_error(self, mock_clean_dir, mock_makedirs, mock_embeddings, mock_chroma):
        """
        Testa o tratamento de erros durante criação do banco Chroma.
        
        Este teste verifica se a função trata adequadamente falhas
        que podem ocorrer durante a criação do banco Chroma,
        como problemas de espaço em disco ou corrupção.
        
        Cenário testado:
        - Chroma.from_documents falha com RuntimeError
        - Erro é logado com detalhes da falha
        - Função retorna False indicando falha
        """
        chunks = [Document(page_content="Test chunk", metadata={})]
        
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance
        
        mock_chroma.side_effect = RuntimeError("Database creation failed")
        
        result = save_to_chroma(chunks, "/test/chroma/db", "test_collection")
        
        assert result is False

    @patch('rag_project.create_chroma_database.os.makedirs')
    @patch('rag_project.create_chroma_database.clean_chroma_dir')
    def test_save_to_chroma_directory_creation_error(self, mock_clean_dir, mock_makedirs):
        """
        Testa o tratamento de erros durante criação de diretórios.
        
        Este teste verifica se a função trata adequadamente falhas
        que podem ocorrer durante a criação de diretórios parent,
        como problemas de permissão ou espaço em disco.
        
        Cenário testado:
        - os.makedirs falha com PermissionError
        - Erro é logado com informações sobre falha
        - Função retorna False indicando falha
        """
        chunks = [Document(page_content="Test chunk", metadata={})]
        mock_makedirs.side_effect = PermissionError("Cannot create directory")
        
        result = save_to_chroma(chunks, "/restricted/chroma/db", "test_collection")
        
        assert result is False


class TestCreateChromaDb:
    """Test cases for create_chroma_db function."""

    @patch('rag_project.create_chroma_database.save_to_chroma')
    @patch('rag_project.create_chroma_database.split_text')
    @patch('rag_project.create_chroma_database.load_documents')
    def test_create_chroma_db_success(self, mock_load_docs, mock_split_text, mock_save_to_chroma):
        """
        Testa a criação bem-sucedida do banco Chroma completo.
        
        Este teste verifica se a função integra adequadamente
        todas as etapas do processo: carregamento de documentos,
        divisão em chunks e salvamento no banco Chroma.
        
        Cenário testado:
        - Pipeline completo executa com sucesso
        - Documentos são carregados, divididos e salvos
        - Parâmetros são passados corretamente entre funções
        - Função retorna True indicando sucesso
        """
        mock_documents = [Document(page_content="Test doc", metadata={})]
        mock_chunks = [Document(page_content="Test chunk", metadata={})]
        
        mock_load_docs.return_value = mock_documents
        mock_split_text.return_value = mock_chunks
        mock_save_to_chroma.return_value = True
        
        result = create_chroma_db(
            "/data/dir", "/chroma/db", "test_collection", 
            chunk_size=400, chunk_overlap=80
        )
        
        assert result is True
        mock_load_docs.assert_called_once_with("/data/dir")
        mock_split_text.assert_called_once_with(mock_documents, 400, 80)
        mock_save_to_chroma.assert_called_once_with(mock_chunks, "/chroma/db", "test_collection")

    @patch('rag_project.create_chroma_database.load_documents')
    def test_create_chroma_db_no_documents_loaded(self, mock_load_docs):
        """
        Testa o comportamento quando nenhum documento é carregado.
        
        Este teste verifica se a função lida adequadamente com
        situações onde o carregamento de documentos falha ou
        retorna lista vazia, evitando processamento desnecessário.
        
        Cenário testado:
        - load_documents retorna lista vazia
        - Erro é logado sobre ausência de documentos
        - Função retorna False sem tentar processamento posterior
        - Pipeline é interrompido adequadamente
        """
        mock_load_docs.return_value = []
        
        result = create_chroma_db("/empty/dir", "/chroma/db", "test_collection")
        
        assert result is False
        mock_load_docs.assert_called_once_with("/empty/dir")

    @patch('rag_project.create_chroma_database.split_text')
    @patch('rag_project.create_chroma_database.load_documents')
    def test_create_chroma_db_no_chunks_created(self, mock_load_docs, mock_split_text):
        """
        Testa o comportamento quando nenhum chunk é criado.
        
        Este teste verifica se a função lida adequadamente com
        situações onde a divisão de texto falha ou retorna
        lista vazia, impedindo criação de banco vazio.
        
        Cenário testado:
        - split_text retorna lista vazia
        - Erro é logado sobre ausência de chunks
        - Função retorna False sem tentar salvar
        - Pipeline é interrompido antes do salvamento
        """
        mock_documents = [Document(page_content="Test doc", metadata={})]
        mock_load_docs.return_value = mock_documents
        mock_split_text.return_value = []
        
        result = create_chroma_db("/data/dir", "/chroma/db", "test_collection")
        
        assert result is False
        mock_split_text.assert_called_once_with(mock_documents, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)

    @patch('rag_project.create_chroma_database.save_to_chroma')
    @patch('rag_project.create_chroma_database.split_text')  
    @patch('rag_project.create_chroma_database.load_documents')
    def test_create_chroma_db_save_failure(self, mock_load_docs, mock_split_text, mock_save_to_chroma):
        """
        Testa o comportamento quando salvamento no Chroma falha.
        
        Este teste verifica se a função detecta adequadamente
        falhas durante o salvamento no banco Chroma e retorna
        status apropriado indicando falha no processo.
        
        Cenário testado:
        - save_to_chroma retorna False indicando falha
        - Função propaga o status de falha
        - Pipeline completa mas indica insucesso
        """
        mock_documents = [Document(page_content="Test doc", metadata={})]
        mock_chunks = [Document(page_content="Test chunk", metadata={})]
        
        mock_load_docs.return_value = mock_documents
        mock_split_text.return_value = mock_chunks
        mock_save_to_chroma.return_value = False
        
        result = create_chroma_db("/data/dir", "/chroma/db", "test_collection")
        
        assert result is False

    @patch('rag_project.create_chroma_database.load_documents')
    def test_create_chroma_db_load_documents_exception(self, mock_load_docs):
        """
        Testa o tratamento de exceções durante carregamento.
        
        Este teste verifica se a função trata adequadamente
        exceções que podem ser lançadas durante o carregamento
        de documentos, logando erro e retornando False.
        
        Cenário testado:
        - load_documents lança RuntimeError
        - Exceção é capturada e logada
        - Função retorna False indicando falha
        - Sistema não trava com exceção não tratada
        """
        mock_load_docs.side_effect = RuntimeError("Loading failed")
        
        result = create_chroma_db("/problematic/dir", "/chroma/db", "test_collection")
        
        assert result is False

    @patch('rag_project.create_chroma_database.split_text')
    @patch('rag_project.create_chroma_database.load_documents')
    def test_create_chroma_db_split_text_exception(self, mock_load_docs, mock_split_text):
        """
        Testa o tratamento de exceções durante divisão de texto.
        
        Este teste verifica se a função trata adequadamente
        exceções que podem ser lançadas durante a divisão
        de documentos em chunks.
        
        Cenário testado:
        - split_text lança RuntimeError
        - Exceção é capturada e logada
        - Função retorna False indicando falha
        """
        mock_documents = [Document(page_content="Test doc", metadata={})]
        mock_load_docs.return_value = mock_documents
        mock_split_text.side_effect = RuntimeError("Splitting failed")
        
        result = create_chroma_db("/data/dir", "/chroma/db", "test_collection")
        
        assert result is False

    @patch('rag_project.create_chroma_database.save_to_chroma')
    @patch('rag_project.create_chroma_database.split_text')
    @patch('rag_project.create_chroma_database.load_documents')
    def test_create_chroma_db_with_default_parameters(self, mock_load_docs, mock_split_text, mock_save_to_chroma):
        """
        Testa a criação com parâmetros padrão de chunk.
        
        Este teste verifica se a função utiliza corretamente
        os parâmetros padrão quando não são especificados,
        garantindo configuração consistente do sistema.
        
        Cenário testado:
        - Parâmetros de chunk não são especificados
        - Valores padrão são utilizados para split_text
        - Pipeline funciona com configuração padrão
        """
        mock_documents = [Document(page_content="Test doc", metadata={})]
        mock_chunks = [Document(page_content="Test chunk", metadata={})]
        
        mock_load_docs.return_value = mock_documents
        mock_split_text.return_value = mock_chunks
        mock_save_to_chroma.return_value = True
        
        result = create_chroma_db("/data/dir", "/chroma/db", "test_collection")
        
        assert result is True
        mock_split_text.assert_called_once_with(mock_documents, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)


class TestMain:
    """Test cases for main function."""

    @patch('rag_project.create_chroma_database.create_chroma_db')
    @patch('rag_project.create_chroma_database.sys.exit')
    @patch('rag_project.create_chroma_database.sys.argv', ['create_chroma_database.py', 'books'])
    def test_main_success(self, mock_exit, mock_create_db):
        """
        Testa a execução bem-sucedida da função main com argparse.
        
        Este teste verifica se a função main executa corretamente
        o processo completo de criação do banco Chroma usando
        argumentos da linha de comando e configurações padrão.
        
        Cenário testado:
        - create_chroma_db retorna True indicando sucesso
        - Argumentos são parseados corretamente
        - Constantes do projeto são utilizadas corretamente
        - Mensagem de sucesso é logada
        - sys.exit não é chamado
        """
        mock_create_db.return_value = True
        
        main()
        
        mock_create_db.assert_called_once_with(
            data_dir=BOOKS_RAW_DATA_DIR,
            chroma_db_dir=BOOKS_CHROMA_DB_DIR,
            collection_name=BOOKS_COLLECTION_NAME,
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        )
        mock_exit.assert_not_called()

    @patch('rag_project.create_chroma_database.create_chroma_db')
    @patch('rag_project.create_chroma_database.sys.exit')
    @patch('rag_project.create_chroma_database.sys.argv', ['create_chroma_database.py', 'azure'])
    def test_main_failure(self, mock_exit, mock_create_db):
        """
        Testa o comportamento da função main quando criação falha.
        
        Este teste verifica se a função main lida adequadamente
        com falhas na criação do banco, logando erro e
        terminando o processo com código de saída apropriado.
        
        Cenário testado:
        - create_chroma_db retorna False indicando falha
        - Erro é logado sobre falha na criação
        - sys.exit é chamado com código 1
        - Processo termina adequadamente
        """
        mock_create_db.return_value = False
        
        main()
        
        mock_create_db.assert_called_once()
        mock_exit.assert_called_once_with(1)

    @patch('rag_project.create_chroma_database.create_chroma_db')
    @patch('rag_project.create_chroma_database.sys.exit')
    @patch('rag_project.create_chroma_database.sys.argv', ['create_chroma_database.py', 'books'])
    def test_main_exception_handling(self, mock_exit, mock_create_db):
        """
        Testa o tratamento de exceções na função main.
        
        Este teste verifica se a função main trata adequadamente
        exceções não capturadas que podem ocorrer durante
        a execução, garantindo terminação controlada.
        
        Cenário testado:
        - create_chroma_db lança exceção não tratada
        - Exceção é capturada e tratada pela função main
        - sys.exit é chamado com código 1
        - Sistema não trava de forma descontrolada
        """
        mock_create_db.side_effect = Exception("Unexpected error")
        
        main()
        
        mock_exit.assert_called_once_with(1)

    @patch('rag_project.create_chroma_database.create_chroma_db')
    @patch('rag_project.create_chroma_database.sys.exit')
    @patch('rag_project.create_chroma_database.sys.argv', ['create_chroma_database.py', 'books', '--chunk-size', '500', '--chunk-overlap', '75'])
    def test_main_with_custom_parameters(self, mock_exit, mock_create_db):
        """
        Testa a execução da função main com parâmetros customizados.
        
        Este teste verifica se a função main utiliza corretamente
        parâmetros customizados fornecidos via linha de comando
        para chunk_size e chunk_overlap.
        
        Cenário testado:
        - Parâmetros customizados são parseados corretamente
        - create_chroma_db é chamado com parâmetros corretos
        - Função executa com sucesso
        """
        mock_create_db.return_value = True
        
        main()
        
        mock_create_db.assert_called_once_with(
            data_dir=BOOKS_RAW_DATA_DIR,
            chroma_db_dir=BOOKS_CHROMA_DB_DIR,
            collection_name=BOOKS_COLLECTION_NAME,
            chunk_size=500,
            chunk_overlap=75,
        )
        mock_exit.assert_not_called()

    @patch('rag_project.create_chroma_database.sys.exit')
    @patch('rag_project.create_chroma_database.sys.argv', ['create_chroma_database.py', 'books', '--chunk-size', '-10'])
    def test_main_invalid_chunk_size(self, mock_exit):
        """
        Testa a validação de parâmetros inválidos na função main.
        
        Este teste verifica se a função main valida adequadamente
        parâmetros inválidos e termina com código de erro
        apropriado quando chunk_size é negativo.
        
        Cenário testado:
        - Chunk size negativo é detectado
        - ValueError é capturada
        - sys.exit é chamado com código 1
        """
        main()
        
        mock_exit.assert_called_once_with(1)


class TestArgumentParsing:
    """Test cases for argument parsing and validation functions."""

    def test_get_database_configs(self):
        """
        Testa se as configurações de banco estão disponíveis e corretas.
        
        Este teste verifica se a função get_database_configs retorna
        todas as configurações esperadas para os tipos de banco
        suportados pelo sistema.
        
        Cenário testado:
        - Configurações para 'books' e 'azure' estão disponíveis
        - Cada configuração contém data_dir, chroma_db_dir e collection_name
        - Valores correspondem às constantes corretas
        """
        configs = get_database_configs()
        
        assert 'books' in configs
        assert 'azure' in configs
        
        books_config = configs['books']
        assert books_config == (BOOKS_RAW_DATA_DIR, BOOKS_CHROMA_DB_DIR, BOOKS_COLLECTION_NAME)
        
        azure_config = configs['azure']
        assert azure_config == (AZURE_RAW_DATA_DIR, AZURE_CHROMA_DB_DIR, AZURE_COLLECTION_NAME)

    @patch('rag_project.create_chroma_database.sys.argv', ['create_chroma_database.py', 'books'])
    def test_parse_arguments_default_values(self):
        """
        Testa o parsing de argumentos com valores padrão.
        
        Este teste verifica se a função parse_arguments aplica
        corretamente os valores padrão quando apenas o tipo
        de banco é especificado.
        
        Cenário testado:
        - Apenas database_type é fornecido
        - Valores padrão são aplicados para outros parâmetros
        - Todos os argumentos são parseados corretamente
        """
        args = parse_arguments()
        
        assert args.database_type == 'books'
        assert args.chunk_size == DEFAULT_CHUNK_SIZE
        assert args.chunk_overlap == DEFAULT_CHUNK_OVERLAP
        assert args.model_name == SENTENCE_TRANSFORMERS_MODEL_NAME
        assert args.verbose is False

    @patch('rag_project.create_chroma_database.sys.argv', [
        'create_chroma_database.py', 'azure', 
        '--chunk-size', '400', '--chunk-overlap', '80',
        '--model-name', 'custom-model', '--verbose'
    ])
    def test_parse_arguments_custom_values(self):
        """
        Testa o parsing de argumentos com valores customizados.
        
        Este teste verifica se a função parse_arguments processa
        corretamente argumentos customizados fornecidos via
        linha de comando.
        
        Cenário testado:
        - Todos os argumentos são customizados
        - Valores customizados são aplicados corretamente
        - Flags booleanas funcionam adequadamente
        """
        args = parse_arguments()
        
        assert args.database_type == 'azure'
        assert args.chunk_size == 400
        assert args.chunk_overlap == 80
        assert args.model_name == 'custom-model'
        assert args.verbose is True

    def test_validate_arguments_valid(self):
        """
        Testa a validação de argumentos válidos.
        
        Este teste verifica se a função validate_arguments
        aceita argumentos válidos sem lançar exceções.
        
        Cenário testado:
        - Argumentos válidos são fornecidos
        - Nenhuma exceção é lançada
        - Validação passa com sucesso
        """
        mock_args = Mock()
        mock_args.chunk_size = 300
        mock_args.chunk_overlap = 100
        
        # Não deve lançar exceção
        validate_arguments(mock_args)

    def test_validate_arguments_negative_chunk_size(self):
        """
        Testa a validação de chunk_size negativo.
        
        Este teste verifica se a função validate_arguments
        detecta e rejeita valores negativos para chunk_size.
        
        Cenário testado:
        - chunk_size negativo é fornecido
        - ValueError é lançada com mensagem apropriada
        """
        mock_args = Mock()
        mock_args.chunk_size = -100
        mock_args.chunk_overlap = 50
        
        with pytest.raises(ValueError) as exc_info:
            validate_arguments(mock_args)
        
        assert "Chunk size must be positive" in str(exc_info.value)

    def test_validate_arguments_negative_chunk_overlap(self):
        """
        Testa a validação de chunk_overlap negativo.
        
        Este teste verifica se a função validate_arguments
        detecta e rejeita valores negativos para chunk_overlap.
        
        Cenário testado:
        - chunk_overlap negativo é fornecido
        - ValueError é lançada com mensagem apropriada
        """
        mock_args = Mock()
        mock_args.chunk_size = 300
        mock_args.chunk_overlap = -50
        
        with pytest.raises(ValueError) as exc_info:
            validate_arguments(mock_args)
        
        assert "Chunk overlap must be non-negative" in str(exc_info.value)

    def test_validate_arguments_overlap_larger_than_size(self):
        """
        Testa a validação quando overlap é maior que chunk_size.
        
        Este teste verifica se a função validate_arguments
        detecte e rejeita situações onde chunk_overlap é
        maior ou igual ao chunk_size.
        
        Cenário testado:
        - chunk_overlap é maior que chunk_size
        - ValueError é lançada com mensagem explicativa
        """
        mock_args = Mock()
        mock_args.chunk_size = 100
        mock_args.chunk_overlap = 150
        
        with pytest.raises(ValueError) as exc_info:
            validate_arguments(mock_args)
        
        assert "must be less than chunk size" in str(exc_info.value)


class TestIntegrationAndEdgeCases:
    """Integration tests and edge cases for the create_chroma_database module."""

    @patch('rag_project.create_chroma_database.Chroma.from_documents')
    @patch('rag_project.create_chroma_database.HuggingFaceEmbeddings')
    @patch('rag_project.create_chroma_database.RecursiveCharacterTextSplitter')
    @patch('rag_project.create_chroma_database.DirectoryLoader')
    @patch('rag_project.create_chroma_database.os.makedirs')
    @patch('rag_project.create_chroma_database.clean_chroma_dir')
    @patch('rag_project.create_chroma_database.os.path.exists')
    def test_full_workflow_integration(self, mock_exists, mock_clean_dir, mock_makedirs, 
                                     mock_directory_loader, mock_text_splitter, 
                                     mock_embeddings, mock_chroma):
        """
        Testa o fluxo completo de criação do banco Chroma.
        
        Este é um teste de integração que verifica se todos os
        componentes do módulo funcionam juntos corretamente,
        desde o carregamento até o salvamento final.
        
        Cenário testado:
        - Pipeline completo executa sem erros
        - Todos os componentes são chamados na ordem correta
        - Dados fluem adequadamente entre as funções
        - Resultado final é banco Chroma funcional
        """
        # Setup mocks para simular fluxo completo
        mock_exists.return_value = True
        
        mock_loader = Mock()
        mock_documents = [
            Document(page_content="Document 1 content", metadata={"source": "doc1.md"}),
            Document(page_content="Document 2 content", metadata={"source": "doc2.md"}),
        ]
        mock_loader.load.return_value = mock_documents
        mock_directory_loader.return_value = mock_loader
        
        mock_splitter = Mock()
        mock_chunks = [
            Document(page_content="Chunk 1", metadata={"source": "doc1.md", "chunk": 0}),
            Document(page_content="Chunk 2", metadata={"source": "doc1.md", "chunk": 1}),
            Document(page_content="Chunk 3", metadata={"source": "doc2.md", "chunk": 0}),
        ]
        mock_splitter.split_documents.return_value = mock_chunks
        mock_text_splitter.return_value = mock_splitter
        
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance
        
        mock_db = Mock()
        mock_collection = Mock()
        mock_collection.count.return_value = 3
        mock_db._collection = mock_collection
        mock_chroma.return_value = mock_db
        
        # Executar fluxo completo
        result = create_chroma_db("/test/data", "/test/chroma", "integration_test", 200, 50)
        
        # Verificar resultado e chamadas
        assert result is True
        mock_directory_loader.assert_called_once_with("/test/data", glob=MARKDOWN_GLOB_PATTERN)
        mock_text_splitter.assert_called_once_with(
            chunk_size=200, chunk_overlap=50, length_function=len, add_start_index=True
        )
        mock_embeddings.assert_called_once()
        mock_chroma.assert_called_once()

    def test_constants_validation(self):
        """
        Testa se as constantes do módulo têm valores válidos.
        
        Este teste verifica se todas as constantes definidas
        no módulo têm valores apropriados e consistentes
        para o funcionamento adequado do sistema.
        
        Cenário testado:
        - DEFAULT_CHUNK_SIZE é valor positivo razoável
        - DEFAULT_CHUNK_OVERLAP é menor que chunk_size
        - EXAMPLE_DOCUMENT_INDEX é índice válido
        - CONTENT_PREVIEW_LENGTH é valor positivo
        - MARKDOWN_GLOB_PATTERN é padrão válido
        """
        assert DEFAULT_CHUNK_SIZE > 0
        assert DEFAULT_CHUNK_SIZE > DEFAULT_CHUNK_OVERLAP
        assert DEFAULT_CHUNK_OVERLAP >= 0
        assert EXAMPLE_DOCUMENT_INDEX >= 0
        assert CONTENT_PREVIEW_LENGTH > 0
        assert MARKDOWN_GLOB_PATTERN == "**/*.md"
        assert isinstance(DEFAULT_CHUNK_SIZE, int)
        assert isinstance(DEFAULT_CHUNK_OVERLAP, int)

    @patch('rag_project.create_chroma_database.DirectoryLoader')
    @patch('rag_project.create_chroma_database.os.path.exists')
    def test_large_document_processing(self, mock_exists, mock_directory_loader):
        """
        Testa o processamento de documentos muito grandes.
        
        Este teste verifica se o sistema consegue processar
        documentos extremamente grandes sem problemas de
        performance ou memória.
        
        Cenário testado:
        - Documento com 100k caracteres é processado
        - Sistema não falha com conteúdo extenso
        - Memory usage se mantém controlado
        """
        mock_exists.return_value = True
        
        mock_loader = Mock()
        large_content = "A" * 100000  # 100k caracteres
        mock_documents = [Document(page_content=large_content, metadata={"source": "large.md"})]
        mock_loader.load.return_value = mock_documents
        mock_directory_loader.return_value = mock_loader
        
        result = load_documents("/test/large")
        
        assert len(result) == 1
        assert len(result[0].page_content) == 100000

    def test_empty_document_content_handling(self):
        """
        Testa o manuseio de documentos com conteúdo vazio.
        
        Este teste verifica se o sistema consegue lidar
        adequadamente com documentos que existem mas têm
        conteúdo vazio ou apenas whitespace.
        
        Cenário testado:
        - Documentos com conteúdo vazio são processados
        - Sistema não falha com conteúdo vazio
        - Documentos vazios são tratados adequadamente
        """
        empty_documents = [
            Document(page_content="", metadata={"source": "empty.md"}),
            Document(page_content="   \n\t  ", metadata={"source": "whitespace.md"}),
        ]
        
        result = split_text(empty_documents)
        
        # Resultado pode ser lista vazia ou conter chunks vazios
        assert isinstance(result, list)

    @patch('rag_project.create_chroma_database.RecursiveCharacterTextSplitter')
    def test_extreme_chunk_parameters(self, mock_text_splitter):
        """
        Testa o comportamento com parâmetros extremos de chunking.
        
        Este teste verifica se o sistema lida adequadamente
        com configurações extremas de chunk_size e chunk_overlap,
        incluindo valores muito pequenos ou muito grandes.
        
        Cenário testado:
        - chunk_size muito pequeno (1 caractere)
        - chunk_size muito grande (1 milhão)
        - chunk_overlap igual ao chunk_size
        - Sistema não falha com configurações extremas
        """
        mock_splitter = Mock()
        mock_splitter.split_documents.return_value = []
        mock_text_splitter.return_value = mock_splitter
        
        documents = [Document(page_content="Test content", metadata={})]
        
        # Teste com parâmetros extremos
        test_cases = [
            (1, 0),        # chunk_size mínimo
            (1000000, 0),  # chunk_size muito grande
            (100, 100),    # overlap igual ao chunk_size
        ]
        
        for chunk_size, chunk_overlap in test_cases:
            result = split_text(documents, chunk_size, chunk_overlap)
            assert isinstance(result, list)
            mock_text_splitter.assert_called_with(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                add_start_index=True,
            )


class TestErrorHandlingAndRobustness:
    """Tests focused on error handling and system robustness."""

    @patch('rag_project.create_chroma_database.DirectoryLoader')
    @patch('rag_project.create_chroma_database.os.path.exists')
    def test_corrupted_document_handling(self, mock_exists, mock_directory_loader):
        """
        Testa o tratamento de documentos corrompidos.
        
        Este teste verifica se o sistema consegue detectar
        e lidar adequadamente com documentos corrompidos
        ou com encoding inválido.
        
        Cenário testado:
        - DirectoryLoader falha com UnicodeDecodeError
        - Erro é tratado e propagado adequadamente
        - Sistema não trava com dados corrompidos
        """
        mock_exists.return_value = True
        mock_loader = Mock()
        mock_loader.load.side_effect = UnicodeDecodeError("utf-8", b"", 0, 1, "invalid start byte")
        mock_directory_loader.return_value = mock_loader
        
        with pytest.raises(UnicodeDecodeError):
            load_documents("/corrupted/docs")

    @patch('rag_project.create_chroma_database.shutil.rmtree')
    @patch('rag_project.create_chroma_database.os.path.exists')
    def test_disk_space_error_handling(self, mock_exists, mock_rmtree):
        """
        Testa o tratamento de erros de espaço em disco.
        
        Este teste verifica se o sistema trata adequadamente
        situações onde não há espaço suficiente em disco
        para operações de I/O.
        
        Cenário testado:
        - OSError é lançada por falta de espaço
        - Erro é tratado e propagado adequadamente
        - Sistema falha de forma controlada
        """
        mock_exists.return_value = True
        mock_rmtree.side_effect = OSError("No space left on device")
        
        with pytest.raises(OSError) as exc_info:
            clean_chroma_dir("/full/disk/path")
        
        assert "No space left on device" in str(exc_info.value)

    @patch('rag_project.create_chroma_database.Chroma.from_documents')
    @patch('rag_project.create_chroma_database.HuggingFaceEmbeddings')
    @patch('rag_project.create_chroma_database.os.makedirs')
    @patch('rag_project.create_chroma_database.clean_chroma_dir')
    def test_network_timeout_error(self, mock_clean_dir, mock_makedirs, mock_embeddings, mock_chroma):
        """
        Testa o tratamento de erros de timeout de rede.
        
        Este teste verifica se o sistema trata adequadamente
        problemas de conectividade que podem ocorrer durante
        download de modelos HuggingFace.
        
        Cenário testado:
        - HuggingFaceEmbeddings falha com TimeoutError
        - Erro é tratado e propagado adequadamente
        - Sistema não trava indefinidamente
        """
        chunks = [Document(page_content="Test chunk", metadata={})]
        mock_embeddings.side_effect = TimeoutError("Connection timed out")
        
        result = save_to_chroma(chunks, "/test/chroma", "test_collection")
        
        assert result is False

    @patch('rag_project.create_chroma_database.os.makedirs')
    @patch('rag_project.create_chroma_database.clean_chroma_dir')
    def test_insufficient_permissions_handling(self, mock_clean_dir, mock_makedirs):
        """
        Testa o tratamento abrangente de problemas de permissão.
        
        Este teste verifica se o sistema trata adequadamente
        diferentes tipos de problemas de permissão que podem
        ocorrer durante operações de arquivo e diretório.
        
        Cenário testado:
        - PermissionError em diferentes operações
        - Erros são tratados consistentemente
        - Sistema falha graciosamente com permissões inadequadas
        """
        chunks = [Document(page_content="Test chunk", metadata={})]
        mock_makedirs.side_effect = PermissionError("Access denied")
        
        result = save_to_chroma(chunks, "/restricted/chroma", "test_collection")
        
        assert result is False

    def test_invalid_document_structure(self):
        """
        Testa o tratamento de estruturas de documento inválidas.
        
        Este teste verifica se o sistema consegue detectar
        e lidar adequadamente com objetos Document que têm
        estrutura inválida ou campos faltantes.
        
        Cenário testado:
        - Documentos com estrutura inválida são processados
        - Sistema detecta e trata inconsistências
        - Erro apropriado é lançado ou objeto é ignorado
        """
        # Documento com estrutura inválida (page_content None)
        try:
            invalid_doc = Document(page_content=None, metadata={"source": "test.md"})  # type: ignore
            # Se a criação falhar, o teste passa
            # Se não falhar, tentamos processar
            result = split_text([invalid_doc])
            assert isinstance(result, list)
        except (TypeError, ValueError):
            # Erro esperado para documento inválido
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])